import json
import re
import socket
from typing import Generator, List, Set

import smart_open
import urllib3.util

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.loggers import get_logger
from ..core.paths import cached_path
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata
from ..core.url_blocker import UrlBlocker

LOGGER = get_logger(__name__)


def check_ipv6(n):
    """
    Check if the given string represents a valid IPv6 address.

    Args:
        n (str): The string to be checked.

    Returns:
        bool: True if the string represents a valid IPv6 address, False otherwise.
    """
    try:
        socket.inet_pton(socket.AF_INET6, n)
        return True
    except socket.error:
        return False


def check_ipv4(n):
    """
    Check if the given string represents a valid IPv4 address.

    Args:
        n (str): The string to be checked.

    Returns:
        bool: True if the string represents a valid IPv4 address, False otherwise.
    """
    try:
        socket.inet_pton(socket.AF_INET, n)
        return True
    except socket.error:
        return False


class UrlNotParsedError(ValueError):
    pass


class BaseUrlTagger(BaseTaggerWithMetadata):
    BLOCKLIST_PATHS: List[str]
    URL_METADATA_KEY = "url"
    MAYBE_IP_REGEX = re.compile(r"([0-9a-f\.\:]+)")
    IGNORE_IP_REGEX = re.compile(r"(127\.0\.0\.1|0\.0\.0\.0|::1)")
    IGNORE_IP_REGEX_START = re.compile(r"^{IGNORE_IP_REGEX.pattern}")
    URL_REGEX = re.compile(r"(([a-z0-9\-_]+\.?){2,}|localhost|localdomain)")
    ONLY_URL_REGEX = re.compile(f"^{URL_REGEX.pattern}")
    ADP_FORMAT_REGEX = re.compile(f"\\|+{URL_REGEX.pattern}\\^")
    MAYBE_IP_AND_URL_REGEX = re.compile(f"{MAYBE_IP_REGEX.pattern}\\s+{URL_REGEX.pattern}")

    def __init__(self) -> None:
        self.blocklist: Set[str] = set()

        # doing the loading here
        for blocklist_path in self.BLOCKLIST_PATHS:
            with smart_open.open(cached_path(blocklist_path)) as blocklist_file:
                for i, ln in enumerate(blocklist_file):
                    try:
                        for url in self.parse_line(ln):
                            self.blocklist.add(url)
                    except UrlNotParsedError:
                        message = f"Invalid line {i} in {blocklist_path}: '{ln}'"
                        LOGGER.info(message)

        assert len(self.blocklist) > 0, f"Blocklist is empty for {self.__class__.__name__} tagger"

    def parse_line(self, ln: str) -> Generator[str, None, None]:
        if not (ln := ln.strip().lower()) or ln.startswith("#") or ln.startswith(";") or ln.startswith("!"):
            # either empty or a comment
            return
        if expr := self.MAYBE_IP_AND_URL_REGEX.match(ln):
            # the line contains both an IP and a URL; we yield both
            maybe_ipv6_or_ipv4 = expr.group(1)
            url = expr.group(2)

            # further check if the IP is valid
            if not check_ipv6(maybe_ipv6_or_ipv4) and not check_ipv4(maybe_ipv6_or_ipv4):
                raise UrlNotParsedError(f"Invalid IP: {maybe_ipv6_or_ipv4}")

            if not self.IGNORE_IP_REGEX_START.match(maybe_ipv6_or_ipv4):
                # do not yield the IP if it a localhost
                yield maybe_ipv6_or_ipv4

            if url != "localhost" and url != "localdomain":
                # do not yield the URL if it is a localhost
                yield from self.clean_url(url)
        elif expr := self.ONLY_URL_REGEX.match(ln):
            # the line contains only a URL; we yield it
            yield from self.clean_url(ln)
        elif expr := self.ADP_FORMAT_REGEX.match(ln):
            # this is in case we need to deal with data with ADP format
            yield expr.group(1)
        else:
            raise UrlNotParsedError(f"Invalid line: {ln}")

    @classmethod
    def clean_url(cls, url: str) -> Generator[str, None, None]:
        """Remove query parameters and protocol from a URL."""
        if url is None or not url.strip():
            return

        parsed = urllib3.util.parse_url(url)
        yield f"{parsed.host}{(f':{parsed.port}') if parsed.port else ''}{parsed.path or ''}".rstrip("/").lower()

    def check_url(self, url: str) -> bool:
        return url in self.blocklist

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        url = doc.metadata.get(self.URL_METADATA_KEY) or ""
        spans = []
        for cleaned_url in self.clean_url(url):
            if self.check_url(cleaned_url):
                spans = [Span(start=0, end=len(doc.text), type=self.URL_METADATA_KEY, score=1.0)]
                break

        return DocResult(doc=doc, spans=spans)


class BaseDomainTagger(BaseUrlTagger):
    @classmethod
    def clean_url(cls, url: str) -> Generator[str, None, None]:
        if url is None or not url.strip():
            return

        for url in super().clean_url(url):
            hostname = urllib3.util.parse_url(url).host
            if not hostname:
                return
            yield (hostname := hostname.lstrip("www."))
            yield f"www.{hostname}"


@TaggerRegistry.add("domain_blocklist_utp_v1")
class DomainBlocklistUniversiteToulouseCapitoleTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_utp/blocklist_utp-20240205/adult/domains"]


@TaggerRegistry.add("link_blocklist_phishing_v1")
class LinkBlocklistPhishingTagger(BaseUrlTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_phishing_db/blocklist_phishing_db-20240205/domains.txt.gz"
    ]

    def parse_line(self, ln: str) -> Generator[str, None, None]:
        if (ln := ln.strip().lower()).startswith("#"):
            return
        yield ln


@TaggerRegistry.add("domain_blocklist_phishing_v1")
class DomainBlocklistPhishingTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_phishing_db/blocklist_phishing_db-20240205/domains.txt.gz"
    ]


class AdbUrlTagger(BaseUrlTagger):
    def __init__(self) -> None:
        # from dolma import UrlBlocker
        self.engine = UrlBlocker.from_adb_paths(*[cached_path(p) for p in self.BLOCKLIST_PATHS])

    def check_url(self, url: str) -> bool:
        return self.engine.check_network_urls(url)


@TaggerRegistry.add("oisd_small_abp_v1")
class OISDSmallAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_oisd/blocklist_oisd-20240205/oisd_small_abp.txt.gz"]


@TaggerRegistry.add("oisd_big_abp_v1")
class OISDBigAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_oisd/blocklist_oisd-20240205/oisd_big_abp.txt.gz"]


@TaggerRegistry.add("oisd_nsfw_abp_v1")
class OISDNSFWAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_oisd/blocklist_oisd-20240205/oisd_nsfw_abp.txt.gz"]


@TaggerRegistry.add("brave_core_abp_v1")
class BraveCoreAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_brave/blocklist_brave-20240206/brave_ad_block_first_party_filters.txt",
        "https://dolma-artifacts.org/blocklist_brave/blocklist_brave-20240206/brave_ad_block_updater.txt",
    ]


@TaggerRegistry.add("brave_nsfw_abp_v1")
class BraveNSFWAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_brave/blocklist_brave-20240206/blocklists_anti_porn.txt"
    ]


@TaggerRegistry.add("blocklist_project_nsfw_v1")
class BlocklistProjectNsfwTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/porn.txt"]


@TaggerRegistry.add("blocklist_project_social_v1")
class BlocklistProjectSocialTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/facebook.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/fortnite.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/tiktok.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/twitter.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/whatsapp.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/youtube.txt",
    ]


@TaggerRegistry.add("blocklist_project_crime_v1")
class BlocklistProjectCrimeTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/abuse.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/fraud.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/malware.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/phishing.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/piracy.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/ransomware.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/scam.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/redirect.txt",
    ]


@TaggerRegistry.add("blocklist_project_vice_v1")
class BlocklistProjectViceTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/crypto.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/drugs.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/gambling.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/vaping.txt",
    ]


@TaggerRegistry.add("blocklist_project_ads_v1")
class BlocklistProjectAdsTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/adobe.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/ads.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/basic.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/smart-tv.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/tracking.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_ads_v1")
class BlocklistFirebogAdsTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/blue/hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/AdguardDNS.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/Admiral.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/adservers.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/Easylist-2.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/hosts-2.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/hosts-3.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/serverlist.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/ads/green/simple_ad.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_crypto_v1")
class BlocklistFirebogCryptoTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/crypto/green/hosts_browser.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_malicious_v1")
class BlocklistFirebogMaliciousTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/blue/phishing-filter-hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/blue/Prigent-Malware.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/AntiMalwareHosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/hostfile.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/hosts-2.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/latestdomains.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/main-blacklist.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/Mandiant_APT1_Report_Appendix_D.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/notrack-malware.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/phishing_army_blocklist_extended.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/Prigent-Crypto.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/RPiList-Malware.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/RPiList-Phishing.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/malicious/green/simple_malvertising.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_nsfw_v1")
class BlocklistFirebogNsfwTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/nsfw/blue/pi_blocklist_porn_top1m.list.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/nsfw/blue/Prigent-Adult.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_social_v1")
class BlocklistFirebogSocialTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/social/blue/facebook.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_suspicious_v1")
class BlocklistFirebogSuspiciousTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/hosts-2.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/hosts-3.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/hosts-4.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/hosts-file.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/neohostsbasic.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/SNAFU.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/blue/spammers.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/green/hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/green/KADhosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/suspicious/green/w3kbl.txt",
    ]


@TaggerRegistry.add("blocklist_firebog_trackers_v1")
class BlocklistFirebogTrackersTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/blue/ads-and-tracking-extended.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/blue/AmazonFireTV.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/blue/android-tracking.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/blue/notrack-blocklist.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/blue/SmartTV.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/green/Easyprivacy.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/green/firstparty-trackers-hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/green/hosts.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/green/Prigent-Ads.txt",
        "https://dolma-artifacts.org/blocklist_firebog/blocklist_firebog-20240208/trackers/green/spy.txt",
    ]


@TaggerRegistry.add("blocklist_hosts_adware_malware_v1")
class BlocklistHostsAdwareMalwareTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_hosts/blocklist_hosts-20240208/adware_malware.txt"]


@TaggerRegistry.add("blocklist_hosts_fakenews_v1")
class BlocklistHostsFakenewsTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_hosts/blocklist_hosts-20240208/fakenews.txt"]


@TaggerRegistry.add("blocklist_hosts_gambling_v1")
class BlocklistHostsGamblingTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_hosts/blocklist_hosts-20240208/gambling.txt"]


@TaggerRegistry.add("blocklist_hosts_porn_v1")
class BlocklistHostsPornTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_hosts/blocklist_hosts-20240208/porn.txt"]


@TaggerRegistry.add("blocklist_hosts_social_v1")
class BlocklistHostsSocialTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_hosts/blocklist_hosts-20240208/social.txt"]


@TaggerRegistry.add("allowlist_wikidata_v1")
class AllowlistWikidataTagger(BaseDomainTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/wikidata/wikidata-20220208/periodical-Q1002697/response.json",
        "https://dolma-artifacts.org/wikidata/wikidata-20220208/website-Q35127/response.json",
    ]

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_base_domain(cls, url: str) -> str:
        if url.count(".") > 2:
            _, *domain_components = url.rsplit(".", 2)
            return ".".join(domain_components)
        return url

    @classmethod
    def clean_url(cls, url: str) -> Generator[str, None, None]:
        cleaned_urls = super().clean_url(url)
        for cleaned_url in cleaned_urls:
            yield cleaned_url
            yield cls.get_base_domain(cleaned_url)

    def is_valid_row(self, row: dict) -> bool:
        return True

    def parse_line(self, ln: str) -> Generator[str, None, None]:
        data = json.loads(ln)
        for row in data:
            try:
                for url in self.clean_url(row["url"]):
                    yield from super().parse_line(url)
            except Exception:
                pass

    def check_url(self, url: str) -> bool:
        for cleaned_url in self.clean_url(url):
            if cleaned_url in self.blocklist:
                return True
        return False


@TaggerRegistry.add("allowlist_wikidata_cleaned_v1")
class AllowlistWikidataCleanedTagger(AllowlistWikidataTagger):
    NSFW_WIKI_WORDS_DESC = [
        "sex",
        "adult",
        "satire",
        "adult",
        "gossip",
        "tabloid",
        "tracker",
        "dating",
        "image",
        "humor",
        "joke",
        "comedy",
        "porn",
        "social media",
        "freemium",
        "betting",
        "casino",
        "gambling",
        "celebrity",
        "4chan",
        "camming",
        "escort",
        "hentai",
        "imageboard",
        "image hosting",
        "crowdfunding",
        "nudity",
        "comic",
        "camming",
        "online database",
    ]
    NSFW_WIKI_TLDS = [
        ".xxx",
        ".adult",
        ".sex",
        ".porn",
        ".sexy",
        ".dating",
        ".cam",
        ".tube",
        ".chat",
    ]
    INCOMPLETE_WIKI_DESC = [
        "company",
        "website",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.incomplete_wiki_desc = set(self.INCOMPLETE_WIKI_DESC)
        self.nsfw_wiki_words_desc = set(self.NSFW_WIKI_WORDS_DESC)
        self.nsfw_wiki_tlds = set(self.NSFW_WIKI_TLDS)

    def is_valid_row(self, row: dict) -> bool:
        if row["description"] is None:
            return False
        if any(word in row["description"].lower() for word in self.nsfw_wiki_words_desc):
            return False
        if any(tld in row["url"] for tld in self.nsfw_wiki_tlds):
            return False
        if any(word in row["description"].lower() for word in self.incomplete_wiki_desc):
            return False
        return True
