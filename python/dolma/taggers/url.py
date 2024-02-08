import re
from typing import Generator, List, Set

import smart_open
import urllib3.util

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata
from ..core.url_blocker import UrlBlocker


class BaseUrlTagger(BaseTaggerWithMetadata):
    BLOCKLIST_PATHS: List[str]
    URL_METADATA_KEY = "url"

    def __init__(self) -> None:
        self.blocklist: Set[str] = set()

        for blocklist_path in self.BLOCKLIST_PATHS:
            with smart_open.open(blocklist_path) as blocklist_file:
                for ln in blocklist_file:
                    try:
                        for url in self.parse_line(ln):
                            self.blocklist.add(url)
                    except ValueError as error:
                        breakpoint()
                        print(error)

        assert len(self.blocklist) > 0, f"Blocklist is empty for {self.__class__.__name__} tagger"

    def parse_line(self, ln: str) -> Generator[str, None, None]:
        if not (ln := ln.strip().lower()) or ln.startswith("#"):
            # either empty or a comment
            return
        if expr := re.match(r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) (([a-z0-9-]+\.?){2,})", ln):
            # the line contains both an IP and a URL; we yield both
            yield expr.group(1)
            yield expr.group(2)
        elif expr := re.match(r"^(([a-z0-9-]+\.?){2,})", ln):
            # the line contains only a URL; we yield it
            yield ln
        else:
            raise ValueError(f"Invalid line: {ln}")

    def do_url_cleanup(self, url: str) -> str:
        return url.strip().lower()

    def check_url(self, url: str) -> bool:
        return url in self.blocklist

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        url = doc.metadata.get(self.URL_METADATA_KEY) or ""
        cleaned_url = self.do_url_cleanup(url)
        if cleaned_url and self.check_url(cleaned_url):
            spans = [Span(start=0, end=len(doc.text), type=self.URL_METADATA_KEY, score=1.0)]
        else:
            spans = []
        return DocResult(doc=doc, spans=spans)


class BaseDomainTagger(BaseUrlTagger):
    def do_url_cleanup(self, url: str) -> str:
        hostname = urllib3.util.parse_url(url).host
        return hostname.lstrip("www.") if hostname else ""


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

        self.engine = UrlBlocker.from_adb_paths(*self.BLOCKLIST_PATHS)

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
class BlocklistProjectNsfwTagger(BaseUrlTagger):
    BLOCKLIST_PATHS = ["https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/porn.txt"]


@TaggerRegistry.add("blocklist_project_social_v1")
class BlocklistProjectSocialTagger(BaseUrlTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/facebook.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/fortnite.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/tiktok.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/twitter.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/whatsapp.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/youtube.txt",
    ]


@TaggerRegistry.add("blocklist_project_crime_v1")
class BlocklistProjectCrimeTagger(BaseUrlTagger):
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
class BlocklistProjectViceTagger(BaseUrlTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/crypto.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/drugs.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/gambling.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/vaping.txt",
    ]


@TaggerRegistry.add("blocklist_project_ads_v1")
class BlocklistProjectAdsTagger(BaseUrlTagger):
    BLOCKLIST_PATHS = [
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/adobe.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/ads.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/basic.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/smart-tv.txt",
        "https://dolma-artifacts.org/blocklist_project/blocklist_project-20240207/tracking.txt",
    ]
