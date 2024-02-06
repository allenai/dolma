from typing import Set

import smart_open
import urllib3.util

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata
from ..core.url_blocker import UrlBlocker


class BaseUrlTagger(BaseTaggerWithMetadata):
    BLOCKLIST_PATH: str
    URL_METADATA_KEY = "url"

    def __init__(self) -> None:
        self.blocklist: Set[str] = set()

        with smart_open.open(self.BLOCKLIST_PATH) as blocklist_path:
            for ln in blocklist_path:
                if (ln := ln.strip()).startswith("#"):
                    continue
                self.blocklist.add(ln)

    def do_url_cleanup(self, url: str) -> str:
        return url.strip()

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
    BLOCKLIST_PATH = "https://dolma-artifacts.org/blocklist_utp/blocklist_utp-20240205/adult/domains"


@TaggerRegistry.add("link_blocklist_phishing_v1")
class LinkBlocklistPhishingTagger(BaseUrlTagger):
    BLOCKLIST_PATH = (
        "https://dolma-artifacts.org/blocklist_phishing_db/blocklist_phishing_db-20240205/domains.tar.gz"
    )


@TaggerRegistry.add("domain_blocklist_phishing_v1")
class DomainBlocklistPhishingTagger(BaseDomainTagger):
    BLOCKLIST_PATH = (
        "https://dolma-artifacts.org/blocklist_phishing_db/blocklist_phishing_db-20240205/domains.tar.gz"
    )


class AdbUrlTagger(BaseUrlTagger):
    def __init__(self) -> None:
        # from dolma import UrlBlocker

        self.engine = UrlBlocker.from_adb_paths(self.BLOCKLIST_PATH)

    def check_url(self, url: str) -> bool:
        return self.engine.check_network_urls(url)


@TaggerRegistry.add("oisd_small_abp_v1")
class OISDSmallAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATH = "https://dolma-artifacts.org/blocklist_oisd/blocklist_oisd-20240205/oisd_small_abp.txt.gz"


@TaggerRegistry.add("oisd_big_abp_v1")
class OISDBigAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATH = "https://dolma-artifacts.org/blocklist_oisd/blocklist_oisd-20240205/oisd_big_abp.txt.gz"


@TaggerRegistry.add("oisd_nsfw_abp_v1")
class OISDNSFWAdblockPlusTagger(AdbUrlTagger):
    BLOCKLIST_PATH = "https://dolma-artifacts.org/blocklist_oisd/blocklist_oisd-20240205/oisd_nsfw_abp.txt.gz"
