from typing import Set
import smart_open
import urllib3.util

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata


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

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        url = doc.metadata.get(self.URL_METADATA_KEY)
        cleaned_url = self.do_url_cleanup(str(url))
        if cleaned_url and cleaned_url in self.blocklist:
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
    BLOCKLIST_PATH = "https://dolma-artifacts.org/blocklist_phishing_db/blocklist_phishing_db-20240205/domains.tar.gz"


@TaggerRegistry.add("domain_blocklist_phishing_v1")
class DomainBlocklistPhishingTagger(BaseDomainTagger):
    BLOCKLIST_PATH = "https://dolma-artifacts.org/blocklist_phishing_db/blocklist_phishing_db-20240205/domains.tar.gz"
