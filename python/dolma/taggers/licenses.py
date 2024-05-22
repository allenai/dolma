"""

Filters.

@kylel, @soldni

"""

import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import regex
from necessary import necessary

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata

with necessary("hyperscan", soft=True) as HYPERSCAN_AVAILABLE:
    if TYPE_CHECKING or HYPERSCAN_AVAILABLE:
        from hyperscan import Database


@TaggerRegistry.add("cc_re")
class CreativeCommonsRegexLicenseExtractor(BaseTaggerWithMetadata):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/LicenseDetectorBasic.java"""  # noqa

    PRE_REGEX = (rb"creativecommons\.org/licenses", rb"creativecommons\.org/publicdomain")
    _LICENSE_TYPE = "by(-nc)?(-nd)?(-sa)?"
    _LICENSE_VERSION = r"\d+\.\d+"
    _LICENSE_LANG_PREFIX = r"\w{2}"
    LICENSE_PATTERN = (
        "<(a|A|meta)\\s[\\w\\p{Punct}\\s=]*\n*(href|HREF|content)"
        "=('|\"|&quot;)?http(s*)://creativecommons\\.org/"
        f"((licenses/(?P<type>{_LICENSE_TYPE}))|(?P<type>publicdomain/(zero|certification|mark)))"
        f"(?P<version>/{_LICENSE_VERSION})?"
        f"((/{_LICENSE_LANG_PREFIX})?/((deed|legalcode)\\.)?(?P<lang>{_LICENSE_LANG_PREFIX}))?.*?('|\"|&quot;).*?>"
    )

    def __init__(self):
        assert HYPERSCAN_AVAILABLE, "Hyperscan is not available; please install with `pip install hyperscan`."
        self.db = Database()
        self.db.compile(
            expressions=self.PRE_REGEX,
            ids=list(range(len(self.PRE_REGEX))),
            elements=len(self.PRE_REGEX),
            flags=[0 for _ in self.PRE_REGEX],
        )

        self.license_matcher = regex.compile(self.LICENSE_PATTERN.encode("utf-8"))

        self.has_type_group = "type" in self.license_matcher.groupindex
        self.has_version_group = "version" in self.license_matcher.groupindex
        self.has_lang_group = "lang" in self.license_matcher.groupindex

        if not self.has_type_group:
            raise ValueError("License regex must have a `type` group")

        super().__init__()

    @staticmethod
    def _on_match(id_: int, from_: int, to: int, flags: int, context: Optional[Any] = None) -> None:
        if context is not None:
            context.append((id_, from_, to, flags))

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        html: Optional[bytes] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")

        content: List[Tuple[int, int, int, int]] = []
        self.db.scan(html, match_event_handler=self._on_match, context=content)
        if not content:
            return DocResult(doc=doc, spans=[])

        spans: List[Span] = []
        for i, match in enumerate(self.license_matcher.finditer(html)):
            license_string = match.group("type").decode("utf-8")
            if self.has_version_group and (version := match.group("version")) is not None:
                license_string += f"_{version.strip(b'/').decode('utf-8')}"

            if self.has_lang_group and (lang := match.group("lang")) is not None:
                license_string += f"_{lang.decode('utf-8')}"

            # get location of match in the document
            match_start, match_end = match.span()

            # if multiple license matches are found, the confidence is lowered
            # for each match. The first match has a confidence of 1.0, the second
            # has a confidence of 0.75, the third 0.667, the fourth 0.625, etc.
            score = 0.5 + 0.5 / (i + 1.0)
            spans.append(
                Span(
                    start=match_start,
                    end=match_end,
                    type=f"cc_{license_string}",
                    score=score,
                    location="metadata.html",
                )
            )

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("cc_re_fast")
class CreativeCommonsFastRegexHtmlExtractor(CreativeCommonsRegexLicenseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/FastRegexLicenceDetector.java"""  # noqa

    LICENSE_PATTERN = (
        "http[s]?://creativecommons\\.org/licenses/"
        '(?P<type>by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain)["/ >]'
    )


@TaggerRegistry.add("copyright")
class CopyrightTagger(BaseTaggerWithMetadata):
    """Extracts copyright notices from HTML documents."""

    COPYRIGHT_KEYWORDS = [
        "©",
        "Copyright",
        "版权",
        "Derechos de autor",
        "حقوق النشر",
        "Direitos autorais",
        "著作権",
        "Авторское право",
        "Urheberrecht",
        "Droit d'auteur",
        "저작권",
        "Diritto d'autore",
        "Telif hakkı",
        "Bản quyền",
        "Prawo autorskie",
        "Auteursrecht",
        "Hak cipta",
        "ลิขสิทธิ์",
        "حق نشر",
        "प्रतिलिप्यधिकार",
        "কপিরাইট",
        "Rights reserved",
        "权利保留",
        "Derechos reservados",
        "الحقوق محفوظة",
        "Direitos reservados",
        "権利を保有する",
        "Права защищены",
        "Rechte vorbehalten",
        "Droits réservés",
        "권리 보유",
        "Diritti riservati",
        "Hakları saklıdır",
        "Quyền được bảo lưu",
        "Prawa zastrzeżone",
        "Rechten voorbehouden",
        "Hak dilindungi",
        "สงวนสิทธิ์",
        "حقوق محفوظ است",
        "अधिकार सुरक्षित",
        "স্বত্ব সংরক্ষিত",
    ]

    def __init__(self):
        assert HYPERSCAN_AVAILABLE, "Hyperscan is not available; please install with `pip install hyperscan`."
        self.db = Database()

        all_expressions = [re.escape(keyword).encode("utf-8") for keyword in self.COPYRIGHT_KEYWORDS]
        all_expressions += [exp.lower() for exp in all_expressions]
        self.db.compile(
            expressions=all_expressions,
            ids=list(range(len(all_expressions))),
            elements=len(all_expressions),
            flags=[0 for _ in all_expressions],
        )

    @staticmethod
    def _on_match(id_: int, from_: int, to: int, flags: int, context: Optional[Any] = None) -> None:
        if context is not None:
            context.append(Span(start=from_, end=to, type="copyright"))

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        html: Optional[bytes] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")

        # extract copyright notices
        content: List[Span] = []
        self.db.scan(html, match_event_handler=self._on_match, context=content)
        return DocResult(doc=doc, spans=content)
