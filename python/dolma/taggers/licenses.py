"""

Filters.

@kylel, @soldni

"""

from typing import List, Optional

import regex

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata


@TaggerRegistry.add("cc_re")
class CreativeCommonsRegexLicenseExtractor(BaseTaggerWithMetadata):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/LicenseDetectorBasic.java"""  # noqa

    PRE_REGEX_SEARCH = ("creativecommons.org/licenses", "creativecommons.org/publicdomain")
    LICENSE_TYPE = "by(-nc)?(-nd)?(-sa)?"
    VERSION = r"\d+\.\d+"
    LANG_PREFIX = r"\w{2}"
    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        "<(a|A|meta)\\s[\\w\\p{Punct}\\s=]*\n*(href|HREF|content)"
        "=('|\"|&quot;)?http(s*)://creativecommons\\.org/"
        f"((licenses/(?P<type>{LICENSE_TYPE}))|(?P<type>publicdomain/(zero|certification|mark)))"
        f"(?P<version>/{VERSION})?"
        f"((/{LANG_PREFIX})?/((deed|legalcode)\\.)?(?P<lang>{LANG_PREFIX}))?.*?('|\"|&quot;).*?>"
    )

    def __init__(self):
        self.has_type_group = "type" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex
        self.has_version_group = "version" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex
        self.has_lang_group = "lang" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex

        if not self.has_type_group:
            raise ValueError("License regex must have a `type` group")

        super().__init__()

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        html: Optional[str] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")

        if not any(p in html for p in self.PRE_REGEX_SEARCH):
            return DocResult(doc=doc, spans=[])

        spans: List[Span] = []
        for i, match in enumerate(self.RE_LICENSE_ATTRIBUTE_PATTERN.finditer(html)):
            license_string = match.group("type")
            if self.has_version_group and (version := match.group("version")) is not None:
                license_string += f"_{version.strip('/')}"

            if self.has_lang_group and (lang := match.group("lang")) is not None:
                license_string += f"_{lang}"

            # if multiple license matches are found, the confidence is lowered
            # for each match. The first match has a confidence of 1.0, the second
            # has a confidence of 0.75, the third 0.667, the fourth 0.625, etc.
            score = 0.5 + 0.5 / (i + 1.0)
            spans.append(Span(start=0, end=len(doc.text), type=f"cc_{license_string}", score=score))

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("cc_re_fast")
class CreativeCommonsFastRegexHtmlExtractor(CreativeCommonsRegexLicenseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/FastRegexLicenceDetector.java"""  # noqa

    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        "http[s]?://creativecommons\\.org/licenses/"
        '(?P<type>by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain)["/ >]'
    )
