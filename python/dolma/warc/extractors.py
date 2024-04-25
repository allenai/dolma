from enum import Enum
from typing import Dict, List, Type

import regex

from ..taggers.language import (
    BaseLanguageTagger,
    Cld2LanguageTagger,
    Cld3LanguageTagger,
    FastTextAllLanguagesDocumentTagger,
    LangdetectTagger,
    LinguaTagger,
)
from .documents import ExtractedProperty
from .registries import ExtractorRegistry


class ExtractorInputType(Enum):
    PLAIN = "text/plain"
    HTML = "text/html"


class BaseExtractor:
    """Base class for extractors."""

    CONTENT_TYPE: ExtractorInputType

    def __init__(self) -> None:
        if not hasattr(self, "CONTENT_TYPE"):
            raise ValueError("Extractor must have a CONTENT_TYPE attribute")

        if not isinstance(self.CONTENT_TYPE, ExtractorInputType):
            raise ValueError("Extractor.CONTENT_TYPE must be an instance of ExtractorContentTypes")

    def extract(self, content: str) -> List[ExtractedProperty]:
        """Extract information from the raw content.

        Args:
            content: The content to extract from. Could be HTML or plain text.
        Returns:
            A list of extracted properties.
        """
        raise NotImplementedError()


def partition_extractors(extractors: List["BaseExtractor"]) -> Dict[ExtractorInputType, List["BaseExtractor"]]:
    """Partition extractors by their content type."""
    partitions: Dict[ExtractorInputType, List["BaseExtractor"]] = {
        content_type: [] for content_type in ExtractorInputType
    }
    for extractor in extractors:
        partitions[extractor.CONTENT_TYPE].append(extractor)
    return partitions


class BaseLanguageExtractor(BaseExtractor):
    """Base class for language extractors"""

    TAGGER_CLS: Type[BaseLanguageTagger]
    PROP_TYPE: str = "lang"
    CONTENT_TYPE = ExtractorInputType.PLAIN

    def __init__(self) -> None:
        self.tagger = self.TAGGER_CLS()
        super().__init__()

    def extract(self, content: str) -> List[ExtractedProperty]:
        languages = self.tagger.predict_text(content)
        return [ExtractedProperty(value=lang, score=score, type=self.PROP_TYPE) for (lang, score) in languages]


@ExtractorRegistry.add("fasttext")
class FastTextLanguageExtractor(BaseLanguageExtractor):
    tagger_cls = FastTextAllLanguagesDocumentTagger
    prop_type = "lang-fasttext"


@ExtractorRegistry.add("cld3")
class Cld3LanguageExtractor(BaseLanguageExtractor):
    tagger_cls = Cld3LanguageTagger
    prop_type = "lang-cld3"


@ExtractorRegistry.add("cld2")
class Cld2LanguageExtractor(BaseLanguageExtractor):
    tagger_cls = Cld2LanguageTagger
    prop_type = "lang-cld2"


@ExtractorRegistry.add("lingua")
class LinguaExtractor(BaseLanguageExtractor):
    tagger_cls = LinguaTagger
    prop_type = "lang-lingua"


@ExtractorRegistry.add("langdetect")
class LangdetectExtractor(BaseLanguageExtractor):
    tagger_cls = LangdetectTagger
    prop_type = "lang-langdetect"


@ExtractorRegistry.add("cc_re")
class CreativeCommonsRegexLicenseExtractor(BaseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/LicenseDetectorBasic.java"""  # noqa

    PRE_REGEX_SEARCH = "creativecommons.org/licenses"
    LICENSE_TYPE = "by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain"
    VERSION = "\\d+\\.\\d+"
    LANG_PREFIX = "\\w{2}"
    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        "<(a|A|meta)\\s[\\w\\p{Punct}\\s=]*\n*(href|HREF|content)"
        "=('|\"|&quot;)?http(s*)://creativecommons\\.org/licenses/"
        f"(?P<type>{LICENSE_TYPE})(?P<version>/{VERSION})?"
        f"(?P<lang>/{LANG_PREFIX})?/?('|\"|&quot;).*?>"
    )
    PROP_TYPE: str = "license-cc_re"
    CONTENT_TYPE = ExtractorInputType.HTML

    def __init__(self):
        self.has_type_group = "type" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex
        self.has_version_group = "version" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex
        self.has_lang_group = "lang" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex

        if not self.has_type_group:
            raise ValueError("License regex must have a `type` group")

        super().__init__()

    def extract(self, content: str) -> List[ExtractedProperty]:
        if self.PRE_REGEX_SEARCH not in content:
            return []

        licenses = []
        for i, match in enumerate(self.RE_LICENSE_ATTRIBUTE_PATTERN.finditer(content)):
            license_name = match.group("type")
            if self.has_version_group:
                version = float(v.strip("/")) if (v := match.group("version")) is not None else v
            else:
                version = None

            if self.has_lang_group:
                lang = n.strip() if (n := match.group("lang")) is not None else n
            else:
                lang = None
            licenses.append(
                ExtractedProperty(
                    value=license_name,
                    type=self.PROP_TYPE,
                    # if multiple license matches are found, the confidence is lowered
                    # for each match. The first match has a confidence of 1.0, the second
                    # has a confidence of 0.75, the third 0.667, the fourth 0.625, etc.
                    score=0.5 + 0.5 / (i + 1.0),
                    extra=dict(version=version, lang=lang, text=match.group(0)),
                )
            )
        return licenses


@ExtractorRegistry.add("cc_re_fast")
class CreativeCommonsFastRegexHtmlExtractor(CreativeCommonsRegexLicenseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/FastRegexLicenceDetector.java"""  # noqa

    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        "http[s]?://creativecommons\\.org/licenses/"
        '(?P<type>by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain)["/ >]'
    )
    PROP_TYPE = "license-cc_re_fast"
