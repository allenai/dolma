from abc import abstractmethod
from typing import Dict, Optional, Sequence, Type

import msgspec
import regex


class BaseLicenseExtractor:
    @abstractmethod
    def __call__(self, content: str) -> Sequence["License"]:
        pass


class License(msgspec.Struct):
    name: str
    conf: float = 1.0
    vers: Optional[float] = None
    lang: Optional[str] = None
    text: Optional[str] = None

    def __post_init__(self):
        # Optimize storage by rounding to 3 decimal places
        self.conf = round(self.conf, 3)


class CreativeCommonsRegexLicenseExtractor(BaseLicenseExtractor):
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

    def __init__(self):
        self.has_type_group = "type" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex
        self.has_version_group = "version" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex
        self.has_lang_group = "lang" in self.RE_LICENSE_ATTRIBUTE_PATTERN.groupindex

        if not self.has_type_group:
            raise ValueError("License regex must have a `type` group")

    def __call__(self, content: str) -> Sequence["License"]:
        if self.PRE_REGEX_SEARCH not in content:
            return []

        licenses = []
        for i, match in enumerate(self.RE_LICENSE_ATTRIBUTE_PATTERN.finditer(content)):
            type_ = match.group("type")
            if self.has_version_group:
                version = float(v.strip("/")) if (v := match.group("version")) is not None else v
            else:
                version = None

            if self.has_lang_group:
                lang = n.strip() if (n := match.group("lang")) is not None else n
            else:
                lang = None
            licenses.append(
                License(
                    name=type_,
                    vers=version,
                    lang=lang,
                    text=match.group(0),
                    # if multiple license matches are found, the confidence is lowered
                    # for each match. The first match has a confidence of 1.0, the second
                    # has a confidence of 0.75, the third 0.667, the fourth 0.625, etc.
                    conf=0.5 + 0.5 / (i + 1),
                )
            )
        return licenses


class CreativeCommonsFastRegexHtmlExtractor(CreativeCommonsRegexLicenseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/FastRegexLicenceDetector.java"""  # noqa

    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        "http[s]?://creativecommons\\.org/licenses/"
        '(?P<type>by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain)["/ >]'
    )


class NullExtractor(BaseLicenseExtractor):
    def __call__(self, content: str) -> Sequence["License"]:
        return []


LICENSE_EXTRACTORS: Dict[str, Type[BaseLicenseExtractor]] = {
    "cc_regex": CreativeCommonsRegexLicenseExtractor,
    "cc_regex_fast": CreativeCommonsFastRegexHtmlExtractor,
    "null": NullExtractor,
}
