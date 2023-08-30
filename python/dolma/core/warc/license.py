from abc import abstractmethod
from typing import Dict, Optional, Type, Union

import msgspec
import regex
from charset_normalizer import detect


class BaseLicenseExtractor:
    @abstractmethod
    def __call__(self, content: Union[str, bytes]) -> "License":
        pass


class License(msgspec.Struct):
    type_: str
    version: Optional[float] = None
    lang: Optional[str] = None
    text: Optional[str] = None


class CreativeCommonsRegexLicenseExtractor(BaseLicenseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/LicenseDetectorBasic.java"""  # noqa

    LICENSE_TYPE = "by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain"
    VERSION = "\\d+\\.\\d+"
    LANG_PREFIX = "\\w{2}"
    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        "<(a|A|meta)\\s[\\w\\p{Punct}\\s=]*\n*(href|HREF|content)"
        "=('|\"|&quot;)?http(s*)://creativecommons\\.org/licenses/"
        f"({LICENSE_TYPE})(/{VERSION})?(/{LANG_PREFIX})?/?('|\"|&quot;).*?>"
    )

    def __call__(self, content: Union[str, bytes]) -> License:
        if isinstance(content, bytes):
            if b"creativecommons.org/licenses" not in content:
                return License(type_="unk")

            if not (encoding := detect(content)["encoding"]):
                return License(type_="unk")
            content = content.decode(str(encoding))

        if "creativecommons.org/licenses" not in content:
            return License(type_="unk")

        for match in self.RE_LICENSE_ATTRIBUTE_PATTERN.finditer(content):
            *_, type_, version, lang, __ = match.groups()
            return License(
                type_=type_.strip("/"),
                version=float(version.strip("/")) if version else None,
                lang=lang.strip("/") if lang else None,
                text=match.group(0),
            )

        return License(type_="unk")


class CreativeCommonsFastRegexHtmlExtractor(CreativeCommonsRegexLicenseExtractor):
    """Adapted from https://github.com/dkpro/dkpro-c4corpus/blob/da61281a8a77fad0d6a7d27c06b5e2fe3282e28f/dkpro-c4corpus-license/src/main/java/de/tudarmstadt/ukp/dkpro/c4corpus/license/impl/FastRegexLicenceDetector.java"""  # noqa

    RE_LICENSE_ATTRIBUTE_PATTERN = regex.compile(
        'http[s]?://creativecommons\\.org/licenses/(by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain)["/ >]'
    )


LICENSE_EXTRACTORS: Dict[str, Type[BaseLicenseExtractor]] = {
    "cc_regex": CreativeCommonsRegexLicenseExtractor,
    "cc_regex_fast": CreativeCommonsFastRegexHtmlExtractor,
}
