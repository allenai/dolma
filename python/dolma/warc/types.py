from typing import Optional, Sequence

import msgspec

from .license import License

MAX_DIGITS_LANG_CONF = 3
MIN_SCORE_LANG_CONF = 10**-MAX_DIGITS_LANG_CONF

MAX_DIGITS_LICENSE_CONF = 3
MIN_SCORE_LICENSE_CONF = 10**-MAX_DIGITS_LICENSE_CONF


class WarcDocumentMetadataLanguage(msgspec.Struct):
    code: str
    conf: float

    def __post_init__(self):
        # Optimize storage by rounding to 3 decimal places
        self.conf = round(self.conf, 3)


class WarcDocumentMetadata(msgspec.Struct):
    content: Optional[str]
    url: str
    norm_url: str
    content_type: str
    warc_date: str
    warc_filename: str
    licenses: Sequence[License]
    languages: Sequence[WarcDocumentMetadataLanguage]

    def __post_init__(self):
        # Only keep languages and licenses with a confidence above the threshold
        self.languages = [lang for lang in self.languages if lang.conf > MIN_SCORE_LANG_CONF]
        self.licenses = [l_ for l_ in self.licenses if l_.conf > MIN_SCORE_LICENSE_CONF]


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata
