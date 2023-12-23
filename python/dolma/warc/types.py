from typing import List, Optional

import msgspec

from .license import License


MAX_DIGITS_LANG_CONF = 3
MIN_SCORE_LANG_CONF = 10**-MAX_DIGITS_LANG_CONF


class WarcDocumentMetadataLanguage(msgspec.Struct):
    code: str
    conf: float

    def __post_init__(self):
        # Optimize storage by rounding to 3 decimal places
        self.conf = round(self.conf, 3)


class WarcDocumentMetadata(msgspec.Struct):
    content: Optional[str]
    url: str
    content_type: str
    warc_date: str
    warc_filename: str
    license: License
    languages: List[WarcDocumentMetadataLanguage]

    def __post_init__(self):
        # Only keep languages with a confidence > 0.0
        self.languages = [lang for lang in self.languages if lang.conf > MIN_SCORE_LANG_CONF]


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata
