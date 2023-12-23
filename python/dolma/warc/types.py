from typing import List, Union

import msgspec

from .license import License


class WarcDocumentMetadataLanguage(msgspec.Struct):
    code: str
    conf: float


class WarcDocumentMetadata(msgspec.Struct):
    content: Union[bytes, str]
    url: str
    content_type: str
    warc_date: str
    warc_filename: str
    license: License
    languages: List[WarcDocumentMetadataLanguage]


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata
