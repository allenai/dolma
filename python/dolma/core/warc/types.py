from typing import Union
import msgspec

from .license import License


class WarcDocumentMetadata(msgspec.Struct):
    content: Union[bytes, str]
    url: str
    content_type: str
    warc_date: str
    warc_filename: str
    cc_license: License


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata
