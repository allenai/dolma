import msgspec


class WarcDocumentMetadata(msgspec.Struct):
    content: bytes
    url: str
    content_type: str
    warc_date: str
    warc_filename: str


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata
