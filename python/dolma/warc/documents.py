from typing import Dict, List, Optional, Union

import msgspec

from ..core.taggers import TAGGER_SCORE_PRECISION

PROPERTY_MINIMUM_VALUE = 10**-TAGGER_SCORE_PRECISION


class ExtractedProperty(msgspec.Struct):
    """This is a minimal struct class that can be used to store extracted properties."""

    value: str
    type: str
    score: float
    extra: Optional[Dict[str, Union[str, float, int, None]]] = None

    def __post_init__(self):
        # Optimize storage by rounding and setting to 0 if below minimum value
        self.score = round(self.score, TAGGER_SCORE_PRECISION) if self.score >= PROPERTY_MINIMUM_VALUE else 0.0


class WarcDocumentMetadata(msgspec.Struct):
    html: Optional[str]
    url: str
    norm_url: str
    content_type: str
    warc_date: str
    warc_filename: str
    properties: List[ExtractedProperty] = []

    def __post_init__(self):
        # Only keep extractors with score > 0
        self.properties = [extractor for extractor in self.properties if extractor.score > 0.0]


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata
