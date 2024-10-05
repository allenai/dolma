"""

Filters.

@kylel, @soldni

"""

from abc import abstractmethod
from typing import List

from .data_types import (
    DocResult,
    Document,
    DocumentWithMetadata,
    InputSpec,
    InputSpecWithMetadata,
    TaggerOutputDictType,
)

# digits after the decimal point
TAGGER_SCORE_PRECISION = 5


class BaseTagger:
    FIELDS: List[str] = ["text"]

    @classmethod
    def train(cls, *args, **kwargs):
        raise RuntimeError("This tagger does not support training")

    @classmethod
    def test(cls, *args, **kwargs):
        raise RuntimeError("This tagger does not support testing")

    @property
    def defaults(self) -> List[str]:
        """Returns the default span types for this tagger.
        If not provided, no defaults are set when creating output."""
        return []

    @abstractmethod
    def predict(self, doc: Document) -> DocResult:
        raise NotImplementedError

    def group_output(self, doc_result: DocResult) -> TaggerOutputDictType:
        tagger_output: TaggerOutputDictType = {field: [] for field in self.defaults}
        for span in doc_result.spans:
            output = (span.start, span.end, round(float(span.score), TAGGER_SCORE_PRECISION))
            tagger_output.setdefault(span.type, []).append(output)
        return tagger_output

    def tag(self, row: InputSpec) -> TaggerOutputDictType:
        """Internal function that is used by the tagger to get data"""
        doc = Document.from_spec(row)
        doc_result = self.predict(doc)
        return self.group_output(doc_result)


class BaseTaggerWithMetadata(BaseTagger):
    @abstractmethod
    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        raise NotImplementedError

    def tag(self, row: InputSpecWithMetadata) -> TaggerOutputDictType:
        """Internal function that is used by the tagger to get data"""
        doc = DocumentWithMetadata.from_spec(row)
        doc_result = self.predict(doc)
        return self.group_output(doc_result)
