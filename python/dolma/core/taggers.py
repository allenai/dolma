"""

Filters.

@kylel, @soldni

"""
from abc import abstractmethod
from typing import List

from .data_types import DocResult, Document, InputSpec, TaggerOutputDictType

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

    @abstractmethod
    def predict(self, doc: Document) -> DocResult:
        raise NotImplementedError

    def tag(self, row: InputSpec) -> TaggerOutputDictType:
        """Internal function that is used by the tagger to get data"""
        doc = Document(source=row.source, version=row.version, id=row.id, text=row.text)
        doc_result = self.predict(doc)

        tagger_output: TaggerOutputDictType = {}
        for span in doc_result.spans:
            output = (span.start, span.end, round(float(span.score), TAGGER_SCORE_PRECISION))
            tagger_output.setdefault(span.type, []).append(output)
        return tagger_output
