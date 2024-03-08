"""

Base implementation for a fasttext tagger; all fasttext taggers should inherit from this class.

@kylel, @soldni

"""

from enum import Enum
from typing import TYPE_CHECKING, Iterable, NamedTuple, Union

from cached_path import cached_path
from necessary import necessary

from ...core.data_types import DocResult, Document, Span, TextSlice
from ...core.taggers import BaseTagger
from ...core.utils import split_paragraphs, split_sentences

with necessary(("fasttext", "0.9.2"), soft=True) as FASTTEXT_AVAILABLE:
    if TYPE_CHECKING or FASTTEXT_AVAILABLE:
        from fasttext.FastText import _FastText as FastTextModel


__all__ = ["FastTextPrediction", "FastTextMode", "FastTextTagger"]


class FastTextPrediction(NamedTuple):
    label: str
    score: float


class FastTextMode(Enum):
    sentence = "sentence"
    paragraph = "paragraph"
    document = "document"


class FastTextTagger(BaseTagger):
    def __init__(self, path: str, mode: Union[str, FastTextMode]) -> None:
        if not FASTTEXT_AVAILABLE:
            raise ImportError("fasttext is not available. Install it using `pip install fasttext-wheel`")

        # we use this private attribute to avoid a warning from the fasttext library. See this comment:
        # https://github.com/facebookresearch/fastText/issues/1056#issuecomment-1278058705
        self.classifier = FastTextModel(str(cached_path(path)))
        self.mode = FastTextMode(mode)

    def predict(self, doc: Document) -> DocResult:
        if self.mode == FastTextMode.sentence:
            units = split_sentences(doc.text)
        elif self.mode == FastTextMode.paragraph:
            units = split_paragraphs(doc.text)
        elif self.mode == FastTextMode.document:
            units = [TextSlice(doc=doc.text, start=0, end=len(doc.text))]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        spans = []
        for unit in units:
            for prediction in self.predict_slice(unit):
                spans.append(Span(start=unit.start, end=unit.end, type=prediction.label, score=prediction.score))

        return DocResult(doc=doc, spans=spans)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[FastTextPrediction]:
        raise NotImplementedError("Please implement the predict slice method")
