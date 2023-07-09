"""

Taggers based on perplexity from KenLM

@kylel

"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Literal, NamedTuple, Optional, Sequence, Union

import kenlm
from cached_path import cached_path
from smashed.utils.io_utils import open_file_for_write

from .data_types import DocResult, Document, Span, TextSlice
from .taggers import BaseTagger
from .utils import split_paragraphs, split_sentences


class Prediction(NamedTuple):
    label: str
    score: float


class BaseKenLMTagger(BaseTagger):
    """Adapted from CCNet repo
    https://github.com/facebookresearch/cc_net/blob/main/cc_net/perplexity.py
    """

    SENTENCE_LEVEL_TAGGER = "sentence"
    PARAGRAPH_LEVEL_TAGGER = "paragraph"
    DOCUMENT_LEVEL_TAGGER = "document"

    def __init__(
        self,
        models: Union[Path, Dict[str, Path]],
        field: str,
        output_field: str = "perplexity",
        newline: str = "\n",
        load_method: int = 2,
    ) -> None:
        self.field = field
        self.output_field = output_field
        self.newline = newline
        self._prefetch: Sequence[str] = []
        self.lm_config = kenlm.Config()
        self.lm_config.load_method = load_method

        if isinstance(models, Path):
            self.models = {m.name.split(".")[0]: m for m in models.parent.glob(models.name)}
        else:
            self.models = models
            self._prefetch = list(models.keys())
        self.lm: Dict[str, kenlm.Model] = {}
        self.n_lines = 0

    def get_lm(self, lang: Optional[str]) -> Optional[kenlm.Model]:
        if lang is None:
            return None
        lm = self.lm.get(lang)
        if lm is not None:
            return lm
        model = self.models.get(lang)
        if model is None:
            return None
        lm = kenlm.Model(str(model), self.lm_config)
        self.lm[lang] = lm

        return lm

    def score(self, text: str, lang: Optional[str]) -> float:
        model = self.get_lm(lang)
        log_score = model.score(text)
        length = len(text.split()) + 1
        return round(self.pp(log_score, length), 1)

    @classmethod
    def pp(cls, log_score, length) -> float:
        return 10.0 ** (-log_score / length)

    def predict(self, doc: Document) -> DocResult:
        if self.mode == self.SENTENCE_LEVEL_TAGGER:
            units = split_sentences(doc.text)
        elif self.mode == self.PARAGRAPH_LEVEL_TAGGER:
            units = split_paragraphs(doc.text)
        elif self.mode == self.DOCUMENT_LEVEL_TAGGER:
            units = [TextSlice(doc=doc.text, start=0, end=len(doc.text))]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        spans = []
        for unit in units:
            for prediction in self.predict_slice(unit):
                spans.append(Span(start=unit.start, end=unit.end, type=prediction.label, score=prediction.score))

        return DocResult(doc=doc, spans=spans)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        raise NotImplementedError("Please implement the predict slice method")
