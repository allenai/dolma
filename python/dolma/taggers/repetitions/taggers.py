"""
Taggers to detect repetitions in the text.

@soldni
"""

import re
from typing import Generator

import numpy as np

from ...core.data_types import DocResult, Document, Span
from ...core.registry import TaggerRegistry
from ...core.taggers import BaseTagger
from ...core.utils import split_paragraphs


@TaggerRegistry.add("repetitions_v1")
class RepetitionsTagger(BaseTagger):
    """Tagger to detect repetitions of of groups of characters.
    Only repetitions that occur at least 4 times are detected."""

    def __init__(self) -> None:
        self.re_char_repetitions = re.compile(r"(.+?)(\s?\1){3,}")
        super().__init__()

    def _extract_from_text(self, text: str) -> Generator[Span, None, None]:
        """Extract repetitions of characters in the text."""
        for match in self.re_char_repetitions.finditer(text):
            yield Span(
                start=(start := match.start()),
                end=(end := match.end()),
                type="char_repetition",
                score=float(end - start),
            )

    def _extract_from_doc(self, doc: Document) -> Generator[Span, None, None]:
        yield from self._extract_from_text(doc.text)

    def predict(self, doc: Document) -> DocResult:
        """Predict method for the tagger."""
        char_reps_spans = list(self._extract_from_doc(doc))
        doc_max_span = Span(
            start=0,
            end=len(doc.text),
            type="doc_max_char_repetition",
            score=max(char_reps_spans, key=lambda s: s.score).score if char_reps_spans else 0.0,
        )
        doc_mean_reps_span = Span(
            start=0,
            end=len(doc.text),
            type="doc_mean_char_repetition",
            score=float(np.mean([s.score for s in char_reps_spans]) if char_reps_spans else 0),
        )
        doc_frac_reps_span = Span(
            start=0,
            end=len(doc.text),
            type="doc_frac_char_repetition",
            score=float(sum([s.score for s in char_reps_spans]) / len(doc.text) if char_reps_spans else 0),
        )
        return DocResult(doc=doc, spans=char_reps_spans + [doc_max_span, doc_mean_reps_span, doc_frac_reps_span])


@TaggerRegistry.add("paragraph_repetitions_v1")
class ParagraphRepetitionsTagger(RepetitionsTagger):
    """Tagger to detect repetitions of paragraphs.
    It's faster than the char repetition tagger, but it does not account for
    repetitions of characters that span multiple paragraphs."""

    def _extract_from_doc(self, doc: Document) -> Generator[Span, None, None]:
        offset = 0
        for paragraph in split_paragraphs(doc.text, remove_empty=False):
            for span in self._extract_from_text(paragraph.text):
                span.start += offset
                span.end += offset
                yield span
            offset += len(paragraph.text)
