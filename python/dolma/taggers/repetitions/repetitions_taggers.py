"""
Taggers to detect repetitions in the text.

@soldni
"""

import re
from abc import abstractmethod
from typing import Generator, List

import numpy as np
from tokenizers import Tokenizer

from ...core.data_types import DocResult, Document, Span
from ...core.registry import TaggerRegistry
from ...core.taggers import BaseTagger
from ...core.utils import split_paragraphs
from .utils import find_periodic_sequences


class BaseRepetitionsTagger(BaseTagger):
    @abstractmethod
    def _extract_from_text(self, text: str) -> Generator[Span, None, None]:
        raise NotImplementedError()

    def _extract_from_doc(self, doc: Document) -> Generator[Span, None, None]:
        yield from self._extract_from_text(doc.text)

    def _compute_document_stats(self, spans: List[Span], doc: Document) -> List[Span]:
        doc_max_span = Span(
            start=0,
            end=len(doc.text),
            type="doc_max_repetition",
            score=max(spans, key=lambda s: s.score).score if spans else 0.0,
        )
        doc_mean_reps_span = Span(
            start=0,
            end=len(doc.text),
            type="doc_mean_repetition",
            score=float(np.mean([s.score for s in spans]) if spans else 0),
        )
        doc_frac_reps_span = Span(
            start=0,
            end=len(doc.text),
            type="doc_frac_repetition",
            score=float(sum([s.score for s in spans]) / len(doc.text) if spans else 0),
        )
        return [doc_max_span, doc_mean_reps_span, doc_frac_reps_span]

    def predict(self, doc: Document) -> DocResult:
        """Predict method for the tagger."""
        reps_spans = list(self._extract_from_doc(doc))
        document_stats_spans = self._compute_document_stats(spans=reps_spans, doc=doc)
        return DocResult(doc=doc, spans=reps_spans + document_stats_spans)


@TaggerRegistry.add("repetitions_v1")
class RepetitionsTagger(BaseRepetitionsTagger):
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
                type="repetition",
                score=float(end - start),
            )


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


@TaggerRegistry.add("tokenizer_repetitions_v1")
class TokenizerRepetitionsTagger(BaseRepetitionsTagger):
    """Tagger to detect repetitions of tokens.

    It uses a tokenizer to split the text into tokens, and then identifies
    sequences of tokens that repeat at least 3 times."""

    TOKENIZER_IDENTIFIER = "allenai/eleuther-ai-gpt-neox-20b-pii-special"
    MIN_PERIOD = 1
    MAX_PERIOD = 13

    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_pretrained(self.TOKENIZER_IDENTIFIER)

    def _extract_from_text(self, text: str) -> Generator[Span, None, None]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        sequences_iter = find_periodic_sequences(
            arr=np.array(tokens.ids), min_period=self.MIN_PERIOD, max_period=self.MAX_PERIOD
        )
        for seq in sequences_iter:
            yield Span(
                start=(s := tokens.offsets[seq.start][0]),
                end=(e := tokens.offsets[seq.end - 1][1]),
                type="repetition",
                score=float(e - s),
            )


@TaggerRegistry.add("paragraph_tokenizer_repetitions_v1")
class ParagraphTokenizerRepetitionsTagger(TokenizerRepetitionsTagger):
    """Tagger to detect repetitions of tokens in paragraphs.
    It's faster than the tokenizer repetition tagger, but it does not account for
    repetitions of tokens that span multiple paragraphs."""

    def _extract_from_doc(self, doc: Document) -> Generator[Span, None, None]:
        offset = 0
        for paragraph in split_paragraphs(doc.text, remove_empty=False):
            # space is required to avoid first symbol in the paragraph to be
            # tokenized as a different token.
            for span in self._extract_from_text(" " + paragraph.text):
                span.start += offset - 1
                span.end += offset - 1
                yield span
            offset += len(paragraph.text)
