import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from ..core.data_types import DocResult, Document, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger

MIN_WORDS_PER_LINE = 3
NAUGHTY_LINES = (Path(__file__).parent / "../data/naughty_words_en.txt").absolute().open().read().splitlines()
NAUGHTY_WORDS: Set[str] = set(w for w in NAUGHTY_LINES if " " not in w)
NAUGHTY_PHRASES: Set[str] = set(w for w in NAUGHTY_LINES if " " in w)
EOL_PUNCTUATION = {".", "?", "!", '"'}


@dataclass
class C4Attributes:
    lines_with_no_ending_punctuation: List[Span]
    lines_with_too_few_words: List[Span]
    has_naughty_word: bool = False
    has_javascript: bool = False
    has_lorem_ipsum: bool = False
    has_curly_brace: bool = False
    line_count: int = 0
    character_count: int = 0

    def as_spans(self) -> List[Span]:
        spans = []
        spans.extend(self.lines_with_no_ending_punctuation)
        spans.extend(self.lines_with_too_few_words)
        if self.has_naughty_word:
            spans.append(Span(0, self.character_count, type="has_naughty_word"))
        if self.has_javascript:
            spans.append(Span(0, self.character_count, type="has_javascript"))
        if self.has_lorem_ipsum:
            spans.append(Span(0, self.character_count, type="has_lorem_ipsum"))
        if self.has_curly_brace:
            spans.append(Span(0, self.character_count, type="has_curly_brace"))
        spans.append(Span(0, self.character_count, type="line_count", score=self.line_count))
        return spans


def get_attributes(text: str) -> C4Attributes:
    attrs = C4Attributes([], [])
    attrs.character_count = len(text)
    try:
        lines = text.split("\n")
        attrs.line_count = len(lines)
        offset = 0
        for line_no in range(0, len(lines)):
            original_line = lines[line_no]
            end_offset = offset + len(original_line)
            if line_no < len(lines) - 1:
                end_offset += 1
            line = original_line.lower().strip()
            if not line.endswith((".", "?", "!", '"')):
                attrs.lines_with_no_ending_punctuation.append(
                    Span(offset, end_offset, type="lines_with_no_ending_punctuation")
                )
            words = line.split()
            if len(words) < MIN_WORDS_PER_LINE:
                attrs.lines_with_too_few_words.append(Span(offset, end_offset, type="lines_with_too_few_words"))
            if any(word in NAUGHTY_WORDS for word in words) or any(phrase in line for phrase in NAUGHTY_PHRASES):
                attrs.has_naughty_word = True
            if any(word == "javascript" for word in words):
                attrs.has_javascript = True
            if "lorem ipsum" in line:
                attrs.has_lorem_ipsum = True
            if "{" in line:
                attrs.has_curly_brace = True
            offset = end_offset
    except Exception:
        logging.exception(f"Error parsing text: {text[:200]}")

    return attrs


@TaggerRegistry.add("c4_v1")
class C4Tagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        attrs = get_attributes(doc.text)
        result = DocResult(doc=doc, spans=attrs.as_spans())
        return result


@TaggerRegistry.add("c4_v2")
class FasterC4Tagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        spans: List[Span] = []
        text = doc.text.lower()

        if "{" in text:
            spans.append(Span(0, len(doc.text), type="has_curly_brace"))

        if "lorem ipsum" in text:
            spans.append(Span(0, len(doc.text), type="has_lorem_ipsum"))

        if "javascript" in text:
            spans.append(Span(0, len(doc.text), type="has_javascript"))

        if any(word in NAUGHTY_WORDS for word in text.split()) or any(
            phrase in text for phrase in NAUGHTY_PHRASES
        ):
            spans.append(Span(0, len(doc.text), type="has_naughty_word"))

        start = count = 0
        for sent in text.split("\n"):
            end = start + len(sent)
            if end != len(text):
                # account for the newline
                end += 1

            # strip any trailing whitespace
            sent = sent.strip()

            if not sent.endswith((".", "?", "!", '"')):
                spans.append(Span(start, end, type="lines_with_no_ending_punctuation"))

            if len(sent.split()) < MIN_WORDS_PER_LINE:
                spans.append(Span(start, end, type="lines_with_too_few_words"))

            count += 1
            start = end

        spans.append(Span(0, len(doc.text), type="line_count", score=count))
        return DocResult(doc=doc, spans=spans)
