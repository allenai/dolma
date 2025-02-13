import regex

from ..core.data_types import DocResult, Document, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs


@TaggerRegistry.add("not_alphanum_paragraph_v1")
class NotAlphanumParagraphV1(BaseTagger):
    def __init__(self) -> None:
        self.re_has_alphanum = regex.compile(r"[a-zA-Z0-9]", regex.UNICODE)
        self.re_all_punctuation = regex.compile(
            r"^("
            r"[[:punct:]]|"
            r"\s|"
            r"["
            "\U0001f300-\U0001f64f"
            "\U0001f680-\U0001f6ff"
            "\u2600-\u26ff\u2700-\u27bf"
            r"]+"
            r")+$",
            regex.UNICODE,
        )

    def predict(self, doc: Document) -> DocResult:
        spans = []

        for para in split_paragraphs(text=doc.text):
            if self.re_has_alphanum.search(para.text):
                continue

            if self.re_all_punctuation.search(para.text):
                spans.append(Span(start=para.start, end=para.end, type="all_punct", score=1))

        if not spans:
            spans.append(Span(start=0, end=len(doc.text), type="all_punct", score=0))

        return DocResult(doc=doc, spans=spans)
