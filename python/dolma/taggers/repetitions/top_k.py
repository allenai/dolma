"""
Taggers to detect the top K most repeated sequences in a document.

@soldni
"""

import re
from collections import Counter


from ...core.data_types import DocResult, Document, Span
from ...core.registry import TaggerRegistry
from ...core.taggers import BaseTagger


class BaseTopKTagger(BaseTagger):
    K: int
    EXPRESSION: str = r"(\W+|[A-Z]\w+)"

    def __init__(self):
        self.splitter = re.compile(self.EXPRESSION)

    def predict(self, doc: Document) -> DocResult:
        """Predict method for the tagger."""
        tokens = [ws for w in self.splitter.split(doc.text) if (ws := w.strip())]
        counter = Counter(tokens)
        spans = [
            Span(start=0, end=len(doc.text), type=k, score=v / len(tokens))
            for k, v in counter.most_common(self.K)
        ]

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("top_5_tokens")
class Top5TokensTagger(BaseTopKTagger):
    K = 5


@TaggerRegistry.add("top_10_tokens")
class Top10TokensTagger(BaseTopKTagger):
    K = 10


@TaggerRegistry.add("top_20_tokens")
class Top20TokensTagger(BaseTopKTagger):
    K = 20


@TaggerRegistry.add("top_50_tokens")
class Top50TokensTagger(BaseTopKTagger):
    K = 50


@TaggerRegistry.add("top_100_tokens")
class Top100TokensTagger(BaseTopKTagger):
    K = 100
