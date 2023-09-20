"""

Filters.

@soldni

"""

from tokenizers import Tokenizer

from ..core.data_types import DocResult, Document, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger


class BaseTokenizer(BaseTagger):
    TOKENIZER_PATH: str

    def __init__(self) -> None:
        if not hasattr(self, "TOKENIZER_PATH"):
            raise ValueError("TOKENIZER_PATH must be defined in the subclass")

        self.tokenizer = Tokenizer.from_pretrained(self.TOKENIZER_PATH)

    def predict(self, doc: Document) -> DocResult:
        tokens = self.tokenizer.encode(sequence=doc.text, add_special_tokens=False)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="tokens", score=len(tokens))])


@TaggerRegistry.add("tokenizers_EleutherAI_GPT_NeoX_20B")
class GPTNeoX20BTokenizer(BaseTokenizer):
    TOKENIZER_PATH = "EleutherAI/gpt-neox-20B"


@TaggerRegistry.add("tokenizers_AI2_OLMo_v1")
class OLMoV1Tokenizer(BaseTokenizer):
    TOKENIZER_PATH = "allenai/eleuther-ai-gpt-neox-20b-pii-special"
