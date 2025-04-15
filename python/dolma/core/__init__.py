from .data_types import DocResult, Document, Span
from .registry import TaggerRegistry
from .taggers import BaseTagger

# importing utils to make sure that decompressors for smart_open are registered
from .utils import add_compression  # noqa: F401

__all__ = [
    "BaseTagger",
    "DocResult",
    "Document",
    "Span",
    "TaggerRegistry",
]
