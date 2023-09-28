from typing import List, NamedTuple

from ..core.data_types import InputSpec

__all__ = ["InputSpec", "TokenizerOutput", "Metadata"]


class TokenizerOutput(NamedTuple):
    id: str
    src: str
    loc: int
    tokens: List[int]
    start: int
    end: int

    @classmethod
    def from_tokens(cls, id: str, src: str, loc: int, tokens: List[int]) -> "TokenizerOutput":
        return cls(id=id, src=src, loc=loc, tokens=tokens, start=0, end=len(tokens))

    @classmethod
    def from_output_spec(cls, output_spec: "TokenizerOutput", start: int = -1, end: int = -1) -> "TokenizerOutput":
        start = start if start >= 0 else output_spec.start
        end = end if end >= 0 else output_spec.end
        return cls(
            id=output_spec.id,
            src=output_spec.src,
            loc=output_spec.loc,
            tokens=output_spec.tokens,
            start=start,
            end=end,
        )


class Metadata(NamedTuple):
    id: str
    src: str
    loc: int
    start: int
    end: int

    def to_csv(self) -> str:
        return f"{self.id},{self.src},{self.loc},{self.start},{self.end}"


class MemmapMetadata(NamedTuple):
    start: int
    end: int
    id: str
    src: str
    loc: int
