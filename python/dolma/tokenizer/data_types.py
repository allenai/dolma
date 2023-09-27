from typing import List, NamedTuple

from ..core.data_types import InputSpec


__all__ = [
    "InputSpec",
    "TokenizerOutputSpec",
]


class TokenizerOutputSpec(NamedTuple):
    id: str
    src: str
    loc: int
    tokens: List[int]
    start: int
    end: int

    @classmethod
    def from_tokens(cls, id: str, src: str, loc: int, tokens: List[int]) -> "TokenizerOutputSpec":
        return cls(id=id, src=src, loc=loc, tokens=tokens, start=0, end=len(tokens))

    @classmethod
    def from_output_spec(cls, output_spec: "TokenizerOutputSpec", start: int = -1, end: int = -1) -> "TokenizerOutputSpec":
        start = start if start >= 0 else output_spec.start
        end = end if end >= 0 else output_spec.end
        return cls(id=output_spec.id, src=output_spec.src, loc=output_spec.loc, tokens=output_spec.tokens, start=start, end=end)
