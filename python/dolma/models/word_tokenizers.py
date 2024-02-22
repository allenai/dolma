from typing import Callable, Dict, Generator, List, Tuple, Type, TypeVar

from tokenizers import Regex, normalizers, pre_tokenizers
from typing_extensions import TypeAlias

from ..core.utils import split_paragraphs

TokensTupleType: TypeAlias = List[Tuple[str, Tuple[int, int]]]


R = TypeVar("R", bound="Type[BaseTokenizer]")


class TokenizerRegistry:
    """Singleton registry for tokenizers. This is used to register and retrieve tokenizers."""

    __registry__: Dict[str, Type["BaseTokenizer"]] = {}

    def __new__(cls):
        # Singleton pattern
        return cls

    @classmethod
    def add(cls, name: str) -> Callable[[R], R]:
        """Decorator to register a tokenizer."""

        def decorator(tok_cls: R) -> R:
            if not issubclass(tok_cls, BaseTokenizer):
                raise TypeError(f"{tok_cls} must be a subclass of BaseTokenizer")
            cls.__registry__[name] = tok_cls
            return tok_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type["BaseTokenizer"]:
        """Retrieve a tokenizer by name; raises KeyError if the tokenizer is not registered."""
        if name not in cls.__registry__:
            raise KeyError(f"Unknown tokenizer: {name}; must be one of {cls.s()}")
        return cls.__registry__[name]

    @classmethod
    def s(cls) -> str:
        """Return a string representation of the available tokenizers."""
        return ", ".join((ts := sorted(cls.__registry__.keys()))[:-1]) + f", and {ts[-1]}"


class BaseTokenizer:
    norm: normalizers.Normalizer
    pretok: pre_tokenizers.PreTokenizer

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def join(self, text: str, tokens: TokensTupleType) -> str:
        """Join the tokens back into a string"""
        raise NotImplementedError

    def tokenize(self, text: str) -> TokensTupleType:
        """Tokenize the text"""
        normalized_text = self.norm.normalize_str(text)
        tokens = self.pretok.pre_tokenize_str(normalized_text)
        return tokens

    def preprocess(self, text: str) -> Generator[str, None, None]:
        """Preprocess the text, e.g. split into paragraphs, sentences, etc."""
        yield text

    def __call__(self, text: str) -> Generator[str, None, None]:
        """Tokenize the text and join the tokens back into a string"""
        for unit in self.preprocess(text=text):
            tokens = self.tokenize(text=unit)
            if tokenized_text := self.join(text=unit, tokens=tokens).strip():
                yield tokenized_text


@TokenizerRegistry.add("noop")
class NoOpTokenizer(BaseTokenizer):  # noqa: W0223
    def __call__(self, text: str) -> Generator[str, None, None]:
        yield text


@TokenizerRegistry.add("ws")
class FastTextWhitespaceTokenizer(BaseTokenizer):
    norm: normalizers.Normalizer
    pretok: pre_tokenizers.PreTokenizer

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = normalizers.Strip()
        self.pretok = pre_tokenizers.WhitespaceSplit()

    def join(self, text: str, tokens: TokensTupleType) -> str:
        return " ".join(t for t, _ in tokens)


@TokenizerRegistry.add("ws_lower")
class FastTextWhitespaceLowercaseTokenizer(FastTextWhitespaceTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase()])  # pyright: ignore


@TokenizerRegistry.add("punct")
class FastTextPunctTokenizer(FastTextWhitespaceTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pretok = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), pre_tokenizers.Whitespace()])


@TokenizerRegistry.add("punct_lower")
class FastTextPunctLowercaseTokenizer(FastTextPunctTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase()])  # pyright: ignore


@TokenizerRegistry.add("no_punct")
class FastTextRemovePunctTokenizer(FastTextWhitespaceTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = normalizers.Sequence(
            [normalizers.Strip(), normalizers.Replace(Regex(r"\p{Punct}+"), " ")]
        )  # pyright: ignore
        self.pretok = pre_tokenizers.Whitespace()


@TokenizerRegistry.add("no_punct_lower")
class FastTextRemovePunctLowercaseTokenizer(FastTextRemovePunctTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = normalizers.Sequence(
            [normalizers.Strip(), normalizers.Replace(Regex(r"\p{Punct}+"), " "), normalizers.Lowercase()]
        )  # pyright: ignore


class BaseParagraphTokenizer(BaseTokenizer):
    def __init__(self, min_length: int = 13, **kwargs) -> None:
        self.min_length = min_length
        super().__init__(**kwargs)

    def join(self, text: str, tokens: TokensTupleType) -> str:
        if len(tokens) >= self.min_length:
            return super().join(text=text, tokens=tokens)
        return ""

    def preprocess(self, text: str) -> Generator[str, None, None]:
        for paragraph in split_paragraphs(text):
            yield paragraph.text.strip()


@TokenizerRegistry.add("para_noop")
class ParagraphNoOpTokenizer(BaseParagraphTokenizer, NoOpTokenizer):  # noqa: W0223
    """Same as NoOpTokenizer, but split into paragraphs."""


@TokenizerRegistry.add("para_ws")
class ParagraphFastTextWhitespaceTokenizer(BaseParagraphTokenizer, FastTextWhitespaceTokenizer):
    """Same as FastTextWhitespaceTokenizer, but split into paragraphs."""


@TokenizerRegistry.add("para_ws_lower")
class ParagraphFastTextWhitespaceLowercaseTokenizer(BaseParagraphTokenizer, FastTextWhitespaceLowercaseTokenizer):
    """Same as FastTextWhitespaceLowercaseTokenizer, but split into paragraphs."""


@TokenizerRegistry.add("para_punct")
class ParagraphFastTextPunctTokenizer(BaseParagraphTokenizer, FastTextPunctTokenizer):
    """Same as FastTextPunctTokenizer, but split into paragraphs."""


@TokenizerRegistry.add("para_punct_lower")
class ParagraphFastTextPunctLowercaseTokenizer(BaseParagraphTokenizer, FastTextPunctLowercaseTokenizer):
    """Same as FastTextPunctLowercaseTokenizer, but split into paragraphs."""


@TokenizerRegistry.add("para_no_punct")
class ParagraphFastTextRemovePunctTokenizer(BaseParagraphTokenizer, FastTextRemovePunctTokenizer):
    """Same as FastTextRemovePunctTokenizer, but split into paragraphs."""


@TokenizerRegistry.add("para_no_punct_lower")
class ParagraphFastTextRemovePunctLowercaseTokenizer(
    BaseParagraphTokenizer, FastTextRemovePunctLowercaseTokenizer
):
    """Same as FastTextRemovePunctLowercaseTokenizer, but split into paragraphs."""
