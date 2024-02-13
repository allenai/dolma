from typing import Callable, Dict, List, Tuple, Type, TypeVar

from tokenizers import Regex, normalizers, pre_tokenizers
from typing_extensions import TypeAlias

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

    def join(self, text: str, tokens: TokensTupleType) -> str:
        """Join the tokens back into a string"""
        raise NotImplementedError

    def tokenize(self, text: str) -> TokensTupleType:
        """Tokenize the text"""
        normalized_text = self.norm.normalize_str(text)
        tokens = self.pretok.pre_tokenize_str(normalized_text)
        return tokens

    def __call__(self, text: str) -> str:
        """Tokenize the text and join the tokens back into a string"""
        tokens = self.tokenize(text)
        return self.join(text=text, tokens=tokens)


@TokenizerRegistry.add("noop")
class NoOpTokenizer(BaseTokenizer):
    def __call__(self, text: str) -> str:
        return text


@TokenizerRegistry.add("ws")
class FastTextWhitespaceTokenizer(BaseTokenizer):
    norm: normalizers.Normalizer
    pretok: pre_tokenizers.PreTokenizer

    def __init__(self) -> None:
        self.norm = normalizers.Strip()
        self.pretok = pre_tokenizers.WhitespaceSplit()

    def join(self, text: str, tokens: TokensTupleType) -> str:
        return " ".join(t for t, _ in tokens)


@TokenizerRegistry.add("ws_lower")
class FastTextWhitespaceLowercaseTokenizer(FastTextWhitespaceTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.norm = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase()])  # pyright: ignore


@TokenizerRegistry.add("punct")
class FastTextPunctTokenizer(FastTextWhitespaceTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pretok = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), pre_tokenizers.Whitespace()])


@TokenizerRegistry.add("punct_lower")
class FastTextPunctLowercaseTokenizer(FastTextPunctTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.norm = normalizers.Sequence([normalizers.Strip(), normalizers.Lowercase()])  # pyright: ignore


@TokenizerRegistry.add("no_punct")
class FastTextRemovePunctTokenizer(FastTextWhitespaceTokenizer):
    def __init__(self) -> None:
        self.norm = normalizers.Sequence(
            [normalizers.Strip(), normalizers.Replace(Regex(r"\p{Punct}+"), " ")]
        )  # pyright: ignore
        self.pretok = pre_tokenizers.Whitespace()


@TokenizerRegistry.add("no_punct_lower")
class FastTextRemovePunctLowercaseTokenizer(FastTextRemovePunctTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.norm = normalizers.Sequence(
            [normalizers.Strip(), normalizers.Replace(Regex(r"\p{Punct}+"), " "), normalizers.Lowercase()]
        )  # pyright: ignore
