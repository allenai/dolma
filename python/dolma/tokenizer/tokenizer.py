from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from copy import deepcopy
from enum import Enum
from functools import cached_property
from itertools import chain
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import gigatoken as gt
import msgspec
import numpy as np
import smart_open
from necessary import necessary
from omegaconf import DictConfig
from tokenizers import Tokenizer as HFTokenizer

from ..core.errors import DolmaConfigError
from ..core.loggers import get_logger
from .data_types import TokenizerOutput

with necessary("transformers", soft=True) as TRANSFORMERS_AVAILABLE:
    if TYPE_CHECKING or TRANSFORMERS_AVAILABLE:
        from transformers import (  # pylint: disable=import-error # pyright: ignore
            AutoTokenizer,
        )

PathOrStr = Union[str, PathLike]

logger = get_logger(__name__)


__all__ = ["Tokenizer"]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


class TokenizerBackend(StrEnum):
    """Which library implements the fast tokenization path.

    ``huggingface`` uses the original ``tokenizers`` Rust bindings; ``gigatoken`` uses the
    newer, faster ``gigatoken`` library. Has no effect when ``use_fast=False``, since that
    always uses the slow ``transformers`` tokenizer.
    """

    huggingface = "hf"
    gigatoken = "gt"

    @classmethod
    def parse(cls, value: "str | TokenizerBackend") -> "TokenizerBackend":
        if isinstance(value, TokenizerBackend):
            return value

        aliases = {
            "hf": cls.huggingface,
            "huggingface": cls.huggingface,
            "gt": cls.gigatoken,
            "gigatoken": cls.gigatoken,
        }
        normalized = str(value).strip().lower()
        if normalized not in aliases:
            raise DolmaConfigError(
                f"Unknown tokenizer backend '{value}'; expected one of: hf, huggingface, gt, gigatoken."
            )
        return aliases[normalized]


class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around a HuggingFace :class:`tokenizers.Tokenizer`.

    :param base_tokenizer: The :class:`tokenizers.Tokenizer` to use.
    :param eos_token_id: The token ID corresponding to the "end-of-sentence" token.
    :param truncate_to: Truncate when tokenizing to this number of token IDs.
    :param truncate_direction: The direction to truncate in. "right" means truncate the tokens
        on the right. "left" means truncate the tokens on the left. If ``truncate_to`` is null,
        this setting has no effect.
    """

    def __init__(
        self,
        base_tokenizer: Any,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        truncate_to: Optional[int] = None,
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,
        segment_before_tokenization: bool = False,
        encode_special_tokens: bool = False,
    ):
        self.base_tokenizer = base_tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        # Gigatoken uses the HuggingFace tokenizer.json format directly, but
        # returns token IDs rather than HuggingFace Encoding objects.
        if isinstance(self.base_tokenizer, gt.Tokenizer):
            self.backend: Optional[TokenizerBackend] = TokenizerBackend.gigatoken
        elif isinstance(self.base_tokenizer, HFTokenizer):
            self.backend = TokenizerBackend.huggingface
        else:
            # slow, transformers-backed tokenizer
            self.backend = None
        self.is_fast = self.backend is not None

        if self.pad_token_id is None:
            logger.warning("No pad token ID provided; using EOS token ID %s.", eos_token_id)
            self.pad_token_id = eos_token_id

        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)
        self.segment_before_tokenization = segment_before_tokenization

        self.config = self.get_base_tokenizer_config()
        self.special_token_ids = {
            token["id"] for token in self.config.get("added_tokens", []) if token.get("special", False)
        }
        self.special_tokens = tuple(
            token["content"] for token in self.config.get("added_tokens", []) if token.get("special", False)
        )
        self.dtype = np.min_scalar_type(self.vocab_size - 1)
        self.encode_special_tokens = encode_special_tokens

    @property
    def encode_special_tokens(self) -> bool:
        return bool(getattr(self, "_encode_special_tokens", False))

    @encode_special_tokens.setter
    def encode_special_tokens(self, value: bool):
        self._encode_special_tokens = value
        if self.backend == TokenizerBackend.huggingface:
            self.base_tokenizer.encode_special_tokens = value  # pyright: ignore

    @cached_property
    def tokenizer_has_prefix(self) -> bool:
        """Returns true if the tokenizer has a prefix space. Used to determine if we need to add a space before
        tokenizer when segment_before_tokenization is True."""

        # if the tokenizer adds a prefix space, we much return True
        pretokenizer_config: dict = self.config.get("pre_tokenizer") or {}
        if pretokenizer_config.get("type") == "Sequence":
            # it's a sequence of pretokenizers, so we gotta check each one
            for pretok in pretokenizer_config.get("pretokenizers", []):
                if pretok.get("add_prefix_space", False):
                    return True
        elif pretokenizer_config.get("add_prefix_space", False):
            # this covers the case where the pretokenizer is a single pretokenizer
            return True

        # check if the normalizer or one of the components of the normalizer appends a prefix
        normalizer_config: dict = self.config.get("normalizer") or {}
        if normalizer_config.get("type") == "Sequence":
            # it's a sequence of normalizers, so we gotta check each one
            for norm in normalizer_config.get("normalizers", []):
                if norm.get("type", None) == "Prepend":
                    return True
        elif normalizer_config.get("type", None) == "Prepend":
            # this covers the case where the normalizer is a single normalizer
            return True

        # all checks above failed, so we return False
        return False

    @cached_property
    def _hf_special_token_fallback(self) -> HFTokenizer:
        """Preserve HF's opt-out behavior for recognized special-token text."""
        return HFTokenizer.from_str(json.dumps(self.config))

    def get_base_tokenizer_config(self) -> dict:
        if self.backend == TokenizerBackend.gigatoken:
            return self.base_tokenizer._hf_config()

        # Rust HuggingFace tokenizers don't have a way to get the full configuration through Python bindings,
        # so we hack around it by saving the tokenizer to a temporary file and reading the config.
        with TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/tokenizer"
            self.save(config_path)
            if not self.is_fast:
                config_path += "/tokenizer_config.json"

            with open(config_path, mode="r", encoding="utf-8") as f:
                return json.load(f)

    @property
    def vocab_size(self) -> int:
        if self.backend == TokenizerBackend.huggingface:
            return self.base_tokenizer.get_vocab_size()
        return self.base_tokenizer.vocab_size  # pyright: ignore

    @classmethod
    def from_train_config(cls, config: DictConfig) -> "Tokenizer":
        tokenizer_identifier = config.tokenizer.identifier
        if Path(tokenizer_identifier).is_file():
            tokenizer = cls.from_file(
                tokenizer_identifier,
                eos_token_id=config.model.eos_token_id,
                pad_token_id=config.model.pad_token_id,
            )
        else:
            tokenizer = cls.from_pretrained(
                tokenizer_identifier,
                eos_token_id=config.model.eos_token_id,
                pad_token_id=config.model.pad_token_id,
            )
        if config.model.vocab_size != tokenizer.vocab_size:
            raise DolmaConfigError("vocab size mismatch between config and tokenizer")
        return tokenizer

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        use_fast: bool = True,
        backend: "str | TokenizerBackend" = TokenizerBackend.huggingface,
        **kwargs,
    ) -> "Tokenizer":
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param use_fast: Whether to use a fast (Rust-backed) tokenizer. If False, ``backend`` is
            ignored and the slow ``transformers`` tokenizer is used instead.
        :param backend: Which fast tokenizer implementation to use: ``"hf"``/``"huggingface"``
            (default) or ``"gt"``/``"gigatoken"``. Only relevant when ``use_fast`` is True.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        if use_fast:
            if TokenizerBackend.parse(backend) == TokenizerBackend.gigatoken:
                base_tokenizer = gt.Tokenizer(identifier)
            else:
                base_tokenizer = HFTokenizer.from_pretrained(identifier)
        else:
            assert TRANSFORMERS_AVAILABLE, "Cannot use slow tokenizers without transformers library installed."
            base_tokenizer = AutoTokenizer.from_pretrained(identifier, use_fast=False)
            cls._check_slow_kwargs(base_tokenizer, kwargs)  # pyright: ignore

        return cls(base_tokenizer=base_tokenizer, **kwargs)  # pyright: ignore

    def save(self, filename: PathOrStr) -> None:
        """Save the tokenizer to a file."""
        if self.backend == TokenizerBackend.gigatoken:
            with open(filename, mode="w", encoding="utf-8") as f:
                json.dump(self.base_tokenizer._hf_config(), f)
        elif self.backend == TokenizerBackend.huggingface:
            self.base_tokenizer.save(filename)  # pyright: ignore
        else:
            assert TRANSFORMERS_AVAILABLE, "Cannot save slow tokenizers without transformers library installed."
            self.base_tokenizer.save_pretrained(filename)  # pyright: ignore

    @classmethod
    def _check_slow_kwargs(cls, tokenizer: "AutoTokenizer", kwargs: dict) -> None:
        if tokenizer.bos_token_id != (id_ := kwargs.get("bos_token_id", None)):  # pyright: ignore
            logger.warning("bos_token_id mismatch: %s != %s", tokenizer.bos_token_id, id_)  # pyright: ignore
        if tokenizer.eos_token_id != (id_ := kwargs.get("eos_token_id", None)):  # pyright: ignore
            logger.warning("eos_token_id mismatch: %s != %s", tokenizer.eos_token_id, id_)  # pyright: ignore
        if tokenizer.pad_token_id != (id_ := kwargs.get("pad_token_id", None)):  # pyright: ignore
            logger.warning("pad_token_id mismatch: %s != %s", tokenizer.pad_token_id, id_)  # pyright: ignore

    @classmethod
    def from_file(
        cls,
        filename: PathOrStr,
        use_fast: bool = True,
        backend: "str | TokenizerBackend" = TokenizerBackend.huggingface,
        **kwargs,
    ) -> "Tokenizer":
        """
        Initialize a tokenizer from a file.

        You can create those files with :meth:`Tokenizer.save`.

        :param filename: The name of a file containing a tokenizer specification.
        :param use_fast: Whether to use a fast (Rust-backed) tokenizer. If False, ``backend`` is
            ignored and the slow ``transformers`` tokenizer is used instead.
        :param backend: Which fast tokenizer implementation to use: ``"hf"``/``"huggingface"``
            (default) or ``"gt"``/``"gigatoken"``. Only relevant when ``use_fast`` is True.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        if use_fast:
            if TokenizerBackend.parse(backend) == TokenizerBackend.gigatoken:
                base_tokenizer = gt.Tokenizer(filename)
            else:
                # unlike gigatoken, the tokenizers Rust bindings require a str, not a PathLike
                base_tokenizer = HFTokenizer.from_file(str(filename))
        else:
            assert TRANSFORMERS_AVAILABLE, "Cannot use slow tokenizers without transformers library installed."
            base_tokenizer = AutoTokenizer.from_pretrained(filename, use_fast=False)
            cls._check_slow_kwargs(base_tokenizer, kwargs)  # pyright: ignore

        return cls(base_tokenizer=base_tokenizer, **kwargs)  # pyright: ignore

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if not input_ids:
            return input_ids

        if self.bos_token_id is not None and input_ids[0] != self.bos_token_id:
            input_ids.insert(0, self.bos_token_id)

        if self.eos_token_id is not None and input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)

        return input_ids

    def num_special_tokens_to_add(self) -> int:
        return (1 if self.eos_token_id is not None else 0) + (1 if self.bos_token_id is not None else 0)

    def _truncate(
        self, input_ids: List[int], truncate_to: Optional[int], direction: TruncationDirection
    ) -> list[int]:
        if truncate_to is None or len(input_ids) <= truncate_to:
            return input_ids
        elif direction == TruncationDirection.left:
            return input_ids[len(input_ids) - truncate_to :]
        else:
            return input_ids[: -(len(input_ids) - truncate_to)]

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.
        """
        return self.encode_batch([input], add_special_tokens=add_special_tokens)[0]

    def split_into_paragraphs(self, inputs: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
        slices = []
        batch = []
        curr = 0
        for input_ in inputs:
            paragraphs = [
                # if a tokenizer adds a prefix in front of sequences, then the tokenization of the first
                # symbol in each paragraph will be different depending on whether paragraphs are split
                # before tokenization or not. To counter this, we add a space in front of each paragraph
                # except the first one. We will remove the space from the tokenized symbols later.
                (" " if self.tokenizer_has_prefix and i > 0 else "") + match.group()
                # this regular expression keeps newlines at the beginning of paragraphs unless
                # the paragraph is the first one in the document
                for i, match in enumerate(re.finditer(r"(^\n*|\n+)[^\n]*", input_))
            ]
            slices.append((curr, curr + len(paragraphs)))
            batch.extend(paragraphs)
            curr += len(paragraphs)
        return batch, slices

    def merge_paragraphs(self, encoded: List[List[int]], slices: List[Tuple[int, int]]) -> List[List[int]]:
        merged = []
        for start, end in slices:
            encoded_slice_iter = (
                # the slicing operation is required if we have added a space in front of each paragraph
                # during the `split_into_paragraphs` method.
                encoded[pos][1:] if (self.tokenizer_has_prefix and pos > start) else encoded[pos]
                for pos in range(start, end)
            )
            merged.append(list(chain.from_iterable(encoded_slice_iter)))
        return merged

    def encode_batch(
        self,
        inputs: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add()

        if self.segment_before_tokenization:
            sliced_inputs, slice_locs = self.split_into_paragraphs(inputs)
            if self.is_fast:
                slice_encoding = self._encode_fast_batch(sliced_inputs)
            else:
                slow_seq = self.base_tokenizer(sliced_inputs, add_special_tokens=False)  # pyright: ignore
                slice_encoding = slow_seq.input_ids

            batch_encoding = self.merge_paragraphs(slice_encoding, slice_locs)
        else:
            if self.is_fast:
                batch_encoding = self._encode_fast_batch(inputs)
            else:
                slow_batch = self.base_tokenizer(
                    inputs, add_special_tokens=False, split_special_tokens=self.encode_special_tokens
                )  # pyright: ignore
                batch_encoding = slow_batch.input_ids

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = self._truncate(encoding, truncate_to, self.truncate_direction)
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)
            all_input_ids.append(input_ids)
        return all_input_ids

    def _encode_fast_batch(self, inputs: List[str]) -> List[List[int]]:
        if self.backend == TokenizerBackend.huggingface:
            fast_seq = self.base_tokenizer.encode_batch(inputs, add_special_tokens=False)  # pyright: ignore
            return [e.ids for e in fast_seq]

        # Gigatoken recognizes added special tokens by default, which matches
        # HuggingFace when ``encode_special_tokens`` is false. The opt-in flag
        # instead encodes the special-token text as ordinary text.
        if self.encode_special_tokens and any(token in text for text in inputs for token in self.special_tokens):
            fallback = self._hf_special_token_fallback
            fallback.encode_special_tokens = True
            return [encoding.ids for encoding in fallback.encode_batch(inputs)]
        return self.base_tokenizer.encode_batch(inputs).to_list()

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs to a string.
        """
        if self.backend == TokenizerBackend.gigatoken:
            if skip_special_tokens:
                token_ids = [token_id for token_id in token_ids if token_id not in self.special_token_ids]
            return self.base_tokenizer.decode(token_ids).decode("utf-8", errors="replace")
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)  # pyright: ignore


def make_tokenizer(
    tokenizer_name_or_path: str,
    **tokenizer_kwargs,
) -> Tokenizer:
    tokenizer = (
        Tokenizer.from_file(tokenizer_name_or_path, **tokenizer_kwargs)
        if os.path.exists(tokenizer_name_or_path) and os.path.isfile(tokenizer_name_or_path)
        else Tokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    )
    return tokenizer


# we need type: ignore because mypy cannot deal with recursive type aliases
NestedDict: TypeAlias = dict[str, Union[type, "NestedDict"]]  # type: ignore


def make_spec_from_fields(name: str, *fields: tuple[str, type] | None) -> type[msgspec.Struct]:
    """This function builds a msgspec.Struct from a list of field names and types.
    The field names can be nested, and the types can be nested dictionaries of types.
    """
    # first, we split the fields in components on the "." character;
    # we ignore any fields that are None; we group them into nested dictionary for shared prefixes.
    # we also need to keep track of the type of the field, so that we can use the correct decoder.

    nested_dict: NestedDict = {}
    for field_name, field_type in (f for f in fields if f is not None):
        nd = nested_dict
        *components, last_component = field_name.split(".")
        for component in components:
            nd = cast(NestedDict, nd.setdefault(component, {}))
        nd[last_component] = field_type  # pyright: ignore

    def recursively_make_struct(name: str, nested_dict: NestedDict) -> type[msgspec.Struct]:
        """This function recursively builds a msgspec.Struct from a nested dictionary of field names and types."""
        spec = []
        for k, v in nested_dict.items():
            if isinstance(v, type):
                spec.append((k, v))
            else:
                spec.append((k, recursively_make_struct(k, v)))
        return msgspec.defstruct(name, spec)

    return recursively_make_struct(name, nested_dict)


T = TypeVar("T")


def make_retriever_for_field(field_name: str, field_type: Type[T]) -> Callable[[msgspec.Struct], T]:
    if "." in field_name:
        curr, rest = field_name.split(".", 1)
        fn = make_retriever_for_field(rest, field_type)

        def retriever(spec: msgspec.Struct) -> T:
            if not hasattr(spec, curr):
                raise AttributeError(f"Field {field_name} not found in {spec}")
            return fn(getattr(spec, curr))

        return retriever
    else:

        def retriever(spec: msgspec.Struct) -> T:
            if not hasattr(spec, field_name):
                raise AttributeError(f"Field {field_name} not found in {spec}")
            return getattr(spec, field_name)

        return retriever


def tokenize_file(
    tokenizer_name_or_path: str,
    path: str,
    text_field_name: str = "text",
    text_field_type: type = str,
    id_field_name: Optional[str] = "id",
    id_field_type: type = str,
    refresh_tokenizer_every: int = 0,
    batch_size: int = 64,
    batch_max_bytes: int = 8 * 1024 * 1024,
    **tokenizer_kwargs,
) -> Generator[TokenizerOutput, None, None]:
    """Tokenize a file of documents using the provided tokenizer; file is expected to be a gzipped JSON lines
    file, each containing a field named `text`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_max_bytes <= 0:
        raise ValueError("batch_max_bytes must be positive")

    tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)
    dtype = deepcopy(tokenizer.dtype)

    spec = make_spec_from_fields(
        "TokenizerInputSpec",
        (text_field_name, text_field_type),
        ((id_field_name, id_field_type) if id_field_name else None),
    )
    text_retriever: Callable[[msgspec.Struct], str] = make_retriever_for_field(
        field_name=text_field_name, field_type=text_field_type
    )
    id_retriever: Callable[[msgspec.Struct], str] | None = (
        make_retriever_for_field(
            field_name=id_field_name,
            field_type=id_field_type,
        )
        if id_field_name
        else None
    )
    decoder = msgspec.json.Decoder(spec)
    force_refresh = False
    batch: list[tuple[str, str, int]] = []
    batch_bytes = 0

    def flush_batch() -> list[TokenizerOutput]:
        """Tokenize the pending records while retaining per-record failure handling."""
        nonlocal batch, batch_bytes
        if not batch:
            return []

        pending = batch
        batch = []
        batch_bytes = 0
        texts = [text for _, text, _ in pending]
        token_batches: list[list[int] | None]
        try:
            token_batches = list(tokenizer.encode_batch(texts, add_special_tokens=True))
        except Exception as ex:
            # Maintain the old behavior if one input makes a batch fail: log
            # and skip only failing records, not every valid record in it.
            logger.warning("Error tokenizing batch from %s: %s", path, ex)
            token_batches = []
            for row_id, text, loc in pending:
                try:
                    token_batches.append(tokenizer.encode(text, add_special_tokens=True))
                except Exception as record_ex:
                    logger.warning("Error processing line %s:%d: %s", path, loc, record_ex)
                    token_batches.append(None)

        outputs = []
        for (row_id, _, loc), maybe_tokens in zip(pending, token_batches):
            if maybe_tokens is None:
                continue
            tokens = maybe_tokens
            if refresh_tokenizer_every:
                # Extra copy to prevent memory leaks, matching the previous
                # per-document tokenization path.
                tokens = np.array(tokens, dtype=dtype)
            outputs.append(
                TokenizerOutput.from_tokens(id=row_id, src=path, loc=loc, tokens=tokens)
            )  # pyright: ignore
        return outputs

    try:
        with smart_open.open(path, mode="rt") as input_stream:
            path_hash = hashlib.sha256(path.encode()).hexdigest()
            for i, line in enumerate(input_stream, start=1):
                try:
                    row = decoder.decode(line)
                    row_id = id_retriever(row) if id_retriever else f"{path_hash}-{i}"
                    row_text = text_retriever(row)

                    if not (text := row_text.strip()):
                        # skip empty docs
                        continue

                    text_bytes = len(text.encode("utf-8"))
                    if batch and (len(batch) >= batch_size or batch_bytes + text_bytes > batch_max_bytes):
                        yield from flush_batch()

                    batch.append((row_id, text, i))
                    batch_bytes += text_bytes

                    # An oversized document is retained and emitted as a
                    # one-item batch instead of being dropped or split.
                    refresh_due = (
                        refresh_tokenizer_every > 0 and i % refresh_tokenizer_every == 0
                    ) or force_refresh
                    if len(batch) >= batch_size or batch_bytes >= batch_max_bytes or refresh_due:
                        yield from flush_batch()

                    if refresh_due:
                        # to prevent memory leaks, we refresh the tokenizer every so often
                        del tokenizer
                        gc.collect()
                        tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)

                        # we reset the flag after refreshing the tokenizer
                        force_refresh = False

                except Exception as ex:
                    # in case of failure, we log the error and continue
                    # We refresh the tokenizer to prevent memory leaks from affecting the rest of the processing
                    logger.warning("Error processing line %s:%d: %s", path, i, ex)
                    # Do not let a malformed record strand preceding valid
                    # documents in a partially-filled batch.
                    yield from flush_batch()
                    force_refresh = True
                    continue
            yield from flush_batch()
    except Exception as ex:
        # more catastrophic error, so we log the error and re-raise
        logger.error("Error processing file %s", path, exc_info=ex)
