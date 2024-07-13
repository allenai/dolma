from __future__ import annotations

import gc
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
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple, Union

import msgspec
import numpy as np
import smart_open
from necessary import necessary
from omegaconf import DictConfig
from tokenizers import Tokenizer as BaseTokenizer

from ..core.errors import DolmaConfigError
from ..core.loggers import get_logger
from .data_types import InputSpec, TokenizerOutput

with necessary("transformers", soft=True) as TRANSFORMERS_AVAILABLE:
    if TYPE_CHECKING or TRANSFORMERS_AVAILABLE:
        from transformers import AutoTokenizer  # pylint: disable=import-error

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
        base_tokenizer: BaseTokenizer,
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
        self.is_fast = isinstance(self.base_tokenizer, BaseTokenizer)

        if self.pad_token_id is None:
            logger.warning("No pad token ID provided; using EOS token ID %s.", eos_token_id)
            self.pad_token_id = eos_token_id

        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)
        self.segment_before_tokenization = segment_before_tokenization

        self.config = self.get_base_tokenizer_config()
        self.dtype = np.min_scalar_type(self.vocab_size - 1)
        self.encode_special_tokens = encode_special_tokens

    @property
    def encode_special_tokens(self) -> bool:
        return bool(getattr(self, "_encode_special_tokens", False))

    @encode_special_tokens.setter
    def encode_special_tokens(self, value: bool):
        self._encode_special_tokens = value
        if self.is_fast:
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

    def get_base_tokenizer_config(self) -> dict:
        # Rust HuggingFace tokenizers don't have a way to get the full configuration through Python bindings,
        # so we hack around it by saving the tokenizer to a temporary file and reading the config.

        with TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/tokenizer"
            self.save(config_path)
            if not self.is_fast:
                config_path += "/tokenizer_config.json"

            with open(config_path, mode="r", encoding="utf-8") as f:
                config = json.load(f)

        return config

    @property
    def vocab_size(self) -> int:
        if self.is_fast:
            return self.base_tokenizer.get_vocab_size()
        else:
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
    def from_pretrained(cls, identifier: str, use_fast: bool = True, **kwargs) -> "Tokenizer":
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        if use_fast:
            base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        else:
            assert TRANSFORMERS_AVAILABLE, "Cannot use slow tokenizers without transformers library installed."
            base_tokenizer = AutoTokenizer.from_pretrained(identifier, use_fast=False)
            cls._check_slow_kwargs(base_tokenizer, kwargs)  # pyright: ignore

        return cls(base_tokenizer=base_tokenizer, **kwargs)  # pyright: ignore

    def save(self, filename: PathOrStr) -> None:
        """Save the tokenizer to a file."""
        if self.is_fast:
            self.base_tokenizer.save(filename)
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
    def from_file(cls, filename: PathOrStr, use_fast: bool = True, **kwargs) -> "Tokenizer":
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        if use_fast:
            base_tokenizer = BaseTokenizer.from_file(filename)
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
                encoded[pos][1:] if (self.tokenizer_has_prefix and pos > 0) else encoded[pos]
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
                fast_seq = self.base_tokenizer.encode_batch(sliced_inputs, add_special_tokens=False)
                slice_encoding = [e.ids for e in fast_seq]
            else:
                slow_seq = self.base_tokenizer(sliced_inputs, add_special_tokens=False)  # pyright: ignore
                slice_encoding = slow_seq.input_ids

            batch_encoding = self.merge_paragraphs(slice_encoding, slice_locs)
        else:
            if self.is_fast:
                fast_batch = self.base_tokenizer.encode_batch(inputs, add_special_tokens=False)
                batch_encoding = [e.ids for e in fast_batch]
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

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs to a string.
        """
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


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


def tokenize_file(
    tokenizer_name_or_path: str,
    path: str,
    refresh_tokenizer_every: int = 0,
    **tokenizer_kwargs,
) -> Generator[TokenizerOutput, None, None]:
    """Tokenize a file of documents using the provided tokenizer; file is expected to be a gzipped JSON lines
    file, each containing a field named `text`.
    """
    tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)
    dtype = deepcopy(tokenizer.dtype)
    decoder = msgspec.json.Decoder(InputSpec)
    with smart_open.open(path, mode="rt") as input_stream:
        for i, line in enumerate(input_stream, start=1):
            try:
                row = decoder.decode(line)
                if text := row.text.strip():
                    # skip empty docs
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    if refresh_tokenizer_every:
                        # extra copy to prevent memory leaks
                        tokens = np.array(tokens, dtype=dtype)
                    yield TokenizerOutput.from_tokens(id=row.id, src=path, loc=i, tokens=tokens)  # pyright: ignore

                if refresh_tokenizer_every > 0 and i % refresh_tokenizer_every == 0:
                    # to prevent memory leaks, we refresh the tokenizer every so often
                    del tokenizer
                    gc.collect()
                    tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)

            except Exception as ex:
                logger.error("Error processing %s:%d", path, i, exc_info=ex)
