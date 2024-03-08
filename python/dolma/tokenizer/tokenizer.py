from __future__ import annotations

import json
import re
from enum import Enum
from functools import cached_property
from itertools import chain
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional, Tuple, Union

import msgspec
import smart_open
from omegaconf import DictConfig
from tokenizers import Tokenizer as BaseTokenizer

from ..core.errors import DolmaConfigError
from ..core.loggers import get_logger
from .data_types import InputSpec, TokenizerOutput

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
    ):
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.no_truncation()
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        if self.pad_token_id is None:
            logger.warning(f"No pad token ID provided; using EOS token ID {eos_token_id}.")
            self.pad_token_id = eos_token_id

        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)
        self.segment_before_tokenization = segment_before_tokenization

        self.config = self.get_base_tokenizer_config()

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
        with NamedTemporaryFile() as f:
            self.base_tokenizer.save(f.name)
            f.flush()
            f.seek(0)
            config = json.load(f)
        return config

    @property
    def vocab_size(self) -> int:
        return self.base_tokenizer.get_vocab_size()

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
    def from_pretrained(cls, identifier: str, **kwargs) -> "Tokenizer":
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        return cls(base_tokenizer=base_tokenizer, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, **kwargs) -> "Tokenizer":
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_file(filename)
        return cls(base_tokenizer=base_tokenizer, **kwargs)

    # @classmethod
    # def from_checkpoint(cls, checkpoint_dir: PathOrStr) -> "Tokenizer":
    #     """
    #     Load a tokenizer from a checkpoint.
    #     """
    #     from cached_path import cached_path

    #     # Load configs.
    #     config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))
    #     tokenizer_config = om.load(config_path).tokenizer
    #     model_config = om.load(config_path).model

    #     # Initialize tokenizer and validate vocab size.
    #     tokenizer = cls.from_pretrained(
    #         tokenizer_config.identifier,
    #         eos_token_id=model_config.eos_token_id,
    #         pad_token_id=model_config.pad_token_id,
    #     )
    #     if model_config.vocab_size != tokenizer.vocab_size:
    #         raise DolmaConfigError("vocab size mismatch between config and tokenizer")
    #     return tokenizer

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
        for input in inputs:
            paragraphs = [
                # if a tokenizer adds a prefix in front of sequences, then the tokenization of the first
                # symbol in each paragraph will be different depending on whether paragraphs are split
                # before tokenization or not. To counter this, we add a space in front of each paragraph
                # except the first one. We will remove the space from the tokenized symbols later.
                (" " if self.tokenizer_has_prefix and i > 0 else "") + match.group()
                # this regular expression keeps newlines at the beginning of paragraphs unless
                # the paragraph is the first one in the document
                for i, match in enumerate(re.finditer(r"(^\n*|\n+)[^\n]*", input))
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

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add()

        if self.segment_before_tokenization:
            sliced_inputs, slice_locs = self.split_into_paragraphs(inputs)
            sliced_batch_encoding = [
                e.ids for e in self.base_tokenizer.encode_batch(sliced_inputs, add_special_tokens=False)
            ]
            batch_encoding = self.merge_paragraphs(sliced_batch_encoding, slice_locs)
        else:
            batch_encoding = [e.ids for e in self.base_tokenizer.encode_batch(inputs, add_special_tokens=False)]

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


def tokenize_file(tokenizer: Tokenizer, path: str) -> Generator[TokenizerOutput, None, None]:
    """Tokenize a file of documents using the provided tokenizer; file is expected to be a gzipped JSON lines
    file, each containing a field named `text`.
    """
    decoder = msgspec.json.Decoder(InputSpec)
    with smart_open.open(path, mode="rt") as input_stream:
        for i, line in enumerate(input_stream, start=1):
            try:
                row = decoder.decode(line)
                if text := row.text.strip():
                    # skip empty docs
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    yield TokenizerOutput.from_tokens(id=row.id, src=path, loc=i, tokens=tokens)
                i += 1
            except Exception as ex:
                logger.error("Error processing %s:%d", path, i, exc_info=ex)
