from __future__ import annotations

import os
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Generator, List, Optional, Union

import msgspec
import smart_open
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf as om
from tokenizers import Tokenizer as BaseTokenizer

from ..core.errors import DolmaConfigError
from ..core.loggers import get_logger
from .data_types import InputSpec, TokenizerOutput

PathOrStr = Union[str, PathLike]

log = get_logger(__name__)


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
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        truncate_to: Optional[int] = None,
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,
    ):
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.no_truncation()
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)

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
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, **kwargs) -> "Tokenizer":
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_file(filename)
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: PathOrStr) -> "Tokenizer":
        """
        Load a tokenizer from a checkpoint.
        """
        from cached_path import cached_path

        # Load configs.
        config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))
        tokenizer_config = om.load(config_path).tokenizer
        model_config = om.load(config_path).model

        # Initialize tokenizer and validate vocab size.
        tokenizer = cls.from_pretrained(
            tokenizer_config.identifier,
            eos_token_id=model_config.eos_token_id,
            pad_token_id=model_config.pad_token_id,
        )
        if model_config.vocab_size != tokenizer.vocab_size:
            raise DolmaConfigError("vocab size mismatch between config and tokenizer")
        return tokenizer

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if not input_ids or input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1

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

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = self._truncate(encoding.ids, truncate_to, self.truncate_direction)
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
                log.error("Error processing %s:%d", path, i, exc_info=ex)
