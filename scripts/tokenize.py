"""
Use this to prepare a numpy memory-mapped language modeling dataset from raw *.json.gz
dataset files, such as those from c4. Each file is expected to be a gzipped JSON lines
file, which each JSON line has a field named "text" that is a string representing a single
document from the dataset.

To test out this script, run:

```bash
python scripts/prepare_memmap_dataset.py test_fixtures/*.json.gz -o /tmp/out.npy
```
"""

import concurrent.futures
import functools
from io import BytesIO
import itertools
import json
import logging
import multiprocessing as mp
from enum import Enum
import os
import random
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional, Sequence, Tuple, TypeVar, Union

import click
import msgspec
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from typing_extensions import TypeAlias
import smart_open

from dolma.core.data_types import InputSpec
from dolma.core.paths import glob_path

from tokenizers import Tokenizer as BaseTokenizer


PathOrStr: TypeAlias = Union[str, os.PathLike]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=Sequence)


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
    def from_pretrained(cls, identifier: str, **kwargs) -> 'Tokenizer':
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

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
    ) -> List[int]:
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


def get_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        "files",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


class OutputSpec(msgspec.Struct):
    id: str
    src: str
    loc: int
    tokens: List[int]
    start: int = -1
    end: int = -1

    def __post_init__(self):
        self.start = max(self.start, 0)
        if self.end < 0:
            self.end = len(self.tokens)


def tokenize_file(tokenizer: Tokenizer, path: str) -> Generator[OutputSpec, None, None]:
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
                    yield OutputSpec(id=row.id, src=path, loc=i, tokens=tokens)
                i += 1
            except Exception as ex:
                log.error("Error processing %s:%d", path, i, exc_info=ex)


class Metadata(msgspec.Struct):
    id: str
    src: str
    loc: int
    start: int
    end: int


class MemMapWriter:
    """Context manager responsible for writing, resizing, and closing / uploading a memmap file."""

    DEFAULT_MAX_TOKENS = 512 * 1024 * 1024  # 500M tokens / 1GB
    MEMMAP_EXTENSION = ".npy"
    METADATA_EXTENSION = ".jsonl.gz"

    def __init__(
        self,
        path: str,
        dtype: np.dtype,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Create a new memmap file.

        Args:
            path (str): Location for the memmap file. If the path is not local, the memmap file will be
                written to a temporary file first and then uploaded to the destination.
            dtype (np.dtype): Data type for the memmap file; must be a valid numpy dtype.
            max_tokens (int, optional): Maximum number of tokens per file. Defaults to 500M tokens, which is 1GB.
        """
        base_path = re.sub(r"\.[a-zA-Z]*(\.npy?)\.[a-zA-Z]*$", "", path)
        self.memmap_path = f"{base_path}{self.MEMMAP_EXTENSION}"
        self.metadata_path = f"{base_path}{self.METADATA_EXTENSION}"
        self.dtype = dtype
        self.max_tokens = max_tokens

        self._local_memmap_path: Optional[Path] = None
        self._local_metadata_path: Optional[Path] = None
        self._written_tokens = 0
        self._memmap_file: Optional[np.memmap] = None
        self._metadata_file: Optional[BytesIO] = None

    def __len__(self) -> int:
        """Length of the memmap file in tokens that have been written."""
        return self._written_tokens

    # def write(self, values: List[int], flush: bool = False) -> Optional[List[int]]:
    def write(self, output: OutputSpec, flush: bool = False) -> Optional[OutputSpec]:
        """Write a list of token IDs to the memmap file; if only a subset of the values can be written,
        return the rest.

        Args:
            values (List[int]): List of token IDs to write.
            flush (bool, optional): Whether to flush the memmap file after writing. Defaults to False.
        """

        if self._memmap_file is None:
            raise RuntimeError("MemmapFile is not open")

        if self._metadata_file is None:
            raise RuntimeError("Metadata file is not open")

        if (len(output.tokens) + self._written_tokens) >= self.max_tokens:
            values = output.tokens[: self.max_tokens - self._written_tokens]
            start = 0
            end = self.max_tokens - self._written_tokens

            rest = OutputSpec(
                id=output.id,
                src=output.src,
                loc=output.loc,
                tokens=output.tokens[end:],
                start=end,
                end=len(output.tokens),
            )
        else:
            values = output.tokens
            start = 0
            end = len(output.tokens)
            rest = None

        metadata = Metadata(
            id=output.id,
            src=output.src,
            loc=output.loc,
            start=start,
            end=end,
        )
        self._memmap_file[self._written_tokens : self._written_tokens + start] = values
        self._written_tokens += end - start

        self._metadata_file.write(msgspec.json.encode(metadata) + b"\n")

        if flush:
            self._memmap_file.flush()
            self._metadata_file.flush()

        return rest

    @property
    def is_remote_path(self) -> bool:
        return re.match('[a-zA-Z0-9]+://', self.memmap_path) is not None and not self.memmap_path.startswith("file://")

    def __enter__(self) -> "MemMapWriter":
        """Context manager entry point. Creates the memmap file and returns self."""

        assert self._memmap_file is None and self._metadata_file is None, "MemmapFile is already open"

        if self.is_remote_path:
            with ExitStack() as stack:
                # if the destination for the memmap is not local, we need to write to a temporary file first
                _memmap_file = stack.enter_context(
                    NamedTemporaryFile(delete=False, prefix="olmo_memmap", suffix=self.MEMMAP_EXTENSION)
                )
                self._local_memmap_path = Path(_memmap_file.name)
                _metadata_file = stack.enter_context(
                    NamedTemporaryFile(delete=False, prefix="olmo_metadata", suffix=self.METADATA_EXTENSION)
                )
                self._local_metadata_path = Path(_metadata_file.name)
        else:
            self._local_memmap_path = Path(self.memmap_path)
            self._local_metadata_path = Path(self.metadata_path)
            # make sure the directory exists
            self._local_memmap_path.parent.mkdir(parents=True, exist_ok=True)
            self._local_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # these two assertions ensure type checking
        assert self._local_memmap_path is not None
        assert self._local_metadata_path is not None

        self._memmap_file = np.memmap(
            mode="w+",
            filename=self._local_memmap_path,
            dtype=self.dtype,
            shape=(self.max_tokens,)
        )
        self._metadata_file = smart_open.open(self._local_metadata_path, mode="wb")     # type: ignore

        log.info(f"Created memmap file at {self._local_memmap_path} of size {self._memmap_file.nbytes:,} bytes")

        return self

    def __exit__(self, *_):
        """Context manager exit point. Closes the memmap file."""
        return self.close()

    def close(self):
        """Close the memmap file and optionally upload it to the destination (in the case of a remote path)."""
        assert self._local_memmap_path is not None, "Local Memmap path is not provided"
        assert self._local_metadata_path is not None, "Local Metadata path is not provided"
        assert self._memmap_file is not None, "Memmap file is not open"
        assert self._metadata_file is not None, "Metadata file is not open"

        try:
            # write the memmap to the destination
            self._memmap_file.flush()

            # we resize the memmap to the number of tokens actually written
            if self._written_tokens < self.max_tokens:
                del self._memmap_file
                os.rename(self._local_memmap_path, (temp_path := self._local_memmap_path.with_suffix(".tmp")))
                new_memmap = np.memmap(
                    mode="w+", filename=self._local_memmap_path, dtype=self.dtype, shape=(self._written_tokens,)
                )
                old_memmap = np.memmap(mode="r", filename=temp_path, dtype=self.dtype, shape=(self.max_tokens,))
                new_memmap[:] = old_memmap[: self._written_tokens]
                new_memmap.flush()
                log.info(f"Resized memmap file from {old_memmap.nbytes:,} to {new_memmap.nbytes:,} bytes")
                os.remove(temp_path)

            if self.is_remote_path:
                with ExitStack() as stack:
                    f = stack.enter_context(smart_open.open(self._local_memmap_path, "rb"))
                    g = stack.enter_context(smart_open.open(self.memmap_path, mode="wb"))
                    g.write(f.read())
                log.info(f"Written memmap file to {self.memmap_path}")
        finally:
            if not self.is_remote_path:
                # delete the temporary file under any circumstances
                os.remove(self._local_memmap_path)

        self._local_memmap_path = self._memmap = None


def fill_memmap(
    tokenizer_id: str,
    path_or_paths: Union[str, List[str]],
    memmap_path: str,
    dtype: np.dtype,
    max_tokens: int = 1024 * 1024 * 1024,  # 1024 tokens * 2 bytes per token (uint16) = 2GB
    sample_rate: float = 1.0,
    random_seed: int = 3920,
    repeat_sequence: int = 1,
) -> int:
    """Write a memmap file from a file of documents."""

    # set the seed in case we need to sample
    np.random.seed(random_seed)

    # we need to make a new tokenizer here because it's not pickleable
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

    # first memmap file will be created in the loop below
    memmap_writer: Optional[MemMapWriter] = None

    # we increment this every time we create a new memmap file
    file_index = 0

    # total number of tokens written
    total_tokens = 0

    # make sure path is a list
    path_or_paths = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths

    with ExitStack() as stack:
        it = itertools.chain.from_iterable(
            # repeat the sequence if necessary
            tokenize_file(tokenizer=tokenizer, path=path)
            for _ in range(repeat_sequence)
            for path in path_or_paths
        )
        for line_no, output in enumerate(it, start=1):
            # perform sampling if necessary
            if sample_rate < 1.0 and np.random.rand() > sample_rate:
                continue

            # flush any 10k lines or so; improves stability
            flush = line_no % 10_000 == 0

            # increment the total number of tokens written
            total_tokens += len(output.tokens)

            if memmap_writer is not None:
                # leftovers_to_write is gonna be an OutputSpec with the tokens that didn't fit in the
                # current memmap, or None if all tokens fit
                leftovers_to_write = memmap_writer.write(output=output, flush=flush)
            else:
                # memmap hasn't been created yet, so technically the entire output is leftovers
                leftovers_to_write = output

            if leftovers_to_write is not None:
                # close the previous memmap (if one is open)
                stack.pop_all().close()

                # create a new memmap file; progressively name them with an index
                curr_memmap_path = f"{memmap_path}_{file_index:05d}.npy"
                memmap_writer = stack.enter_context(
                    MemMapWriter(path=curr_memmap_path, dtype=dtype, max_tokens=max_tokens)
                )

                # increment the file index and reset the tokens index
                file_index += 1

                # do the actual writing
                memmap_writer.write(leftovers_to_write)

        # close the last memmap
        stack.pop_all().close()

    return total_tokens


def make_source_and_target(
    src: Tuple[str, ...],
    output: str,
    random_seed: int = 3920,
    paths_per_worker: int = 1,
) -> Tuple[Tuple[Union[str, List[str]], ...], Tuple[str, ...]]:
    """Recursively list all files in the source directories and create a corresponding list of destination."""

    np.random.seed(random_seed)
    random.seed(random_seed)

    exploded_src = list(set(path for prefix in src for path in glob_path(prefix)))
    output_digits = np.ceil(np.log10(len(exploded_src) + 1)).astype(int)

    # shuffle the source paths
    random.shuffle(exploded_src)

    grouped_src: Union[List[str], List[List[str]]]
    if paths_per_worker > 1:
        assert (
            len(exploded_src) >= paths_per_worker
        ), f"Number of paths ({len(exploded_src)}) must be <= paths_per_worker ({paths_per_worker})"

        # group the paths into chunks of paths_per_worker
        grouped_src = [
            sorted(exploded_src[i : i + paths_per_worker]) for i in range(0, len(exploded_src), paths_per_worker)
        ]
    else:
        grouped_src = exploded_src

    # determine the destination paths
    exploded_dst = [f'{output.rstrip("/")}/{i:0{output_digits}d}' for i in range(len(grouped_src))]

    return tuple(grouped_src), tuple(exploded_dst)


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    prompt="Output directory",
)
@click.option(
    "--tokenizer",
    "tokenizer_id",
    type=str,
    help="Name of path of a pretrained tokenizer",
    default="allenai/eleuther-ai-gpt-neox-20b-pii-special",
)
@click.option("--dtype", "dtype_str", default="uint16")
@click.option("--validate/--no-validate", default=False)
@click.option("--sample-rate", type=click.FloatRange(min=0.0, max=1.0), default=1.0)
@click.option("--random-seed", type=int, default=3920)
@click.option("--repeat-sequence", type=click.IntRange(min=1), default=1)
@click.option("--paths-per-worker", type=click.IntRange(min=1), default=1)
@click.option(
    "--cache-dir",
    type=str,
    default=None,
    help="Cache directory for the tokenizer; use system default if not specified",
)
@click.option(
    "--max-tokens",
    default=512 * 1024 * 1024,
    type=int,
    help="Maximum number of tokens to store in a single memmap file (default: 512M tokens or 1GB)",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug (single process mode)")
@click.option(
    "--safe-mode/--fast-mode", default=False, help="Safe mode caches locally and decompresses using gzip.open"
)
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Defaults to number of CPUs")
def main(
    src: Tuple[str, ...],
    output: str,
    tokenizer_id: str = "EleutherAI/gpt-neox-20b",
    dtype_str: str = "uint16",
    validate: bool = False,
    max_tokens: int = 512 * 1024 * 1024,
    safe_mode: bool = False,
    debug: bool = False,
    sample_rate: float = 1.0,
    random_seed: int = 3920,
    repeat_sequence: int = 1,
    paths_per_worker: int = 1,
    max_workers: int = 1,
    cache_dir: Optional[str] = None,
):
    print("=== CONFIGURATION ===")
    print(f"src:              {src}")
    print(f"output:           {output}")
    print(f"tokenizer_id:     {tokenizer_id}")
    print(f"dtype_str:        {dtype_str}")
    print(f"validate:         {validate}")
    print(f"max_tokens:       {max_tokens}")
    print(f"debug:            {debug}")
    print(f"sample_rate:      {sample_rate}")
    print(f"random_seed:      {random_seed}")
    print(f"repeat_sequence:  {repeat_sequence}")
    print(f"paths_per_worker: {paths_per_worker}")
    print(f"max_workers:      {max_workers}")
    print("=====================")

    dtype = np.dtype(dtype_str)
    exploded_src, exploded_dst = make_source_and_target(
        src=src, output=output, random_seed=random_seed, paths_per_worker=paths_per_worker
    )

    # creating a partial here with all the arguments we need to pass to fill_memmap except for the paths
    # so that we don't make mistakes between debug and non-debug mode
    fill_memmap_fn = functools.partial(
        fill_memmap,
        tokenizer_id=tokenizer_id,
        dtype=dtype,
        max_tokens=max_tokens,
        sample_rate=sample_rate,
        random_seed=random_seed,
        repeat_sequence=repeat_sequence,
    )

    total_tokens_written = 0

    if debug:
        log.info("Running in debug mode. Only one process will be used.")
        for src_path, dst_path in zip(exploded_src, exploded_dst):
            total_tokens_written += fill_memmap_fn(path_or_paths=src_path, memmap_path=dst_path)
    else:
        # Now tokenizer all documents again and populate the memmap array. We do this in parallel.
        workers_cnt = min(max_workers or os.cpu_count() or 1, len(exploded_src))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
            futures: List[Future[int]] = []
            for src_path, dst_path in zip(exploded_src, exploded_dst):
                future = executor.submit(fill_memmap_fn, path_or_paths=src_path, memmap_path=dst_path)
                futures.append(future)
            with get_progress() as progress:
                for future in progress.track(
                    concurrent.futures.as_completed(futures),
                    description="Filling memmap arrays...",
                    total=len(futures),
                ):
                    total_tokens_written += future.result()

    log.info(f"Done! File(s) written to {output}")
    log.info(f"Total tokens written: {total_tokens_written:,}")

    if validate:
        log.info("Validating...")
        tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

        def encode_fn(row):
            return tokenizer.encode(json.loads(row)["text"], add_special_tokens=True)  # noqa

        total_tokens = total_docs = 0
        for input_path in (path for prefix in src for path in glob_path(prefix)):
            with smart_open.open(input_path, mode="rb") as g:
                for row in g:
                    total_docs += 1
                    total_tokens += len(encode_fn(row))

        for output_path in glob_path(output):
            memmap = np.memmap(output_path, mode="r", dtype=dtype)
            total_tokens -= len(memmap)
            total_docs -= (memmap == tokenizer.eos_token_id).sum()
            assert (memmap < tokenizer.vocab_size).all(), f"Invalid token ID in {output_path}"

        assert total_tokens == 0, f"Total tokens mismatch: {total_tokens} != 0"
        assert total_docs == 0, f"Total docs mismatch: {total_docs} != 0"

        log.info("All good!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
