import functools
import os
import re
from contextlib import ExitStack
from csv import writer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, TextIO

import numpy as np
import smart_open

from ..core.loggers import get_logger
from .data_types import MemmapMetadata, TokenizerOutput

log = get_logger(__name__)


class MemmapWriter:
    """Context manager responsible for writing, resizing, and closing / uploading a memmap file."""

    DEFAULT_MAX_TOKENS = 512 * 1024 * 1024  # 500M tokens / 1GB
    MEMMAP_EXTENSION = ".npy"
    METADATA_EXTENSION = ".csv.gz"

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
        base_path = re.sub(r"(\.npy?)?(\.[a-zA-Z]+)*$", "", path)
        self.memmap_path = f"{base_path}{self.MEMMAP_EXTENSION}"
        self.metadata_path = f"{base_path}{self.METADATA_EXTENSION}"
        self.dtype = dtype
        self.max_tokens = max_tokens

        self._local_memmap_path: Optional[Path] = None
        self._local_metadata_path: Optional[Path] = None
        self._written_tokens = 0
        self._memmap_file: Optional[np.memmap] = None
        self._metadata_file: Optional[TextIO] = None

    @functools.cached_property
    def metadata_writer(self):
        if self._metadata_file is None:
            raise RuntimeError("Metadata file is not open")
        return writer(self._metadata_file)

    def __len__(self) -> int:
        """Length of the memmap file in tokens that have been written."""
        return self._written_tokens

    def write(self, output: TokenizerOutput, flush: bool = False) -> bool:
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
            # return false if the memmap file is full
            return False

        metadata = MemmapMetadata(
            id=output.id,
            src=output.src,
            loc=output.loc,
            start=self._written_tokens,
            end=self._written_tokens + output.end,
        )
        self._memmap_file[self._written_tokens : self._written_tokens + output.end] = output.tokens
        self._written_tokens += output.end

        # self._metadata_file.write(msgspec.json.encode(metadata) + b"\n")
        self.metadata_writer.writerow(metadata)

        if flush:
            self.flush()

        return True

    def write_many(self, outputs: List[TokenizerOutput], flush: bool = False) -> List[TokenizerOutput]:
        remaining: List[TokenizerOutput] = []

        for i, output in enumerate(outputs):
            if not self.write(output=output):
                remaining = outputs[i:]
                break

        if flush:
            self.flush()

        return remaining

    def flush(self):
        """Flush the memmap file."""
        if self._memmap_file is not None:
            self._memmap_file.flush()

        if self._metadata_file is not None:
            self._metadata_file.flush()

    @property
    def is_remote_path(self) -> bool:
        return re.match("[a-zA-Z0-9]+://", self.memmap_path) is not None and not self.memmap_path.startswith(
            "file://"
        )

    def __enter__(self) -> "MemmapWriter":
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
            mode="w+", filename=self._local_memmap_path, dtype=self.dtype, shape=(self.max_tokens,)
        )
        self._metadata_file = smart_open.open(self._local_metadata_path, mode="wt")

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
            self.flush()
            self._metadata_file.close()

            # we resize the memmap to the number of tokens actually written
            if self._written_tokens < self.max_tokens and self._written_tokens > 0:
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

                    f = stack.enter_context(smart_open.open(self._local_metadata_path, "rb"))
                    g = stack.enter_context(smart_open.open(self.metadata_path, mode="wb"))
                    g.write(f.read())

                log.info(f"Written memmap file to {self.memmap_path}")
        finally:
            if self.is_remote_path:
                # delete the temporary file under any circumstances
                os.remove(self._local_memmap_path)

        # reset to none, clear cache
        self._local_memmap_path = self._memmap_file = None
        self._local_metadata_path = self._metadata_file = None

        try:
            del self.metadata_writer
        except AttributeError:
            # this is in case the metadata file was never opened
            pass
