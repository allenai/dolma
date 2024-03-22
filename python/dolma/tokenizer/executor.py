import hashlib
import multiprocessing
import os
import random
import tempfile
from contextlib import ExitStack
from math import ceil, log10
from queue import Queue  # pylint: disable=unused-import
from typing import Any, Dict, List, Optional

import numpy as np
from typing_extensions import TypeAlias

from ..core.loggers import get_logger
from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.paths import glob_path, join_path, mkdir_p
from .data_types import TokenizerOutput  # pylint: disable=unused-import
from .memmap_writer import MemmapWriter
from .tokenizer import Tokenizer, tokenize_file

TokenizedSeqsQueueType: TypeAlias = "Queue[List[TokenizerOutput]]"
PathsQueueType: TypeAlias = "Queue[str]"


class MemMapParallelWriter(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore[override]    # pylint: disable=arguments-differ
        cls,
        queue: QueueType,
        /,
        files: int = 0,
        documents: int = 0,
        tokens: int = 0,
        memmaps: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(
            queue, files=files, documents=documents, tokens=tokens, memmaps=memmaps
        )

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        logger = get_logger(__name__)

        max_size: int = kwargs.pop("max_size", 1024 * 1024 * 1024)
        dtype: np.dtype = np.dtype(kwargs.pop("dtype", "uint16"))
        local_shuffle: int = kwargs.pop("local_shuffle", 10_000)
        ring_size: int = kwargs.pop("ring_size", 8)

        global_source_paths = kwargs.pop("grouped_source_prefixes", None)
        if not isinstance(global_source_paths, list):
            raise RuntimeError("grouped_source_prefixes should be a list of paths")
        source_paths = global_source_paths[int(source_path)]

        tokenizer_name_or_path = kwargs.pop("tokenizer_name_or_path", None)
        if tokenizer_name_or_path is None:
            raise RuntimeError("tokenizer_name_or_path not provided")

        tokenizer_kwargs = {}
        tokenizer_kwargs["bos_token_id"] = kwargs.pop("bos_token_id", None)
        tokenizer_kwargs["eos_token_id"] = kwargs.pop("eos_token_id", None)
        if tokenizer_kwargs["bos_token_id"] is None and tokenizer_kwargs["eos_token_id"] is None:
            raise ValueError(
                "Neither eos_token_id nor bos_token_id specified. "
                "At least one of them should be provided; otherwise, documents will not be properly separated."
            )

        tokenizer_kwargs["pad_token_id"] = kwargs.pop("pad_token_id", None)
        if tokenizer_kwargs["pad_token_id"] is None:
            logger.warning("pad_token_id not provided, using eos_token_id")
            tokenizer_kwargs["pad_token_id"] = tokenizer_kwargs["eos_token_id"]

        # flag to control whether to segment the documents before tokenization
        tokenizer_kwargs["segment_before_tokenization"] = kwargs.pop("segment_before_tokenization", False)

        # this is useful for making sure the queue does not grows too much
        cpu_count = multiprocessing.cpu_count()

        # these are used to keep track of the progress
        documents_cnt = tokens_cnt = 0
        update_interval = 1
        mm_cnt = 0

        # def test(**kwargs):
        #     breakpoint()

        # create the tokenizer from file if it exists, otherwise from pretrained
        if os.path.exists(tokenizer_name_or_path) and os.path.isfile(tokenizer_name_or_path):
            tokenizer = Tokenizer.from_file(tokenizer_name_or_path, **tokenizer_kwargs)
        else:
            tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

        tokenizer_ring = []
        for _ in range(min(ring_size, len(source_paths))):
            path = source_paths.pop()
            tokenizer_ring.append(tokenize_file(tokenizer=tokenizer, path=path))

        accumulator = []

        with ExitStack() as stack:
            memwriter = stack.enter_context(
                MemmapWriter(path=destination_path + f"-{mm_cnt:05d}", dtype=dtype, max_tokens=max_size)
            )
            cls.increment_progressbar(queue, memmaps=1)

            while len(source_paths) > 0 or len(tokenizer_ring) > 0:
                for i in range(local_shuffle):
                    j = i % len(tokenizer_ring)
                    try:
                        content = next(tokenizer_ring[j])
                        accumulator.append(content)
                    except StopIteration:
                        cls.increment_progressbar(queue, files=1)
                        tokenizer_ring.pop(j)
                        if len(tokenizer_ring) == 0:
                            break
                        if len(source_paths) > 0:
                            path = source_paths.pop()
                            tokenizer_ring.append(tokenize_file(tokenizer=tokenizer, path=path))

                # shuffle sequence order to ensure that the sequences are well mixed
                random.shuffle(accumulator)

                # try to write all the sequences, collect the ones that don't fit in remaining
                remaining = memwriter.write_many(outputs=accumulator, flush=documents_cnt == 0)

                if remaining:
                    # if we have remaining sequences, we need to close the current memwriter and open a new one
                    mm_cnt += 1
                    stack.pop_all().close()
                    memwriter = stack.enter_context(
                        MemmapWriter(
                            path=destination_path + f"-{mm_cnt:05d}",
                            dtype=np.dtype("uint16"),  # pyright: ignore
                            max_tokens=max_size,
                        )
                    )
                    cls.increment_progressbar(queue, memmaps=1)

                    # finally, write the remaining sequences
                    memwriter.write_many(outputs=remaining, flush=True)

                tokens_cnt += sum(seq.end for seq in accumulator)
                documents_cnt += len(accumulator)

                if documents_cnt >= update_interval:
                    cls.increment_progressbar(queue, documents=documents_cnt, tokens=tokens_cnt)
                    tokens_cnt = documents_cnt = 0

                    if queue.qsize() >= cpu_count:
                        # double the update interval if the queue is full
                        update_interval *= 2

                accumulator = []

                memwriter.flush()

        cls.increment_progressbar(queue, documents=documents_cnt, tokens=tokens_cnt)

    def __call__(self, num_readers: Optional[int] = None, **process_single_kwargs: Any):
        """Run the processor."""

        # get all source paths; shuffle them well
        all_source_paths = [p for p in set([p for source in self.src_prefixes for p in glob_path(source)])]
        random.shuffle(all_source_paths)

        # TRICKY BIT: Group source paths into buckets
        # First, check what the step size should be. The step is the minimum between the
        # number of readers requested, and the number of source_paths per process.
        # The float("inf") bit is required to handle the case when num_readers is None.
        step_size = min(num_readers or float("inf"), len(all_source_paths) / self.num_processes)

        # Now, we step over all files in increments of step_size, and group them into buckets
        # we need to make sure we don't add empty buckets.
        grouped_source_prefixes: List[List[str]] = []
        current_step = 0.0
        while current_step < len(all_source_paths):  # can't use range here because of the float
            prefix_slice = all_source_paths[int(current_step) : int(current_step + step_size)]
            if prefix_slice:
                grouped_source_prefixes.append(prefix_slice)
            current_step += step_size

        # Finally, we optionally redefine num_processes to be the number of groups otherwise some
        # processors will not have any work to do.
        self.num_processes = min(len(grouped_source_prefixes), self.num_processes)

        # We have one set of sanity checks here to make sure that the grouping was done correctly
        if any(len(bucket) == 0 for bucket in grouped_source_prefixes):
            raise ValueError("Some buckets are empty. This should not happen.")
        if len(grouped_source_prefixes) < self.num_processes:
            raise ValueError("The number of groups is less than the number of processes. This should not happen.")
        if len(all_source_paths) < len(grouped_source_prefixes):
            raise ValueError(
                "The number of groups is greater than the number of source paths. This should not happen."
            )

        # this is a bit of a hack but: we pass indices to grouped_source_prefixes to the processors
        # so that they can load the correct source paths
        source_indices = [str(i) for i in range(len(grouped_source_prefixes))]

        # check that only one value of destination and metadata is provided
        if len(set(self.dst_prefixes)) != 1 or len(set(self.meta_prefixes)) != 1:
            raise ValueError("Only one destination and metadata should be provided.")

        # make necessary destination directories
        destination = self.dst_prefixes[0]
        mkdir_p(destination)

        # each parallel processor will write a file name like part-dddddd-dddd.npy and part-dddddd-dddd.csv.gz
        digits = int(ceil(log10(len(grouped_source_prefixes) + 1)))
        all_destination_paths = [
            join_path(None, destination, f"part-{i:0{digits}d}") for i in range(len(grouped_source_prefixes))
        ]

        # same for metadata
        metadata = self.meta_prefixes[0]
        mkdir_p(metadata)
        all_metadata_path = [join_path(None, metadata, f"{i}.done") for i in range(len(all_destination_paths))]

        # finally run the processors
        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=source_indices,
            all_destination_paths=all_destination_paths,
            all_metadata_paths=all_metadata_path,
            grouped_source_prefixes=grouped_source_prefixes,
            **process_single_kwargs,
        )


def tokenize_in_parallel(
    sources: List[str],
    destination: str,
    num_writers: int = 1,
    num_readers: Optional[int] = None,
    local_shuffle: int = 10_000,
    ring_size: int = 8,
    tokenizer_name_or_path: str = "allenai/gpt-neox-olmo-dolma-v1_5",
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = 50279,
    pad_token_id: Optional[int] = 1,
    segment_before_tokenization: bool = False,
    seed: int = 3920,
    metadata_dir: Optional[str] = None,
    max_size: int = 1024 * 1024 * 1024,
    dtype: str = "uint16",
    debug: bool = False,
):
    """
    Tokenizes the input sources in parallel using multiple writers and readers.

    Args:
        sources (List[str]): List of source file paths to tokenize.
        destination (str): Destination directory to store the tokenized files.
        num_writers (int, optional): Number of writer processes to use. Defaults to 1.
        num_readers (int, optional): Number of reader processes to use. Defaults to None.
        local_shuffle (int, optional): Number of lines to shuffle locally before writing. Defaults to 10_000.
        ring_size (int, optional): Size of the ring buffer for inter-process communication. Defaults to 8.
        tokenizer_name_or_path (str, optional): Name or path of the tokenizer to use.
            Defaults to "allenai/gpt-neox-olmo-dolma-v1_5". Note that, if the tokenizer is changed,
            you may need to adjust `bos_token_id`, `eos_token_id`, and `pad_token_id` accordingly.
        bos_token_id (int, optional): ID of the beginning-of-sentence token. Defaults to None.
        eos_token_id (int, optional): ID of the end-of-sentence token. Defaults to 50279.
        pad_token_id (int, optional): ID of the padding token. Defaults to 1.
        segment_before_tokenization (bool, optional): Whether to segment the input before tokenization.
            Defaults to False.
        seed (int, optional): Seed value for randomization. Defaults to 3920.
        metadata_dir (str, optional): Directory to store metadata files. Defaults to None.
        max_size (int, optional): Maximum size of each tokenized file. Defaults to 1024 * 1024 * 1024.
        dtype (str, optional): Data type for tokenized files. Defaults to "uint16".
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
    """
    # variables to avoid issues with parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # do it once so it gets cached (unless it's local path, so no need)
    if not os.path.exists(tokenizer_name_or_path):
        Tokenizer.from_pretrained(
            identifier=tokenizer_name_or_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    # get a run hash
    run_hash = hashlib.sha256(("".join(sources) + tokenizer_name_or_path).encode("utf-8")).hexdigest()[:8]
    metadata_dir = metadata_dir or join_path(None, tempfile.gettempdir(), f"dolma-{run_hash}")

    parallel_writer = MemMapParallelWriter(
        source_prefix=sources,
        # the call action will actually get the first destination and
        # make relative paths from there. Unfortunately, BaseParallelProcessor
        # expects as many destinations as there are sources, so we employ
        # this "hack" (that is, repeating destination len(sources) times)
        # to get around that. Same thing applies to metadata_dir.
        destination_prefix=[destination for _ in sources],
        metadata_prefix=[metadata_dir for _ in sources],
        num_processes=num_writers,
        seed=seed,
        debug=debug,
    )
    parallel_writer(
        num_readers=num_readers,
        local_shuffle=local_shuffle,
        ring_size=ring_size,
        max_size=max_size,
        dtype=dtype,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        segment_before_tokenization=segment_before_tokenization,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
