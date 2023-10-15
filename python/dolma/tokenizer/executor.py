import hashlib
import multiprocessing
import os
import random
import tempfile
from contextlib import ExitStack
from math import ceil, log10
from queue import Queue
from typing import Any, Dict, List, Optional

import numpy as np
from typing_extensions import TypeAlias

from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.paths import glob_path, join_path, mkdir_p
from .data_types import TokenizerOutput
from .memmap_writer import MemmapWriter
from .tokenizer import Tokenizer, tokenize_file

TokenizedSeqsQueueType: TypeAlias = "Queue[List[TokenizerOutput]]"
PathsQueueType: TypeAlias = "Queue[str]"


class MemMapParallelWriter(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore[override]
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
        max_size: int = kwargs.pop("max_size", 1024 * 1024 * 1024)
        dtype: np.dtype = np.dtype(kwargs.pop("dtype", "uint16"))
        local_shuffle: int = kwargs.pop("local_shuffle", 10_000)
        ring_size: int = kwargs.pop("ring_size", 8)

        global_source_paths = kwargs.pop("grouped_source_prefixes", None)
        if not isinstance(global_source_paths, list):
            raise RuntimeError("grouped_source_prefixes should be a list of paths")
        source_paths = global_source_paths[int(source_path)]

        cpu_count = multiprocessing.cpu_count()

        documents_cnt = tokens_cnt = 0
        update_interval = 1
        mm_cnt = 0

        tokenizer = Tokenizer.from_pretrained("allenai/eleuther-ai-gpt-neox-20b-pii-special")
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
                                dtype=np.dtype("uint16"),
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

        # group source paths into a number determined by num_readers or the (# sources / number of processes)
        num_readers_per_writer = float(num_readers or len(all_source_paths) / self.num_processes)
        grouped_source_prefixes = []
        i = 0.0
        while i < len(all_source_paths):
            grouped_source_prefixes.append(all_source_paths[int(i) : int(i + num_readers_per_writer)])
            i += num_readers_per_writer

        # redefine num_processes to be the number of groups
        self.num_processes = min(len(grouped_source_prefixes), self.num_processes)

        # this is a bit of a hack but: we pass indices to grouped_source_prefixes to the processors
        # so that they can load the correct source paths
        source_indices = [str(i) for i in range(len(grouped_source_prefixes))]

        # check that only one destination and metadata is provided
        if len(self.dst_prefixes) != 1 or len(self.meta_prefixes) != 1:
            raise ValueError("Only one destination and metadata should be provided.")

        # make necessary destination directories
        destination = self.dst_prefixes[0]
        mkdir_p(destination)

        # each parallel processor will write a file name like part-dddddd-dddd.npy and part-dddddd-dddd.csv.gz
        digits = int(ceil(log10(len(grouped_source_prefixes) + 1)))
        destinations = [
            join_path(None, destination, f"part-{i:0{digits}d}") for i in range(len(grouped_source_prefixes))
        ]

        # same for metadata
        metadata = self.meta_prefixes[0]
        mkdir_p(metadata)
        metadatas = [join_path(None, metadata, f"{i}.done") for i in range(len(destinations))]

        # finally run the processors
        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=source_indices,
            all_destination_paths=destinations,
            all_metadata_paths=metadatas,
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
    tokenizer_name_or_path: str = "allenai/eleuther-ai-gpt-neox-20b-pii-special",
    seed: int = 3920,
    metadata_dir: Optional[str] = None,
    max_size: int = 1024 * 1024 * 1024,
    dtype: str = "uint16",
    debug: bool = False,
):
    multiprocessing.set_start_method("spawn")

    # variables for the nice debugging and tokenizers
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # do it once so it gets cached
    Tokenizer.from_pretrained(tokenizer_name_or_path, truncate_to=None)

    # get a run hash
    run_hash = hashlib.sha256(("".join(sources) + tokenizer_name_or_path).encode("utf-8")).hexdigest()[:8]
    metadata_dir = metadata_dir or join_path(None, tempfile.gettempdir(), f"dolma-{run_hash}")

    parallel_writer = MemMapParallelWriter(
        source_prefix=sources,
        destination_prefix=destination,
        metadata_prefix=metadata_dir,
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
    )
