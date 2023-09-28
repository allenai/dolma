import hashlib
import multiprocessing
import os
import random
import tempfile
from contextlib import ExitStack
from math import ceil, log10
from queue import Queue, Empty
from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np
import smart_open
from typing_extensions import TypeAlias

from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.paths import glob_path, join_path, mkdir_p
from .data_types import TokenizerOutput
from .memmap_writer import MemmapWriter
from .tokenizer import Tokenizer, tokenize_file

TokenizedSeqsQueueType: TypeAlias = Queue[List[TokenizerOutput]]
PathsQueueType: TypeAlias = Queue[str]


class MemMapParallelWriter(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(      # type: ignore[override]
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
    def process_single(cls, source_path: List[str], destination_path: str, queue: QueueType, **kwargs: Any):
        max_size: int = kwargs.pop("max_size")
        seed: int = kwargs.pop("seed")
        cpu_count = multiprocessing.cpu_count()

        documents_cnt = tokens_cnt = 0
        update_interval = 1
        mm_cnt = 0

        tokenizer = Tokenizer.from_pretrained("allenai/eleuther-ai-gpt-neox-20b-pii-special")
        tokenizer_ring = []
        for _ in range(8):
            path = source_path.pop()
            tokenizer_ring.append(tokenize_file(tokenizer=tokenizer, path=path))

        accumulator = []

        with ExitStack() as stack:
            memwriter = stack.enter_context(
                MemmapWriter(path=destination_path + f"-{mm_cnt:05d}", dtype=np.dtype("uint16"), max_tokens=max_size)
            )
            cls.increment_progressbar(queue, memmaps=1)

            while len(source_path) > 0 or len(tokenizer_ring) > 0:
                for i in range(10_000):
                    j = i % len(tokenizer_ring)
                    try:
                        content = next(tokenizer_ring[j])
                        accumulator.append(content)
                    except StopIteration:
                        cls.increment_progressbar(queue, files=1)
                        tokenizer_ring.pop(j)
                        if len(tokenizer_ring) == 0:
                            break
                        if len(source_path) > 0:
                            path = source_path.pop()
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
                            MemmapWriter(path=destination_path + f"-{mm_cnt:05d}", dtype=np.dtype("uint16"), max_tokens=max_size)
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

    def __call__(self, **process_single_kwargs: Any):
        """Run the processor."""
        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=self.src_prefixes,
            all_destination_paths=self.dst_prefixes,
            all_metadata_paths=self.meta_prefixes,
            **process_single_kwargs,
        )

def tokenize_in_parallel(
    sources: List[str],
    destination: str,
    num_tokenizers: int = 1,
    num_writers: int = 1,
    tokenizer_id: str = "allenai/eleuther-ai-gpt-neox-20b-pii-special",
    seed: int = 3920,
    metadata_dir: Optional[str] = None,
    max_size: int = 1024 * 1024 * 1024,
):
    multiprocessing.set_start_method("spawn")

    # variables for the nice debugging and tokenizers
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set a seed for shuffling
    random.seed(seed)

    # do it once so it gets cached
    Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

    # shuffle to ensure sequences are well mixed
    source_paths = [p for p in set([p for source in sources for p in glob_path(source)])]
    random.shuffle(source_paths)

    # get a run hash
    run_hash = hashlib.sha256(("".join(source_paths) + tokenizer_id).encode("utf-8")).hexdigest()[:8]
    metadata_dir = metadata_dir or join_path(None, tempfile.gettempdir(), f"dolma-{run_hash}")

    # make sure the destination exists
    mkdir_p(destination)

    sources_per_num_writers = len(source_paths) / num_writers
    source_prefix = [
        source_paths[int(i * sources_per_num_writers) : int((i + 1) * sources_per_num_writers)]
        for i in range(num_writers)
    ]

    tokenizer_metadata_dir = join_path(None, metadata_dir, "tokenizers")
    mkdir_p(tokenizer_metadata_dir)

    # each parallel processor will write a file name like part-dddddd-dddd.npy and part-dddddd-dddd.csv.gz
    digits = int(ceil(log10(num_writers + 1)))
    destinations = [join_path(None, destination, f"part-{i:0{digits}d}") for i in range(num_writers)]

    parallel_writer = MemMapParallelWriter(
        source_prefix=source_prefix,
        destination_prefix=destinations,
        metadata_prefix=[join_path(None, metadata_dir, "writers") for _ in range(num_writers)],
        num_processes=num_writers,
    )
    parallel_writer(max_size=max_size, seed=seed)
