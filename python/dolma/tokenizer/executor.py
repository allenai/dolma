import hashlib
import multiprocessing
import os
import random
import tempfile
from contextlib import ExitStack
from math import ceil, log10
from queue import Queue
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
    def increment_progressbar(cls, queue: QueueType, /, documents: int = 0, tokens: int = 0) -> Dict[str, int]:
        return super().increment_progressbar(queue, documents=documents, tokens=tokens)

    @classmethod
    def _make_memwriter(cls, path: str, stack: ExitStack, max_size: int) -> MemmapWriter:
        memwriter = stack.enter_context(MemmapWriter(path=path, dtype=np.dtype("uint16"), max_tokens=max_size))
        return memwriter

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        tokenized_seqs_queue: TokenizedSeqsQueueType = kwargs.pop("tokenized_seqs_queue")
        paths_queue: PathsQueueType = kwargs.pop("paths_queue")
        max_size: int = kwargs.pop("max_size")
        seed: int = kwargs.pop("seed")
        cpu_count = multiprocessing.cpu_count()
        random.seed(seed)

        documents_cnt = tokens_cnt = 0
        update_interval = 1
        memwriter_cnt = 0

        with ExitStack() as stack:
            # create a memmap writer

            memwriter = cls._make_memwriter(
                path=destination_path + f"-{memwriter_cnt:05d}", stack=stack, max_size=max_size
            )

            while True:
                try:
                    tokenized_seqs = tokenized_seqs_queue.get(timeout=1)
                except Exception:
                    tokenized_seqs = None

                if tokenized_seqs is None and paths_queue.empty():
                    break
                elif tokenized_seqs is None:
                    sleep(0.1)
                    continue

                # shuffle sequence order to ensure that the sequences are well mixed
                random.shuffle(tokenized_seqs)

                # try to write all the sequences, collect the ones that don't fit in remaining
                remaining = memwriter.write_many(outputs=tokenized_seqs)

                if remaining:
                    # if we have remaining sequences, we need to close the current memwriter and open a new one
                    stack.pop_all().close()
                    memwriter_cnt += 1
                    memwriter = cls._make_memwriter(
                        path=destination_path + f"-{memwriter_cnt:05d}", stack=stack, max_size=max_size
                    )

                    # finally, write the remaining sequences
                    memwriter.write_many(outputs=remaining)

                documents_cnt += 1
                tokens_cnt += sum(seq.end for seq in tokenized_seqs)
                documents_cnt += len(tokenized_seqs)

                if documents_cnt % update_interval == 0:
                    cls.increment_progressbar(queue, documents=update_interval, tokens=tokens_cnt)
                    tokens_cnt = documents_cnt = 0

                    if queue.qsize() >= cpu_count:
                        # double the update interval if the queue is full
                        update_interval *= 2

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


def tokenize_and_enqueue(
    tokenizer_id: str,
    paths_queue: PathsQueueType,
    tokenized_seqs_queue: TokenizedSeqsQueueType,
    metadata_dir: str,
    enqueue_every: int = 5_000,
):
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
    accumulator = []

    while not paths_queue.empty():
        try:
            path = paths_queue.get(timeout=1)
        except Exception:
            continue

        # skip if we've already processed this file
        path_hash = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
        metadata_path = join_path(None, metadata_dir, f"{path_hash}.done")
        if os.path.exists(metadata_path):
            continue

        # tokenize the file, and enqueue the results every `enqueue_every`  documents
        for output_spec in tokenize_file(tokenizer=tokenizer, path=path):
            accumulator.append(output_spec)
            if len(accumulator) >= enqueue_every:
                tokenized_seqs_queue.put(accumulator)
                accumulator = []

        # enqueue the remaining documents
        if accumulator:
            tokenized_seqs_queue.put(accumulator)

        # write a metadata file to indicate that this file is done
        with smart_open.open(metadata_path, "w") as f:
            f.write("")


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

    with multiprocessing.Manager() as manager:
        # create a queue where workers can dump data
        tokenized_seqs_queue: TokenizedSeqsQueueType = manager.Queue()
        paths_queue: "Queue[str]" = manager.Queue()
        for path in source_paths:
            paths_queue.put(path)

        # store tokenizer processes here
        processes = []

        tokenizer_metadata_dir = join_path(None, metadata_dir, "tokenizers")
        mkdir_p(tokenizer_metadata_dir)

        for i in range(num_tokenizers):
            process = multiprocessing.Process(
                target=tokenize_and_enqueue,
                kwargs={
                    "tokenizer_id": tokenizer_id,
                    "paths_queue": paths_queue,
                    "tokenized_seqs_queue": tokenized_seqs_queue,
                    "metadata_dir": tokenizer_metadata_dir,
                },
            )
            processes.append(process)
            process.start()

            # give the process some time to start
            sleep(0.5)

        # each parallel processor will write a file name like part-dddddd-dddd.npy and part-dddddd-dddd.csv.gz
        digits = int(ceil(log10(num_writers + 1)))
        destinations = [join_path(None, destination, f"part-{i:0{digits}d}") for i in range(num_writers)]

        parallel_writer = MemMapParallelWriter(
            source_prefix=["" for _ in range(num_writers)],
            destination_prefix=destinations,
            metadata_prefix=[join_path(None, metadata_dir, "writers") for _ in range(num_writers)],
            debug=True,
        )
        parallel_writer(
            tokenized_seqs_queue=tokenized_seqs_queue, paths_queue=paths_queue, max_size=max_size, seed=seed
        )

        # wait for the tokenizer processes to finish
        for process in processes:
            process.join()
