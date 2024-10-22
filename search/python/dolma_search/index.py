"""
python -m dolma_decontamination.search.index \
    -i /data/flan_index \
    -d "s3://ai2-llm/pretraining-data/sources/tulu_flan/v1-decontaminated-60M-shots_all-upweight_1-dialog_false-sep_rulebased/documents/train/*.gz" \
    -n 4 \
    -N 12 \
    -b 1000 \
    -B 50000 \
    -f
"""

from functools import partial
import logging
import json
import shutil
from queue import Queue
import argparse
import tqdm
from pathlib import Path
from tantivy import Document, SchemaBuilder, Index
import fsspec
import smart_open
import time
from urllib.parse import urlparse
from multiprocessing import Pool, Manager, set_start_method
from contextlib import ExitStack


from .common import create_index, IndexFields


INDEX_DESCRIPTION = "Index documents into a tantivy index"


QueueType = Queue[Document | None]


def get_fs(uri: str) -> fsspec.AbstractFileSystem:
    return fsspec.filesystem(urlparse(uri).scheme or "file")


def list_path(pattern: str) -> list[str]:
    fs = get_fs(pattern)
    protocol = urlparse(pattern).scheme
    paths = []
    for path in fs.glob(pattern):
        if protocol:
            paths.append(f"{protocol}://{path}")
        else:
            paths.append(str(path))
    del fs
    return paths


def list_paths(glob_patterns: list[str], num_workers: int = 1) -> list[str]:
    with Pool(processes=num_workers) as pool:
       return [p for ps in pool.map(list_path, glob_patterns) for p in ps]


def read_file_for_indexing(file_path: str, docs_queue: Queue[list[Document]], batch_size: int = 1_000):
    batch: list[Document] = []
    with smart_open.open(file_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            doc = Document(**{f.value: (row[f.value] or "") for f in IndexFields})
            batch.append(doc)

            if len(batch) >= batch_size:
                docs_queue.put(batch)
                batch = []

    if batch:
        docs_queue.put(batch)


def read_many_and_index(
    index: Index,
    paths: list[str],
    num_readers: int = 1,
    num_indexers: int = 1,
    indexer_batch_size: int = 1_000,
    reader_batch_size: int = 1_000,
    heap_size: int = 1024 * 1024 * 1024,
    queue_size: int = 1000,
):
    with ExitStack() as stack:
        reader_pool = stack.enter_context(Pool(processes=num_readers))

        files_pbar = stack.enter_context(tqdm.tqdm(desc="Reading files", unit=" files", unit_scale=True, total=len(paths)))
        docs_pbar = stack.enter_context(tqdm.tqdm(desc="Indexing documents", unit=" docs", unit_scale=True))

        writer_fn = partial(index.writer, num_threads=num_indexers, heap_size=heap_size)
        writer = writer_fn()

        docs_queue: Queue[list[Document]] = (manager := Manager()).Queue(queue_size)

        fn = partial(read_file_for_indexing, docs_queue=docs_queue, batch_size=reader_batch_size)
        async_results = [
            reader_pool.apply_async(fn, [p], callback=lambda _: files_pbar.update(1))
            for p in paths
        ]
        # for p in paths:
        #     fn(p)

        indexed_count = 0
        while any(not r.ready() for r in async_results) or not docs_queue.empty():
            # check if there are any documents to index
            if docs_queue.empty():
                time.sleep(0.1)
            else:
                batch = docs_queue.get()
                for doc in batch:
                    writer.add_document(doc)
                    indexed_count += 1

                if indexed_count >= indexer_batch_size:
                    docs_pbar.update(indexed_count)
                    indexed_count = 0
                    writer.commit()

        for r in async_results:
            r.wait()

        if indexed_count:
            docs_pbar.update(indexed_count)
            writer.commit()
        writer.wait_merging_threads()


def make_index_parser(parser: argparse.ArgumentParser | None = None):
    parser = parser or argparse.ArgumentParser(INDEX_DESCRIPTION)
    parser.add_argument(
        "-d",
        "--documents",
        type=str,
        required=True,
        nargs="+",
        help="The documents to index. Can be any glob pattern supported by smart-open library."
    )
    parser.add_argument(
        "-i",
        "--index-path",
        type=str,
        help="The path to the index. If not provided, an in-memory index will be used."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If the index already exists, delete it and create a new one."
    )
    parser.add_argument(
        "-n",
        "--num-readers",
        type=int,
        default=1,
        help="The number of readers to use."
    )
    parser.add_argument(
        "-N",
        "--num-indexers",
        type=int,
        default=1,
        help="The number of indexers to use."
    )
    parser.add_argument(
        "-b",
        "--reader-batch-size",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-B",
        "--indexer-batch-size",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-H",
        "--heap-size",
        type=int,
        default=1024 * 1024 * 1024,
    )
    parser.add_argument(
        "-q",
        "--queue-size-per-thread",
        type=int,
        default=125,
        help="The size of the queue to use for storing documents."
    )
    return parser


def index_data(args: argparse.Namespace):
    set_start_method("spawn")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    index = create_index(args.index_path, reuse=not args.force)
    logger.info("Created index" + (f" stored at {args.index_path}" if args.index_path else " in memory"))

    files = list_paths(args.documents, num_workers=args.num_readers)
    logger.info(f"Found {len(files)} files to index")

    # add_paths_to_index(args.index_path, files, num_workers=args.num_workers, batch_size=args.batch_size)
    read_many_and_index(
        index,
        paths=files,
        num_readers=args.num_readers,
        num_indexers=args.num_indexers,
        indexer_batch_size=args.indexer_batch_size,
        reader_batch_size=args.reader_batch_size,
        heap_size=args.heap_size,
        queue_size=args.queue_size_per_thread * args.num_readers,
    )
    logger.info("Indexed all documents")


if __name__ == "__main__":
    index_data(make_index_parser().parse_args())
