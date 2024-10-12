from collections import namedtuple
from functools import partial
import logging
import json
import shutil
from queue import Queue, Empty as EmptyError
import argparse
import tqdm
from pathlib import Path
from tantivy import Document, SchemaBuilder, Index
import fsspec
import smart_open
import time
from urllib.parse import urlparse
from multiprocessing import Pool, Manager, set_start_method, Process
from multiprocessing.managers import BaseManager
from contextlib import ExitStack


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


def read_file_for_indexing(file_path: str, docs_queue: Queue[Document], batch_size: int = 1_000):
    batch = []
    with smart_open.open(file_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            doc = Document(id=row["id"], text=row["text"])
            batch.append(doc)

            if len(batch) >= batch_size:
                docs_queue.put_nowait(batch)
                batch = []

    if batch:
        docs_queue.put_nowait(batch)


def read_many_and_index(
    index: Index,
    paths: list[str],
    num_readers: int = 1,
    num_indexers: int = 1,
    indexer_batch_size: int = 1_000,
    reader_batch_size: int = 1_000,
    heap_size: int = 1024 * 1024 * 1024,
):
    with ExitStack() as stack:
        reader_pool = stack.enter_context(Pool(processes=num_readers))
        docs_pbar = stack.enter_context(tqdm.tqdm(desc="Indexing documents", unit=" docs", unit_scale=True))
        writer = index.writer(
            num_threads=num_indexers,
            heap_size=heap_size,
        )
        docs_queue: Queue[Document] = (manager := Manager()).Queue()
        async_results = reader_pool.map_async(
            partial(read_file_for_indexing, docs_queue=docs_queue, batch_size=reader_batch_size),
            paths,
        )

        indexed_count = 0
        while not async_results.ready() or not docs_queue.empty():
            # check if there are any documents to index
            if docs_queue.empty():
                time.sleep(0.1)
            else:
                for doc in docs_queue.get():
                    writer.add_document(doc)
                    indexed_count += 1
                docs_pbar.update(indexed_count)
                if indexed_count >= indexer_batch_size:
                    indexed_count = 0
                    writer.commit()
        if indexed_count:
            writer.commit()

        writer.wait_merging_threads()


def create_index(path: str | Path | None = None, reuse: bool = False) -> Index:
    # Declaring our schema.
    schema_builder = SchemaBuilder()
    schema_builder.add_text_field("text", stored=True)
    schema_builder.add_text_field("id", stored=True)
    schema = schema_builder.build()

    if path:
        path = Path(path) / "index"
        if not reuse and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    # Creating our index (in memory)
    index = Index(schema, path=str(path), reuse=reuse)
    return index


def parse_arguments():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def main():
    set_start_method("spawn")

    args = parse_arguments()
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
    )
    logger.info("Indexed all documents")


if __name__ == "__main__":
    main()
