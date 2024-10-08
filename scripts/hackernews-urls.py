from collections import Counter
from hashlib import md5
import itertools
import multiprocessing
from tempfile import TemporaryDirectory
from dolma.core.parallel import BaseParallelProcessor, BaseProgressBar, QueueType
from dolma.core.data_types import InputSpecWithMetadataAndAttributes
from dolma.core.paths import glob_path, delete_file
from dolma.core.utils import add_compression
from typing import Counter as CounterType
import msgspec
import json
import smart_open
from urllib.parse import urlparse
import s3fs

import fastparquet as fp


class UrlCollectorPbar(BaseProgressBar):
    documents: int = 0
    files: int = 0
    skipped: int = 0
    failed: int = 0


class UrlCollectorCounter(BaseParallelProcessor):
    PROGRESS_BAR_CLS = UrlCollectorPbar

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs):
        add_compression()

        counter: "CounterType[str]" = Counter()

        fs = s3fs.S3FileSystem()

        with UrlCollectorPbar(queue) as pbar:
            pq = fp.ParquetFile(source_path, open_with=fs.open)
            for row_group in pq.iter_row_groups(columns=['url']):
                for url in row_group['url']:
                    if url is None or not (url := str(url)).strip():
                        pbar.skipped += 1
                        continue

                    try:
                        domain = urlparse(url).netloc
                        counter.update([domain])
                        pbar.documents += 1
                    except Exception:
                        pbar.failed += 1

            pbar.files += 1

        destination_dir, _ = destination_path.rsplit("/", 1)
        destination_path = f"{destination_dir}/{md5(source_path.encode()).hexdigest()}.json"

        with smart_open.open(destination_path, "wt") as f_out:
            f_out.write(json.dumps(counter, indent=2))


def main():
    base_path = "s3://ai2-llm/pretraining-data/sources/OpenPipe_hacker-news/raw/data"
    base_dst = "s3://ai2-llm/stats/OpenPipe_hacker-news/raw"

    with TemporaryDirectory() as tmpdir:
        src_paths, dst_paths, meta_paths = [], [], []
        for path in glob_path(base_path + '/*.parquet'):
            src_paths.append(path)
            dst_paths.append(f"{base_dst}")
            meta_paths.append(f"{tmpdir}")

        print(f"Found {len(src_paths):,} files to process")

        counter = UrlCollectorCounter(
            source_prefix=src_paths,
            destination_prefix=dst_paths,
            metadata_prefix=meta_paths,
            num_processes=multiprocessing.cpu_count(),
            debug=False,
            skip_source_glob=True,
            ignore_existing=True,
        )
        counter()

    collated = Counter()
    for path in glob_path(f"{base_dst}/*.json"):
        with smart_open.open(path, "rt") as f:
            collated.update(json.load(f))

    delete_file(f"{base_dst}/urls.json", ignore_missing=True)

    sorted_collated = dict(sorted(collated.items(), key=lambda x: x[1], reverse=True))

    with smart_open.open(f"{base_dst}/urls.json", "wt") as f:
        f.write(json.dumps(sorted_collated, indent=2))

    print(f"Top 100 domains:")
    for k, v in itertools.islice(sorted_collated.items(), 100):
        print(f"\t{k}: {v:,}")


if __name__ == "__main__":
    main()
