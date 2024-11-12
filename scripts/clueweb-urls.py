from collections import Counter
from hashlib import md5
import itertools
import multiprocessing
from tempfile import TemporaryDirectory
from dolma.core.parallel import BaseParallelProcessor, BaseProgressBar, QueueType
from dolma.core.data_types import InputSpecWithMetadataAndAttributes
from dolma.core.paths import glob_path
from dolma.core.utils import add_compression
from typing import Counter as CounterType
import msgspec
import json
import smart_open
from urllib.parse import urlparse


class UrlCollectorPbar(BaseProgressBar):
    documents: int = 0
    files: int = 0
    failed: int = 0


class UrlCollectorCounter(BaseParallelProcessor):
    PROGRESS_BAR_CLS = UrlCollectorPbar

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs):
        add_compression()

        counter: "CounterType[str]" = Counter()

        with smart_open.open(source_path, "rb") as f_in, UrlCollectorPbar(queue) as pbar:
            for line in f_in:
                data = json.loads(line)
                pbar.documents += 1

                try:
                    url = data["URL"]
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
    base_path = "s3://ai2-llm/pretraining-data/sources/clueweb/raw/disk1/txt/en/en00/*"
    base_dst = "s3://ai2-llm/stats/clueweb/B"

    with TemporaryDirectory() as tmpdir:
        src_paths, dst_paths, meta_paths = [], [], []
        for path in glob_path(base_path + '/*.json.gz'):
            src_paths.append(path)
            dst_paths.append(f"{base_dst}")
            meta_paths.append(f"{tmpdir}")

        print(f"Found {len(src_paths):,} files to process")

        counter = UrlCollectorCounter(
            source_prefix=src_paths,
            destination_prefix=dst_paths,
            metadata_prefix=meta_paths,
            num_processes=multiprocessing.cpu_count() - 1,
            debug=False,
            skip_source_glob=True,
            ignore_existing=True,
        )
        counter()

    collated = Counter()
    for path in glob_path(f"{base_dst}/*.json"):
        with smart_open.open(path, "rt") as f:
            collated.update(json.load(f))

    sorted_collated = dict(sorted(collated.items(), key=lambda x: x[1], reverse=True))

    with smart_open.open(f"{base_dst}/urls.json", "wt") as f:
        f.write(json.dumps(sorted_collated, indent=2))

    print(json.dumps(sorted_collated, indent=2))


if __name__ == "__main__":
    main()
