from calendar import c
import datetime
import multiprocessing
import os
import random
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List, Optional, Tuple

import smart_open
from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.paths import glob_path, make_relative, mkdir_p, split_path
from fastparquet import ParquetFile
from msgspec.json import Encoder, Decoder


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def parse_timestamp(date_str):
    # Define the format of the date string
    date_format = "%Y-%m-%d %H:%M:%S"

    # Parse the date string into a datetime object
    try:
        parsed_date = datetime.datetime.strptime(date_str.strip(), date_format)
        return parsed_date
    except ValueError as e:
        print(f"Error parsing date: {e}")
        return None


class OpenWebMath(BaseParallelProcessor):

    @classmethod
    def increment_progressbar(
        cls, queue: "QueueType", /, files: int = 0, documents: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        all_grouped_paths: List[List[str]] = kwargs.pop("all_grouped_paths")
        process_paths = all_grouped_paths[int(source_path)]

        parser = Decoder()
        writer = Encoder()

        update_interval = 1
        docs_cnt = 0

        with smart_open.open(f"{destination_path}", "wb") as wf:
            for path in process_paths:
                _, (*_, fn) = split_path(path)
                name, *_ = fn.split(".")
                coll = ParquetFile(path).to_pandas()

                for i, doc in coll.sample(frac=1).iterrows():
                    metadata = parser.decode(doc.metadata)
                    output = {
                        "id": f"openwebmath-{name}-{i}",
                        "text": doc.text,
                        "created": parse_timestamp(doc.date),
                        "added": convert_timestamp(datetime.datetime.now()),
                        "doc": {"url": doc["url"], **metadata}
                    }
                    wf.write(writer.encode(output) + b"\n")     # pyright: ignore
                    docs_cnt += 1

                    if docs_cnt % update_interval == 0:
                        # update the progress bar every 1000 documents to prevent buffering
                        cls.increment_progressbar(queue, documents=docs_cnt)
                        docs_cnt = 0

                        if queue.qsize() >= multiprocessing.cpu_count():
                            # double the update interval if the queue is full
                            update_interval *= 2

                cls.increment_progressbar(queue, files=1)
            cls.increment_progressbar(queue, documents=docs_cnt)

    def __call__(self, **kwargs):
        grouped_source_path: List[List[str]] = []
        grouped_dest_path: List[str] = []
        grouped_meta_path: List[str] = []

        random.shuffle(raw_paths := list(glob_path(self.src_prefixes[0])))
        for path in raw_paths:
            if not(grouped_source_path) or len(grouped_source_path[-1]) >= 5:
                grouped_source_path.append([])
                grouped_dest_path.append(f"{self.dst_prefixes[0]}/{len(grouped_dest_path):03d}.jsonl.gz")
                grouped_meta_path.append(f"{self.meta_prefixes[0]}/{len(grouped_meta_path)}")
            grouped_source_path[-1].append(path)

        self.num_processes = min(len(grouped_source_path), self.num_processes)
        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=[str(i) for i in range(len(grouped_source_path))],
            all_destination_paths=grouped_dest_path,
            all_metadata_paths=grouped_meta_path,
            all_grouped_paths=grouped_source_path,
        )


def main():
    with TemporaryDirectory() as tmp_dir:
        OpenWebMath(
            source_prefix=f"{os.path.expanduser('~')}/open-web-math/data/*.parquet",
            destination_prefix="s3://ai2-llm/pretraining-data/sources/open-web-math/v0/documents",
            metadata_prefix=tmp_dir,
            num_processes=multiprocessing.cpu_count(),
            debug=False,
        )()


if __name__ == "__main__":
    main()
