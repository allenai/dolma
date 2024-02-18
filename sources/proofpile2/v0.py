from calendar import c
from contextlib import ExitStack
import datetime
import multiprocessing
import os
import random
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import smart_open
from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.paths import glob_path, make_relative, mkdir_p, split_path
from msgspec.json import Encoder, Decoder


BOT = "1970-01-01T00:00:00.000Z"


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def parse_timestamp(date_str: str, date_format: str):
    # Parse the date string into a datetime object
    parsed_date = datetime.datetime.strptime(date_str.strip(), date_format)
    return parsed_date


def add_zstd_support():
    def _handle_zst(file_obj, mode):
        try:
            from pyzstd import ZstdFile
        except ImportError as e:
            raise RuntimeError("cannot add to smart_open: pyzstd is not installed") from e
        return ZstdFile(file_obj, mode)


    #     def _handle_bz2(file_obj, mode):
    # from bz2 import BZ2File
    # result = BZ2File(file_obj, mode)
    # tweak_close(result, file_obj)
    # return result

    smart_open.register_compressor(".zst", _handle_zst)


class ProofPile2(BaseParallelProcessor):

    @classmethod
    def increment_progressbar(cls, queue: "QueueType", /, files: int = 0, documents: int = 0) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        all_grouped_paths: List[List[str]] = kwargs.pop("all_grouped_paths")
        process_paths = all_grouped_paths[int(source_path)]

        add_zstd_support()

        parser = Decoder()
        writer = Encoder()

        update_interval = 1
        docs_cnt = 0

        with ExitStack() as stack:
            wf = stack.enter_context(smart_open.open(destination_path, "wb"))

            for path in process_paths:
                _, (*_, fn) = split_path(path)
                name, *_ = fn.split(".")
                rf = stack.enter_context(smart_open.open(path, "rb"))

                for i, row in enumerate(rf):
                    doc = parser.decode(row)
                    text = doc.pop("text")
                    metadata_str: Union[str, dict] = doc.pop("metadata", None) or {}
                    metadata = {
                        **(parser.decode(metadata_str) if isinstance(metadata_str, str) else metadata_str),
                        **(doc.pop("meta", None) or {})
                    }

                    if "date" in doc:
                        created = parse_timestamp(doc["date"], "%Y-%m-%d %H:%M:%S")
                    elif "timestamp" in metadata:
                        created = parse_timestamp(metadata["timestamp"], "%Y-%m-%dT%H:%M:%S")
                    else:
                        # "max_stars_repo_stars_event_max_datetime": "2022-01-28T06:04:15.000Z",
                        stars_repo_stars_event = parse_timestamp(
                            date_str=metadata.get("max_stars_repo_stars_event_max_datetime", None) or BOT,
                            date_format="%Y-%m-%dT%H:%M:%S.%fZ",
                        )
                        # "max_issues_repo_issues_event_max_datetime": "2020-04-11T11:03:39.000Z",
                        issues_repo_issues_event = parse_timestamp(
                            date_str=metadata.get("max_issues_repo_issues_event_max_datetime", None) or BOT,
                            date_format="%Y-%m-%dT%H:%M:%S.%fZ",
                        )
                        # "max_forks_repo_forks_event_max_datetime": "2021-11-29T13:23:07.000Z",
                        forks_repo_forks_event = parse_timestamp(
                            date_str=metadata.get("max_forks_repo_forks_event_max_datetime", None) or BOT,
                            date_format="%Y-%m-%dT%H:%M:%S.%fZ",
                        )
                        created = max(stars_repo_stars_event, issues_repo_issues_event, forks_repo_forks_event)

                    output = {
                        "id": f"proofpile-{name}-{i}",
                        "text": text,
                        "created": convert_timestamp(created),
                        "added": convert_timestamp(datetime.datetime.now()),
                        "doc": {**metadata, **doc}
                    }
                    wf.write(writer.encode(output) + b"\n")  # pyright: ignore
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
        random.seed(3920)
        grouped_source_path: List[List[str]] = []
        grouped_dest_path: List[str] = []
        grouped_meta_path: List[str] = []

        subgroup_prefix = None
        for path in glob_path(self.src_prefixes[0]):
            _, (*_, group, split, _) = split_path(path)
            current_prefix = f"{group}/{split}"

            if subgroup_prefix != current_prefix or (split == "train" and len(grouped_source_path[-1]) >= 5):
                subgroup_prefix = current_prefix
                grouped_source_path.append([])
                grouped_dest_path.append(
                    f"{self.dst_prefixes[0]}/{group}/{split}/{len(grouped_dest_path):03d}.jsonl.gz"
                )
                grouped_meta_path.append(
                    f"{self.meta_prefixes[0]}/{group}/{split}/{len(grouped_meta_path):03d}.jsonl"
                )
            grouped_source_path[-1].append(path)

        # shuffle the operations to prevent a single group from being processed first
        random.shuffle(operations := [str(i) for i in range(len(grouped_source_path))])

        self.num_processes = min(len(grouped_source_path), self.num_processes)
        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=operations,
            all_destination_paths=grouped_dest_path,
            all_metadata_paths=grouped_meta_path,
            all_grouped_paths=grouped_source_path,
        )


def main():
    with TemporaryDirectory() as tmp_dir:
        ProofPile2(
            source_prefix="s3://ai2-llm/pretraining-data/sources/proof-pile-2/raw/*/*/*.zst",
            destination_prefix="s3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents",
            metadata_prefix=tmp_dir,
            num_processes=multiprocessing.cpu_count(),
            debug=False,
        )()


if __name__ == "__main__":
    main()
