from contextlib import ExitStack
import datetime
import itertools
import multiprocessing
import re
from tempfile import TemporaryDirectory
from typing import Any, Dict, List
from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.paths import glob_path, make_relative, mkdir_p
import smart_open
from msgspec import field, defstruct, Struct
from msgspec.json import Decoder, Encoder


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class ClueWeb22Spec(Struct):
    URL: str
    URL_hash: str
    Language: str
    ClueWeb22_ID: str
    Clean_Text: str


class Clueweb22RawProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(cls, queue: "QueueType", /, files: int = 0, documents: int = 0) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):

        breakpoint()

        parser = Decoder(ClueWeb22Spec)
        writer = Encoder()

        docs_cnt = 0
        update_interval = 1
        max_file_size = 1024 * 1024 * 1024 * 5  # 5GB
        output_cnt = 0

        mkdir_p(destination_path)

        with ExitStack() as stack:
            # with smart_open.open(destination_path, "wb") as wf:
            wf = stack.enter_context(smart_open.open(f"{destination_path}/{output_cnt:05d}.jsonl.gz", "wb"))
            for path in source_path:
                with smart_open.open(path, "rt") as rf:
                    for raw in rf:
                        doc = parser.decode(raw)

                        breakpoint()

                        if re.match(r"clueweb22\-en00*", doc.clueweb22_id):
                            category = "ClueWeb22-B"
                        elif re.match(r"clueweb22\-en00[1-9]*", doc.clueweb22_id):
                            category = "ClueWeb22-A"
                        else:
                            category = "ClueWeb22-L"

                        output = {
                            "id": doc.clueweb22_id,
                            "created": convert_timestamp(datetime.datetime(year=2023, month=3, day=22)),
                            "added": convert_timestamp(datetime.datetime.now()),
                            "source": category,
                            "text": doc.clean_text,
                            "metadata": {"url": doc.url, "url_hash": doc.url_hash, "language": doc.language},
                        }
                        breakpoint()
                        wf.write(writer.encode(output) + b"\n")

                        docs_cnt += 1

                        if docs_cnt % update_interval == 0:
                            # update the progress bar every 1000 documents to prevent buffering
                            cls.increment_progressbar(queue, documents=docs_cnt)
                            docs_cnt = 0

                            if queue.qsize() >= multiprocessing.cpu_count():
                                # double the update interval if the queue is full
                                update_interval *= 2

                        if wf.tell() > max_file_size:
                            stack.pop_all().close()
                            output_cnt += 1
                            wf = stack.enter_context(
                                smart_open.open(f"{destination_path}/{output_cnt:05d}.jsonl.gz", "wb")
                            )

                cls.increment_progressbar(queue, files=1)
            cls.increment_progressbar(queue, documents=docs_cnt)


def group_paths(base_src: str) -> List[List[str]]:
    dest: Dict[str, List[str]] = {}
    for path in glob_path(base_src):
        grouped_path, *_ = path.rsplit("/", 2)
        dest.setdefault(grouped_path, []).append(path)
    return list(dest.values())


def main():
    base_src = "s3://ai2-llm/pretraining-data/sources/clueweb/raw/disk1/txt/en/*/*/*.json.gz"
    base_dst = "s3://ai2-llm/pretraining-data/sources/clueweb/v0/documents/en"

    with TemporaryDirectory() as tmp_dir:
        all_grouped_paths = group_paths(base_src)
        _, relative = make_relative([p for paths in all_grouped_paths for p in paths])
        all_dst_paths = [f"{base_dst}/{rel}" for rel in relative]
        all_meta_path = [f"{tmp_dir}/{rel}" for rel in relative]
        breakpoint()

        proc = Clueweb22RawProcessor(
            source_prefix=all_grouped_paths,
            destination_prefix=all_dst_paths,
            metadata_prefix=all_meta_path,
            num_processes=multiprocessing.cpu_count() - 1,
            debug=True,
        )
        proc()


if __name__ == "__main__":
    main()
