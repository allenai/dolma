"""
Convert a dataset that is in the format used by the WIMBD project to the format used in Dolma.

Author: Luca Soldaini (@soldni)
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
from contextlib import ExitStack
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union

import msgspec
import smart_open
from dolma.core.parallel import BaseParallelProcessor
from dolma.core.paths import join_path
from dolma.core.runtime import _make_paths_from_prefix


class WimbdInputSpec(msgspec.Struct):
    text: str
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


class DolmaInputSpec(msgspec.Struct):
    id: str
    text: str
    source: str
    version: str
    added: str
    created: str
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class WimbdToDolmaProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: "Queue[Union[Tuple[int, ...], None]]",
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(
            queue,
            files=files,
            documents=documents,
        )

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        decoder = msgspec.json.Decoder(WimbdInputSpec)
        encoder = msgspec.json.Encoder()

        source_name = kwargs.pop("source_name", None)
        source_version = kwargs.pop("source_version", "v0")
        assert isinstance(source_name, str), "source_name should be a string"

        documents_count = 0
        update_interval = 1

        current_time = convert_timestamp(datetime.datetime.now())

        with ExitStack() as stack:
            source = stack.enter_context(smart_open.open(source_path, "rt"))
            destination = stack.enter_context(smart_open.open(destination_path, "wb"))

            for line in source:
                data = decoder.decode(line)
                id_ = hashlib.md5(data.text.encode()).hexdigest()
                converted_document = DolmaInputSpec(
                    id=id_,
                    text=data.text,
                    source=source_name,
                    version=source_version,
                    added=current_time,
                    created=current_time,
                    metadata=data.metadata,
                )
                destination.write(encoder.encode(converted_document) + b"\n")  # type: ignore
                documents_count += 1

                if documents_count % update_interval == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(queue, documents=documents_count)

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

        cls.increment_progressbar(queue, files=1, documents=documents_count)


def main(
    source: Union[List[str], str],
    destination: str,
    source_name: str,
    source_version: str = "v0",
    num_workers: int = 1,
    debug: bool = False,
) -> None:
    # make source always a list
    source = [source] if isinstance(source, str) else source
    with TemporaryDirectory() as tempdir:
        if len(source) > 1:
            dest_prefixes = _make_paths_from_prefix(source, join_path(None, destination))
            meta_prefixes = _make_paths_from_prefix(source, join_path(None, tempdir))
        else:
            dest_prefixes = [destination]
            meta_prefixes = [tempdir]

        processor = WimbdToDolmaProcessor(
            source_prefix=source,
            destination_prefix=dest_prefixes,
            metadata_prefix=meta_prefixes,
            num_processes=num_workers,
            debug=debug,
        )
        processor(source_name=source_name, source_version=source_version)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", type=str, required=True, help="glob pattern for source files", nargs="+")
    ap.add_argument("-d", "--destination", type=str, required=True, help="destination prefix")
    ap.add_argument("-w", "--num-workers", type=int, default=1, help="number of workers")
    ap.add_argument("-n", "--source-name", required=True, help="source name")
    ap.add_argument("-v", "--source-version", default="v0", help="version")
    ap.add_argument("--debug", action="store_true", help="debug mode")
    opts = ap.parse_args()

    print(json.dumps(vars(opts), indent=2, sort_keys=True))
    return opts


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
