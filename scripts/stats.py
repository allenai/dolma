import argparse
import multiprocessing
import os
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, Union


from dolma.core.parallel import BaseParallelProcessor
from dolma.core.data_types import InputSpec
import msgspec
import smart_open



class DolmaWebStats(BaseParallelProcessor):

    @classmethod
    def increment_progressbar(
        cls,
        queue: Queue[Union[Tuple[int, ...], None]],
        /,
        files: int = 0,
        documents: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue[Union[Tuple[int, ...], None]],
        **kwargs: Any
    ):
        # for the data sheet, what statistics you think we should include? I could do # of docs, # tokens, distribution of URLs, pronouns, s2 FOS, stack languages?
        decoder = msgspec.json.Decoder(InputSpec)

        with smart_open.open(source_path, "rb") as source_file:
            for line in source_file:
                document = decoder.decode(line)
                breakpoint()


def web(
    num_workers: int = multiprocessing.cpu_count(),
    debug: bool = False,
    **process_single_kwargs: Any
) -> None:

    with TemporaryDirectory() as tempdir:
        documents = [
            "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/c4/*.gz",
            "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_*/*.gz",
        ]

        stats = [
            "s3://ai2-llm/stats/olmo-mix/v1/web/c4",
            "s3://ai2-llm/stats/olmo-mix/v1/web/common-crawl",
        ]

        metadata = [
            os.path.join(tempdir, "c4"),
            os.path.join(tempdir, "common-crawl"),
        ]


        processor = DolmaWebStats(
            source_prefix=documents,
            destination_prefix=stats,
            metadata_prefix=tempdir,
            num_processes=num_workers,
            debug=debug
        )
        processor(**process_single_kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stat", choices=['web'])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = ap.parse_args()

    if args.stat == "web":
        web(num_workers=args.num_workers, debug=args.debug)
