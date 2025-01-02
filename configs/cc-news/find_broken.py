from argparse import ArgumentParser
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Tuple, Union

import smart_open
from dolma.core.parallel import BaseParallelProcessor


class FindBrokenFilesProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: "Queue[Union[Tuple[int, ...], None]]",
        /,
        files: int = 0,
        docs: int = 0,
    ):
        return super().increment_progressbar(queue, files=files, docs=docs)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue,
        **kwargs: Any,
    ):
        """
        This method is called for each file. It reads the file
        line by line, and writes to the destination file only
        if the document is not empty.
        """

        try:
            with smart_open.open(source_path, mode="rt", encoding="utf-8") as f:
                cnt = 0
                for _ in f:
                    cnt += 1
                    if cnt >= 1000:
                        cls.increment_progressbar(queue, docs=cnt)
                        cnt = 0
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error {e} in file {source_path}")

        if cnt > 0:
            cls.increment_progressbar(queue, docs=cnt, files=1)


def parse_args():
    ag = ArgumentParser()
    ag.add_argument("-s", "--source-prefix", type=str, required=True)
    ag.add_argument("-n", "--num-processes", type=int, default=1)
    ag.add_argument("-u", "--debug", action="store_true")
    ag.add_argument("-t", "--temp-dir", type=str, default=None)
    return ag.parse_args()


def main():
    args = parse_args()

    with TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        # create the processor
        processor = FindBrokenFilesProcessor(
            source_prefix=args.source_prefix,
            destination_prefix=tmpdir,
            metadata_prefix=tmpdir,
            num_processes=args.num_processes,
            debug=args.debug,
        )

        # run the processor
        processor()


if __name__ == "__main__":
    main()
