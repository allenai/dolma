import json
import multiprocessing
from argparse import ArgumentParser
from contextlib import ExitStack
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Tuple, Union

import smart_open
from dolma.core.parallel import BaseParallelProcessor


class RemoveEmptyDocumentsProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: "Queue[Union[Tuple[int, ...], None]]",
        /,
        files: int = 0,
        read_docs: int = 0,
        written_docs: int = 0,
    ):
        return super().increment_progressbar(queue, files=files, read_docs=read_docs, written_docs=written_docs)

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

        update_every_n_lines = 1
        read_docs = written_docs = 0

        with ExitStack() as stack:
            # open source and destination files
            source_file = stack.enter_context(smart_open.open(source_path, "rt"))
            destination_file = stack.enter_context(smart_open.open(destination_path, "wt"))
            for ln in source_file:
                # we first load the json document
                document = json.loads(ln)
                read_docs += 1

                # we check if the document is
                # empty, and if it is, we skip it
                if document["text"].strip() == "":
                    continue

                # if the document is not empty,
                # we write it to output
                destination_file.write(ln)
                written_docs += 1

                # we update the progress bar every
                # update_every_n_lines
                if read_docs >= update_every_n_lines:
                    cls.increment_progressbar(
                        queue,
                        read_docs=read_docs,
                        written_docs=written_docs,
                    )
                    read_docs = written_docs = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_every_n_lines *= 2

            # we update the progress bar one last time
            cls.increment_progressbar(
                queue,
                files=1,
                read_docs=read_docs,
                written_docs=written_docs,
            )


def parse_args():
    ag = ArgumentParser()
    ag.add_argument("-s", "--source-prefix", type=str, required=True)
    ag.add_argument("-d", "--destination-prefix", type=str, required=True)
    ag.add_argument("-n", "--num-processes", type=int, default=1)
    ag.add_argument("-u", "--debug", action="store_true")
    ag.add_argument("-t", "--temp-dir", type=str, default=None)
    return ag.parse_args()


def main():
    args = parse_args()

    with TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        # create the processor
        processor = RemoveEmptyDocumentsProcessor(
            source_prefix=args.source_prefix,
            destination_prefix=args.destination_prefix,
            metadata_prefix=tmpdir,
            num_processes=args.num_processes,
            debug=args.debug,
        )

        # run the processor
        processor()


if __name__ == "__main__":
    main()
