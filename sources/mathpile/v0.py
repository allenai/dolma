from contextlib import ExitStack
from hashlib import md5
from tempfile import TemporaryDirectory
from typing import Any, Optional
import datetime
from queue import Queue
import json
from multiprocessing import cpu_count

import smart_open
from dolma.core.parallel import BaseParallelProcessor


def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"



class MathpileProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: Queue,
        /,
        files: int = 0,
        docs: int = 0,
        words: int = 0,
    ):
        """
        This method is to update the progress bar. We keep
        track of three things:
        - files: the number of files processed
        - read_docs: the number of documents read in
        - written_docs: the number of documents written out
            (i.e., the number of documents that are not empty)
        """
        super().increment_progressbar(
            queue,
            files=files,
            docs=docs,
            words=words,
        )

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

        update_every_n_lines = 10_000
        docs = 0
        words = 0
        with ExitStack() as stack:
            # open source and destination files
            source_file = stack.enter_context(
                smart_open.open(source_path, "rt")
            )
            destination_file = stack.enter_context(
                smart_open.open(destination_path, "wt")
            )

            # Set a fixed creation date
            created = datetime.datetime(2023, 12, 29)

            *_, source, split, subset, fn = source_path.split("/")
            for ln in source_file:
                # we first load the json document
                document = json.loads(ln)
                docs += 1

                docid = md5((ln + source + split + subset).encode('utf-8')).hexdigest()

                metadata = {}

                if "text" in document:
                    text = document.pop("text")
                elif "question" in document and "answers" in document:
                    question = document.pop("question")
                    answers = document.pop("answers")

                    text = f"{question.pop('Title').strip()}\n{question.pop('Body').strip()}\n\n"
                    metadata.update({f"question_{k}": v for k, v in question.items()})

                    for answer in answers:
                        text += f"{answer.pop('Body').strip()}\n\n"
                        metadata.update({f"answer_{k}": v for k, v in answer.items()})
                else:
                    raise ValueError(f"Unknown document type: {document}")

                subset = document.pop("subset")

                output = {
                    "text": text.strip(),
                    "source": f"{source}_{subset}_{split}",
                    "added": format_to_dolma_timestamp(),
                    "created": format_to_dolma_timestamp(created),
                    "id": docid,
                    "metadata": {**document, **metadata, "subset": subset, "split": split, "source": source}
                }

                words += len(text.split())

                # if the document is not empty,
                # we write it to output
                destination_file.write(json.dumps(output) + "\n")

                # we update the progress bar every
                # update_every_n_lines
                if docs > update_every_n_lines:
                    cls.increment_progressbar(queue, docs=docs, words=words)
                    docs = 0
                    words = 0

            # we update the progress bar one last time
            cls.increment_progressbar(
                queue,
                files=1,
                docs=docs,
                words=words,
            )


def main():
    with TemporaryDirectory() as tmpdir:
        # create the processor
        processor = MathpileProcessor(
            source_prefix="/data/mathpile/raw/*/*/*/*.gz",
            destination_prefix="/data/mathpile/v0",
            metadata_prefix=tmpdir,
            num_processes=cpu_count() - 2,
        )

        # run the processor
        processor()


if __name__ == "__main__":
    main()
