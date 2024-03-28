import hashlib
import os
import datetime
from multiprocessing import cpu_count
from tempfile import TemporaryDirectory
from typing import Any, Dict

import smart_open
from msgspec.json import Encoder
from necessary import necessary

from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.utils import split_words

with necessary("pyarrow"):
    import pyarrow.parquet as pq


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class IaPdBooksConverter(BaseParallelProcessor):

    @classmethod
    def increment_progressbar(  # type: ignore
        cls, queue: QueueType, /, files: int = 0, documents: int = 0, words: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents, words=words)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        docs = 0
        print_every = 1
        words = 0
        enc = Encoder()

        destination_path = destination_path.replace(".parquet", ".jsonl.gz")

        with smart_open.open(destination_path, "wb") as output_file:
            table = pq.read_table(source_path)
            column_names = [str(c) for c in table.column_names]
            rows = (dict(zip(column_names, row)) for row in zip(*(table[c] for c in column_names)))
            for i, row in enumerate(rows):
                text = (str(row["full_text"]) or "").strip()
                if not text:
                    continue

                metadata = {
                    "title": (str(row["title"]) or "").strip(),
                    "author": (str(row["author"]) or "").strip(),
                    "year": int(str(row["year"]) or "0"),
                    "page_count": int(str(row["page_count"]) or "0"),
                    "openlibrary_edition": (str(row["openlibrary_edition"]) or "").strip(),
                    "openlibrary_work": (str(row["openlibrary_work"]) or "").strip(),
                }

                doc = {
                    "id": (str(row["ocaid"]) or str(i)),
                    "text": text,
                    "source": "openlibrary_pd_books_en",
                    "created": convert_timestamp(datetime.datetime(metadata['year'], 1, 1)),
                    "added": convert_timestamp(datetime.datetime.now()),
                    "metadata": metadata,
                }
                output_file.write(enc.encode(doc) + b"\n")  # type: ignore
                docs += 1
                words += text.count(" ")

                if docs % print_every == 0:
                    cls.increment_progressbar(queue, documents=print_every, words=words)
                    docs = words = 0

                    if queue.qsize() > cpu_count():
                        print_every = print_every * 2

        cls.increment_progressbar(queue, documents=docs, words=words, files=1)


def main():
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    with TemporaryDirectory() as tempdir:
        src = "/home/ubuntu/openlibrary_pd_books_en/data/*.parquet"
        dst = "s3://ai2-llm/pretraining-data/sources/openlibrary_pd_books_en/v0/documents"

        converter = IaPdBooksConverter(
            source_prefix=src,
            destination_prefix=dst,
            metadata_prefix=tempdir,
            num_processes=cpu_count(),
            debug=False,
        )
        converter()


if __name__ == "__main__":
    main()
