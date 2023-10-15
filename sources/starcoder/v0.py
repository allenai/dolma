import hashlib
import os
from datetime import datetime
from multiprocessing import cpu_count
from tempfile import TemporaryDirectory
from typing import Any, Dict

import smart_open
from msgspec.json import Encoder
from necessary import necessary

from dolma.core.parallel import BaseParallelProcessor, QueueType

with necessary("pyarrow"):
    import pyarrow.parquet as pq


TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class StarcoderFormatConvert(BaseParallelProcessor):

    @classmethod
    def increment_progressbar(  # type: ignore
        cls, queue: QueueType, /, files: int = 0, documents: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs: Any
    ):
        docs = 0
        print_every = 10_000
        enc = Encoder()

        destination_path = destination_path.replace(".parquet", ".jsonl.gz")

        *_, extension, __ = destination_path.split("/")

        with smart_open.open(destination_path, "wb") as output_file:
            table = pq.read_table(source_path)
            column_names = [str(c) for c in table.column_names]
            rows = (dict(zip(column_names, row)) for row in zip(*(table[c] for c in column_names)))
            for row in rows:
                text = str(row.pop('content', ''))
                if not text:
                    continue

                id_ = str(row.pop('id', hashlib.md5(text.encode()).hexdigest()))
                metadata = {key: str(value) for key, value in row.items()}
                metadata['extension'] = extension

                doc = {
                    'id': id_,
                    'text': text,
                    'source': 'starcoder',
                    'created': TIMESTAMP,
                    'added': TIMESTAMP,
                    'metadata': metadata
                }
                output_file.write(enc.encode(doc) + b'\n')  # type: ignore
                docs += 1

                if docs % print_every == 0:
                    cls.increment_progressbar(queue, documents=print_every)

        cls.increment_progressbar(queue, documents=docs % print_every, files=1)


def main():
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    with TemporaryDirectory() as tempdir:
        src = "s3://ai2-llm/pretraining-data/sources/starcoder/raw/**/*.parquet"
        dst = "s3://ai2-llm/pretraining-data/sources/starcoder/v0/documents"

        converter = StarcoderFormatConvert(
            source_prefix=src,
            destination_prefix=dst,
            metadata_prefix=tempdir,
            num_processes=max(cpu_count() // 4, 1),
            debug=False
        )
        converter()


if __name__ == "__main__":
    main()
