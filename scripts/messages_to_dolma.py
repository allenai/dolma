import argparse
import uuid
from datetime import datetime
from typing import Optional
from contextlib import ExitStack
import json

from datasets import load_dataset
import tqdm
import smart_open

def convert_timestamp(d: Optional[datetime]) -> str:
    if d is None:
        d = datetime.utcnow()
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def process_and_upload(
    dataset_name_or_path: str,
    messages_column_name: str,
    save_prefix: str,
    num_rows_per_file: int,
    extension: str = "zst",
):
    """
    Process the dataset and upload to S3 in Dolma format.
    """

    for split, dataset in load_dataset(dataset_name_or_path).items():
        file_cnt = row_cnt = 0
        filename_gen = lambda i: f"{save_prefix}/documents/{split}/part-{i:05d}.jsonl.{extension}"
        with ExitStack() as stack:
            fileobj = stack.enter_context(smart_open.open(filename_gen(file_cnt), 'wt'))
            for row in tqdm.tqdm(dataset, desc=f"{dataset_name_or_path}/{split}"):
                text = "".join([
                    msg["content"] + (" " if not msg["content"].endswith((" ", "\n", "\t")) else "")
                    for msg in row[messages_column_name]
                ]).lstrip()
                id_ = row.get("id", str(uuid.uuid4()))
                doc = {
                    "id": id_,
                    "text": text,
                    "source": "huggingface_dataset",
                    "added": convert_timestamp(datetime.utcnow()),
                    "created": convert_timestamp(datetime.utcnow()),
                    "metadata": {
                        "original_messages": [dict(e) for e in row[messages_column_name]]
                    }
                }
                fileobj.write(json.dumps(doc) + '\n')
                row_cnt += 1
                if row_cnt >= num_rows_per_file:
                    stack.pop_all().close()
                    file_cnt += 1
                    row_cnt = 0
                    fileobj = stack.enter_context(smart_open.open(filename_gen(file_cnt), 'wt'))


def main():
    parser = argparse.ArgumentParser(description="Transform HuggingFace dataset to Dolma format and upload to S3")
    parser.add_argument("--dataset", required=True, help="Name of the HuggingFace dataset")
    parser.add_argument("--column", default="messages", help="Name of the column containing messages")
    parser.add_argument("--destination", required=True, help="S3 destination path")
    parser.add_argument("--rows-per-file", type=int, default=100000, help="Number of rows per output file")
    parser.add_argument("--extension", default="zst", help="Extension of the output file")

    args = parser.parse_args()

    process_and_upload(
        dataset_name_or_path=args.dataset,
        messages_column_name=args.column,
        save_prefix=args.destination,
        num_rows_per_file=args.rows_per_file,
        extension=args.extension
    )

if __name__ == "__main__":
    main()
