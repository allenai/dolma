from contextlib import ExitStack
from hashlib import sha1
import json
import multiprocessing
import random
from typing import Any, Dict, List, Optional
from dolma.core.parallel import BaseParallelProcessor, BaseProgressBar, QueueType
from dolma.core.paths import glob_path
import smart_open
import datetime


def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class SampleProgressBar(BaseProgressBar):
    files: int = 0
    documents: int = 0
    sampled: int = 0

class SampleDclmProcessor(BaseParallelProcessor):
    PROGRESS_BAR_CLS = SampleProgressBar

    @classmethod
    def process_batch(
        cls,
        source_paths: List[str],
        destination_paths: List[str],
        queue: QueueType,
        kwargs: List[Dict[str, Any]]
    ):

        with ExitStack() as stack:
            pb = stack.enter_context(SampleProgressBar(queue))
            h = sha1(json.dumps(destination_paths).encode()).hexdigest()
            output_path = f"s3://ai2-llm/pretraining-data/sources/dclm/v0_sample/{h}.jsonl.zstd"
            g = stack.enter_context(smart_open.open(output_path, "wt"))

            for path in source_paths:
                f = stack.enter_context(smart_open.open(path, "rt"))
                for line in f:
                    pb.documents += 1

                    if random.random() > 0.0004:
                        continue

                    id_ = sha1(line.encode()).hexdigest()

                    row = json.loads(line)
                    created = datetime.datetime.strptime(row['metadata']['WARC-Date'], "%Y-%m-%dT%H:%M:%SZ")
                    doc = {
                        "id": id_,
                        "text": row.pop("text"),
                        "metadata": {"warc": row.pop("metadata")},
                        "source": "dclm",
                        "added": format_to_dolma_timestamp(),
                        "created": format_to_dolma_timestamp(created),
                    }
                    doc.update(row)
                    g.write(json.dumps(doc) + "\n")
                    pb.sampled += 1

                pb.files += 1


class SampleDolmaV1(BaseParallelProcessor):
    PROGRESS_BAR_CLS = SampleProgressBar

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):

        existing = kwargs.get("existing", set())
        score = float(kwargs.get("score", 0.))

        with ExitStack() as stack:
            pb = stack.enter_context(SampleProgressBar(queue))
            g = stack.enter_context(smart_open.open(destination_path.replace('.gz', '.zstd'), "wt"))
            f = stack.enter_context(smart_open.open(source_path, "rt"))

            for line in f:
                pb.documents += 1

                if random.random() > score:
                    continue

                row = json.loads(line)
                if row["id"] in existing:
                    continue

                g.write(line)
                pb.sampled += 1

            pb.files += 1


class SampleDolmaV17(BaseParallelProcessor):
    PROGRESS_BAR_CLS = SampleProgressBar

    @classmethod
    def process_batch(cls, source_paths: List[str], destination_paths: List[str], queue: QueueType, kwargs: List[Dict[str, Any]]):

        existing = kwargs[0].get("existing", set())
        score = float(kwargs[0].get("score", 0.0))

        with ExitStack() as stack:
            pb = stack.enter_context(SampleProgressBar(queue))
            h = sha1(json.dumps(destination_paths).encode()).hexdigest()
            output_path = f"s3://ai2-tylerm-experimental/experiments/rephrase/samples/dolma-cc/v17/{h}.jsonl.zstd"
            g = stack.enter_context(smart_open.open(output_path, "wt"))

            for path in source_paths:
                f = stack.enter_context(smart_open.open(path, "rt"))
                for line in f:
                    pb.documents += 1

                    if random.random() > score:
                        continue

                    row = json.loads(line)
                    if row["id"] in existing:
                        continue

                    g.write(line)
                    pb.sampled += 1

                pb.files += 1

import boto3


def list_s3_files(path):
    """
    List all files in an S3 bucket with the given prefix.

    :param bucket_name: Name of the S3 bucket
    :param prefix: Prefix to filter objects (optional)
    :return: List of file names
    """
    bucket_name, prefix = path.lstrip("s3://").split("/", 1)

    s3 = boto3.client("s3")

    # Use paginator to handle buckets with more than 1000 objects
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                yield f"s3://{bucket_name}/{obj['Key']}"


# Example usage:
# files = list_s3_files('my-bucket', 'my-prefix/')
# for file in files:
#     print(file)

def get_ids(path):
    ids = []
    with smart_open.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["doc_id"])
    return ids


if __name__ == "__main__":
    # s = SampleDclmProcessor(
    #     source_prefix="s3://ai2-llm/pretraining-data/sources/common-crawl/v1-small/documents/*.gz",
    #     destination_prefix="s3://ai2-llm/pretraining-data/sources/dclm/v0_sample",
    #     metadata_prefix="/tmp/dclm_metadata",
    #     num_processes=350,
    #     batch_size=100,
    #     debug=False,
    # )
    # s()
    multiprocessing.set_start_method("spawn")
    existing_paths = [
        "s3://ai2-tylerm-experimental/experiments/rephrase/v0/",
        "s3://ai2-tylerm-experimental/experiments/rephrase/v1/dolma-10k/train/gpt_fix/",
        "s3://ai2-tylerm-experimental/experiments/rephrase/v1/dolma-10k/train/",
        "s3://ai2-tylerm-experimental/experiments/rephrase/v1/dolma-10k/valid/",
    ]
    paths = [p for r in existing_paths for p in list_s3_files(r)]
    print(f"Found {len(paths)} existing files.")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        resp = pool.map(get_ids, paths)
        existing = set([item for sublist in resp for item in sublist])

    print(f"Loaded {len(existing)} existing documents.")

    # s = SampleDolmaV0(
    #     source_prefix="s3://ai2-llm/pretraining-data/sources/common-crawl/v1-small/documents/*.gz",
    #     destination_prefix="s3://ai2-tylerm-experimental/experiments/rephrase/samples/dolma-cc/v1-small",
    #     metadata_prefix="/tmp/dolma_metadata_v1",
    #     num_processes=350,
    #     debug=False,
    # )
    # s(
    #     existing=existing,
    #     # score=0.0033,
    #     score=0.0015,
    # )

    s = SampleDolmaV17(
        source_prefix="s3://ai2-llm/pretraining-data/sources/olmo-mix/danyh-compiled-v1_7/documents/cc_en_*/*.gz",
        destination_prefix="s3://ai2-tylerm-experimental/experiments/rephrase/samples/dolma-cc/v17",
        metadata_prefix="/tmp/dolma_metadata_v17",
        num_processes=350,
        batch_size=5,
        debug=False,
    )
    s(
        existing=existing,
        # score=0.0033,
        score=0.0005,
    )
