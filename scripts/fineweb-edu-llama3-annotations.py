import datasets
import smart_open
import datetime
from typing import Optional
from tqdm import tqdm
from hashlib import sha1
import json
from contextlib import ExitStack


prefix = "s3://ai2-llm/pretraining-data/sources/fineweb-edu-llama3-annotations/v0/documents"


def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations", split="train")
cnt_files = cnt_docs = 0

with ExitStack() as stack:
    f = stack.enter_context(smart_open.open(f"{prefix}/{cnt_files:04d}.jsonl.zst", "wt"))
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing"):
        created = datetime.datetime.fromtimestamp(example['metadata']['date'] / 1000)
        doc = {
            "text": example["text"],
            "id": sha1(json.dumps(example).encode("utf-8")).hexdigest(),
            "added": format_to_dolma_timestamp(),
            "created": format_to_dolma_timestamp(created),
            "source": "fineweb-edu-llama3-annotations",
            "metadata": {"score": example["score"], "prompt": example["prompt"], **example["metadata"]},
        }
        f.write(json.dumps(doc) + "\n")
        cnt_docs += 1
        if cnt_docs % 100_000 == 0:
            cnt_files += 1
            stack.pop_all()
            f = stack.enter_context(smart_open.open(f"{prefix}/{cnt_files:04d}.jsonl.zst", "wt"))
