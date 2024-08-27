import datasets
import smart_open
import datetime
from typing import Optional
from tqdm import tqdm
import json


prefix = "s3://ai2-llm/pretraining-data/sources/gsm8k/v0/documents"


def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

for split in ["train", "test"]:
    for subset in ("main", "socratic"):
        dataset = datasets.load_dataset("gsm8k", subset, split=split)
        with smart_open.open(f"{prefix}/{subset}/{split}/0.jsonl.zst", "wt") as f:
            for i, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {subset} {split}"):
                question = example['question'].strip()
                reasoning, answer = example['answer'].split('####', 1)
                text = f"Question: {question}\nReasoning: {reasoning.strip()}\nAnswer: {answer.strip()}"
                doc = {
                    "text": text,
                    "id": f"{subset}_{split}_{i}",
                    "added": format_to_dolma_timestamp(),
                    # 27 Oct 2021
                    "created": format_to_dolma_timestamp(datetime.datetime(2021, 10, 27)),
                    "source": f"gsm8k-{subset}-{split}",
                    "metadata": dict(**example)
                }
                f.write(json.dumps(doc) + "\n")
