from hashlib import md5
import datasets
import smart_open
import datetime
from typing import Optional
import json
import tqdm


dataset_name = "meta-math/MetaMathQA"
version = "v0"
split = "train"
destination = (
    f"s3://ai2-llm/pretraining-data/sources/{dataset_name.replace("/", "_")}/"
    f"{version}/documents/{split}/0000.jsonl.gz"
)

def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def main():
    dataset = datasets.load_dataset(dataset_name, split=split)

    d = datetime.datetime(2023, 10, 7)

    with smart_open.open(destination, "wt") as f:
        for row in tqdm.tqdm(dataset, desc="Processing dataset"):
            doc_id = md5(json.dumps(row).encode("utf-8")).hexdigest()
            text = row["query"] + "\n" + row["response"]
            source = f"{dataset_name}_{row['type']}_{split}"
            added = format_to_dolma_timestamp()
            created = format_to_dolma_timestamp(d)

            output = {
                "text": text,
                "id": doc_id,
                "source": source,
                "added": added,
                "created": created,
                "version": version,
                "meta": {**row}
            }
            f.write(json.dumps(output) + "\n")

if __name__ == "__main__":
    main()
