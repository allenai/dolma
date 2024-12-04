from hashlib import md5
import datasets
import smart_open
import datetime
from typing import Optional
import json
import tqdm
from contextlib import ExitStack

dataset_name = "code-search-net/code_search_net"
version = "v0"
destination = f"s3://ai2-llm/pretraining-data/sources/{dataset_name.split("/")[1]}/{version}/documents"
max_docs_per_file = 100_000

def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def main():
    created = format_to_dolma_timestamp(datetime.datetime(2019, 9, 20))

    with tqdm.tqdm(unit=" docs", unit_scale=True) as pbar, ExitStack() as stack:
        for language in ["python", "java", "javascript", "go", "ruby", "php"]:
            for split in ["train", "validation", "test"]:
                pbar.set_description(f"Processing {language}/{split}")
                fn = 0
                cnt = 0
                path = f"{destination}/{split}/{language}/{fn:04d}.jsonl.gz"
                print(f"\nCreating new output file {path}")
                f = stack.enter_context(smart_open.open(path, "wt"))
                dataset = datasets.load_dataset(dataset_name, language, split=split)
                for row in dataset:
                    doc = {
                        "id": md5(row["func_code_url"].encode("utf-8")).hexdigest(),
                        "text": row.pop("whole_func_string"),
                        "source": f"{dataset_name}_{language}_{split}",
                        "added": format_to_dolma_timestamp(),
                        "created": created,
                        "metadata": row
                    }
                    f.write(json.dumps(doc) + "\n")

                    pbar.update(1)
                    cnt += 1
                    if cnt >= max_docs_per_file:
                        fn += 1
                        cnt = 0
                        stack.pop_all().close()
                        path = f"{destination}/{split}/{language}/{fn:04d}.jsonl.gz"
                        print(f"\nCreating new output file {path}")
                        f = stack.enter_context(smart_open.open(path, "wt"))
                stack.pop_all().close()



if __name__ == "__main__":
    main()
