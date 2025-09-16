import boto3
from urllib.parse import urlparse
from pathlib import Path
import tqdm
import csv
import argparse


paths = {
    "v01": [
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/s2pdf/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/",
    ],
    "v02": [
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/",
        "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/",
    ],
}


def get_size_of_prefix(prefix: str, ext: str = ".npy") -> int:
    bucket, prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
    s3 = boto3.client("s3")

    total_size = 0
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        for obj in response.get("Contents", []):
            if "Key" not in obj:
                continue

            if not obj["Key"].endswith(ext):
                continue

            if "Size" not in obj:
                continue

            total_size += int(obj["Size"])

        if response.get("IsTruncated", False):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return total_size


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--version", type=str, required=True, help="version", choices=paths.keys())
    return ap.parse_args()


def main():
    args = parse_arguments()

    s3 = boto3.client("s3")

    all_paths: list[dict] = []

    for path in tqdm.tqdm(paths[args.version], desc="Getting sizes"):
        bucket, prefix = (p := urlparse(path)).netloc, p.path.lstrip("/")
        *_, prefix_subset = prefix.rstrip("/").split("/")

        continuation_token = None

        while True:
            if continuation_token:
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            for obj in response.get("Contents", []):
                if "Key" not in obj:
                    continue

                if not obj["Key"].endswith(".npy"):
                    continue

                if "Size" not in obj:
                    continue

                category = obj["Key"].rsplit("/", 1)[0].replace(prefix, "").strip("/")
                entry = {
                    "key": f"s3://{bucket}/{obj['Key']}",
                    "size": int(obj["Size"]),
                    "subset": prefix_subset,
                    "category": category
                }

                all_paths.append(entry)

            if response.get("IsTruncated", False):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

    with open(Path(__file__).parent / f"dolma2-0625-{args.version}.csv", "w") as f:
        wr = csv.DictWriter(f, fieldnames=["key", "size", "subset", "category"])
        wr.writeheader()
        for entry in all_paths:
            wr.writerow(entry)


if __name__ == "__main__":
    main()
