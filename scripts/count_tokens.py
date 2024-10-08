'''Count number of tokens in OLMo config'''

from pathlib import Path
import sys
from typing import Union
from yaml import safe_load
import smart_open
import boto3


def load_config(path: Union[Path, str]) -> dict:
    with smart_open.open(path, mode="rt") as f:
        return dict(safe_load(f))


def count_tokens(config_path: Union[Path, str]) -> int:
    config = load_config(config_path)
    s3_client = boto3.client('s3')

    for path in config["data"]["paths"]:
        if path.startswith("s3://"):
            bucket, key = path.lstrip("s3://").split("/", 1)
            response = s3_client.head_object(Bucket=bucket, Key=key)
            file_size = response["ContentLength"]
        else:
            file_size = Path(path).stat().st_size
        breakpoint()

    return 0


def main():
    path = sys.argv[1]
    print(count_tokens(path))


if __name__ == "__main__":
    main()
