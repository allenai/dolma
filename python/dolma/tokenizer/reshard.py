"""
# Resharding Npy Files

Given a prefix with npy and csv.gz files, this script will merge the npy files so that the output
satisfies a minimum size constraint.


## Usage

In case we wanna reshard from S3, we can do:

```bash
python -m dolma.tokenizer.reshard -s s3://bucket/prefix -d s3://bucket/prefix-resharded
```

If you wanna customize which local tempdir to use, you can do:

```bash
python -m dolma.tokenizer.reshard -s s3://bucket/prefix -d s3://bucket/prefix-resharded -l /mnt/raid0/tempdir
```

If you want to reshard locally, you can do:

```
python -m dolma.tokenizer.reshard -s /path/to/local/prefix -d /path/to/local/prefix-resharded
```

To change number of workers, you can do:

```bash
python -m dolma.tokenizer.reshard -s s3://bucket/prefix -d s3://bucket/prefix-resharded -w 10
```

## Contact info

Author: Luca Soldaini
Email:  luca@soldaini.net
"""

import argparse
import logging
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from csv import reader, writer
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import boto3
import numpy as np
import smart_open
from tqdm import tqdm

from dolma.core.loggers import get_logger
from dolma.tokenizer.tokenizer import Tokenizer

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


@dataclass(frozen=True)
class TokensMetadataPaths:
    npy_path: str
    csv_path: str

    def __post_init__(self):
        assert self.npy_path.endswith(".npy")
        assert self.csv_path.endswith(".csv.gz")
        assert Path(self.npy_path).stem == Path(Path(self.csv_path).stem).stem

    @property
    def size(self) -> int:
        return os.path.getsize(self.npy_path)


def get_local_paths(prefix: str) -> list[TokensMetadataPaths]:
    paths = []
    for root, _, files in os.walk(prefix):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                csv_path = os.path.join(root, file.replace(".npy", ".csv.gz"))
                paths.append(TokensMetadataPaths(npy_path, csv_path))
    return paths


def download_file(remote_path: str, local_prefix: str | Path, client: "S3Client") -> TokensMetadataPaths:
    assert remote_path.endswith(".npy")
    local_npy = os.path.join(local_prefix, basename := os.path.basename(remote_path))
    local_csv_gz = os.path.join(local_prefix, basename.replace(".npy", ".csv.gz"))
    bucket, key = (p := urlparse(remote_path)).netloc, p.path.lstrip("/")
    client.download_file(bucket, key, local_npy)
    client.download_file(bucket, key.replace(".npy", ".csv.gz"), local_csv_gz)
    return TokensMetadataPaths(local_npy, local_csv_gz)


def map_local_paths(local_prefix: str | Path) -> list[TokensMetadataPaths]:
    paths = []
    for root, _, files in os.walk(local_prefix):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                csv_path = os.path.join(root, file.replace(".npy", ".csv.gz"))
                paths.append(TokensMetadataPaths(npy_path, csv_path))
    return paths


def download_remote_paths(
    remote_path: str,
    local_prefix: str | Path,
    client: "S3Client | None" = None,
    max_workers: int | None = None,
) -> list[TokensMetadataPaths]:

    client = client or boto3.client("s3")

    max_workers = max_workers or os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []

        bucket, prefix = (p := urlparse(remote_path)).netloc, p.path.lstrip("/")
        for file_attrs in client.list_objects(Bucket=bucket, Prefix=prefix).get("Contents", []):
            assert isinstance(file_attrs, dict)
            if (key := file_attrs.get("Key")) is None:
                continue

            if not key.endswith(".npy"):
                continue

            remote_path = f"s3://{bucket}/{key}"
            local_path = local_prefix / Path(key).relative_to(prefix)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            future = pool.submit(
                download_file, remote_path=remote_path, local_prefix=local_path.parent, client=client
            )
            futures.append(future)

        logger.info("Downloading %s files from `%s` using %s workers...", len(futures), remote_path, max_workers)

        all_paths: list[TokensMetadataPaths] = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            try:
                all_paths.append(future.result())
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e

        logger.info(
            "Found %s NumPy memmaps; total: %.2f GB",
            len(all_paths),
            sum(p.size for p in all_paths) / 1024 / 1024 / 1024,
        )

    return all_paths


def merge_single_npy(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    dtype: np.dtype,
):
    npy_destination = Path(destination)
    csv_destination = npy_destination.with_suffix(".csv.gz")
    total_size = sum(p.size for p in paths)

    npy_destination.parent.mkdir(parents=True, exist_ok=True)

    target_memmap = np.memmap(npy_destination, mode="w+", shape=(total_size // dtype.itemsize,), dtype=dtype)

    bytes_offset = row_offset = 0
    with smart_open.open(csv_destination, "w", encoding="utf-8") as f:
        for path in paths:
            rw = writer(f)
            source_memmap = np.memmap(path.npy_path, mode="r", dtype=dtype, shape=(path.size // dtype.itemsize,))
            target_memmap[bytes_offset : bytes_offset + source_memmap.shape[0]] = source_memmap

            row_count = 0
            with smart_open.open(path.csv_path, "r", encoding="utf-8") as g:
                rd = reader(g)
                for row in rd:
                    start, end, id_, src, idx = row
                    rw.writerow([int(start) + bytes_offset, int(end) + bytes_offset, id_, src, int(idx)])
                    row_count += 1

            bytes_offset += source_memmap.shape[0]
            row_offset += row_count
            del source_memmap


def merge_all_npys(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    max_size: int = 1024 * 1024 * 1024,
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer",
    max_workers: int | None = None,
):
    max_workers = max_workers or os.cpu_count() or 1

    destination = Path(destination)

    if Path(tokenizer_name_or_path).exists():
        logger.info("Loading tokenizer from local file %s", tokenizer_name_or_path)
        tokenizer = Tokenizer.from_file(tokenizer_name_or_path)
    else:
        logger.info("Loading tokenizer from Hugging Face %s", tokenizer_name_or_path)
        tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path)

    # We group together npys that need to be merged.
    grouped_paths: list[list[TokensMetadataPaths]] = [[]]
    for paired_path in paths:
        if sum(p.size for p in grouped_paths[-1]) + paired_path.size > max_size:
            grouped_paths.append([paired_path])
        else:
            grouped_paths[-1].append(paired_path)

    logger.info(
        "Organizing %s files into %s groups using %s workers...", len(paths), len(grouped_paths), max_workers
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, group in enumerate(grouped_paths):
            future = pool.submit(
                merge_single_npy,
                paths=group,
                destination=destination / f"{i:06d}.npy",
                dtype=tokenizer.dtype,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging files"):
            try:
                future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e

        logger.info("Done merging NumPy memmaps.")


def upload_single_file(
    local_path: str | Path,
    remote_path: str,
    client: "S3Client | None" = None,
):
    client = client or boto3.client("s3")
    bucket, key = (p := urlparse(remote_path)).netloc, p.path.lstrip("/")
    client.upload_file(Filename=str(local_path), Bucket=bucket, Key=key)


def upload_to_s3(
    local_prefix: str | Path,
    remote_prefix: str,
    client: "S3Client | None" = None,
    max_workers: int | None = None,
):
    client = client or boto3.client("s3")
    local_prefix = Path(local_prefix)

    bucket, prefix = (p := urlparse(remote_prefix)).netloc, p.path.lstrip("/")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for root, _, files in os.walk(local_prefix):

            for file in files:
                rel_path = (Path(root) / file).relative_to(local_prefix)
                dst = f"s3://{bucket}/{prefix}/{rel_path}"
                future = pool.submit(
                    upload_single_file,
                    local_path=os.path.join(root, file),
                    remote_path=dst,
                    client=client,
                )
                futures.append(future)

        logger.info("Uploading %s files to `%s` using %s workers...", len(futures), remote_prefix, max_workers)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading files"):
            try:
                future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e

        logger.info("Done uploading files to S3.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-prefix", type=str, required=True)
    parser.add_argument("-d", "--destination-prefix", type=str, required=True)
    parser.add_argument("-l", "--local-tempdir", type=str, default=None)
    parser.add_argument("-m", "--min-size", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("-w", "--max-workers", type=int, default=None)
    parser.add_argument("-r", "--random-seed", type=int, default=42)
    parser.add_argument("-t", "--tokenizer-name-or-path", type=str, default="allenai/dolma2-tokenizer")
    return parser.parse_args()


def is_remote_path(path: str) -> bool:
    prot = urlparse(path).scheme
    if prot == "s3":
        return True
    elif prot == "file" or prot == "":
        return False
    else:
        raise ValueError(f"Invalid protocol: {prot}; only S3 or local paths are supported.")


def main(
    source_prefix: str,
    destination_prefix: str,
    local_tempdir: str | None = None,
    min_size: int = 1024 * 1024 * 1024,
    max_workers: int | None = None,
    random_seed: int = 42,
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer",
):
    random.seed(random_seed)

    try:
        tempdir = Path(local_tempdir) if local_tempdir else Path(mkdtemp())
        if is_remote_path(source_prefix):
            local_paths = download_remote_paths(
                source_prefix,
                local_prefix=tempdir / "input",
                max_workers=max_workers,
            )
        else:
            local_paths = map_local_paths(source_prefix)

        random.shuffle(local_paths)

        merge_destination = tempdir / "output" if is_remote_path(destination_prefix) else destination_prefix
        merge_all_npys(
            local_paths,
            destination=merge_destination,
            max_size=min_size,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_workers=max_workers,
        )
        if is_remote_path(destination_prefix):
            upload_to_s3(
                local_prefix=tempdir / "output",
                remote_prefix=destination_prefix,
                max_workers=max_workers,
            )
    finally:
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    args = parse_args()
    main(
        source_prefix=args.source_prefix,
        destination_prefix=args.destination_prefix,
        min_size=args.min_size,
        max_workers=args.max_workers,
        random_seed=args.random_seed,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        local_tempdir=args.local_tempdir,
    )
