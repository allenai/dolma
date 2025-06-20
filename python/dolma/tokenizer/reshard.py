"""
# Adding tool to reshard npy files based on minimum desired size.

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

import csv
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import urlparse

import numpy as np
import smart_open
import yaml
from tqdm import tqdm

from dolma.core.loggers import get_logger
from dolma.tokenizer.tokenizer import Tokenizer

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


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


def merge_group(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    dtype: np.dtype,
):
    """
    Given a list of paths, merge them into a single memmap.
    """
    npy_destination = Path(destination)
    csv_destination = npy_destination.with_suffix(".csv.gz")
    total_size = sum(p.size for p in paths)

    npy_destination.parent.mkdir(parents=True, exist_ok=True)

    target_memmap = np.memmap(npy_destination, mode="w+", shape=(total_size // dtype.itemsize,), dtype=dtype)

    bytes_offset = row_offset = 0
    with smart_open.open(csv_destination, "w", encoding="utf-8") as f:
        for path in paths:
            rw = csv.writer(f)
            source_memmap = np.memmap(path.npy_path, mode="r", dtype=dtype, shape=(path.size // dtype.itemsize,))
            target_memmap[bytes_offset : bytes_offset + source_memmap.shape[0]] = source_memmap
            target_memmap.flush()

            row_count = 0
            with smart_open.open(path.csv_path, "r", encoding="utf-8") as g:
                rd = csv.reader(g)
                for row in rd:
                    start, end, id_, src, idx = row
                    rw.writerow([int(start) + bytes_offset, int(end) + bytes_offset, id_, src, int(idx)])
                    row_count += 1

            bytes_offset += source_memmap.shape[0]
            row_offset += row_count
            del source_memmap


def group_paths_by_max_size(
    paths: list[TokensMetadataPaths],
    max_size_bytes: int,
) -> list[list[TokensMetadataPaths]]:
    """
    Group paths by max size.
    """
    counts: dict[TokensMetadataPaths, int] = {p: int(c) for p, c in Counter(paths).items()}
    logger.info(
        "Found %s unique paths from %s files; max repetition is %s",
        len(counts),
        len(paths),
        max(counts.values()),
    )

    grouped_paths: list[list[TokensMetadataPaths]] = []
    while len(counts) > 0:
        # add a fresh group
        grouped_paths.append([])

        # partition in groups of max_num_files
        for path, _ in sorted(counts.items(), key=lambda x: -x[1]):
            if sum(p.size for p in grouped_paths[-1]) + path.size > max_size_bytes:
                grouped_paths.append([path])
            else:
                grouped_paths[-1].append(path)

        # decrease counts, remove paths with 0 count.
        counts = {path: new_count for path, count in counts.items() if (new_count := count - 1) > 0}

    logger.info(
        "By size: organized %s files into %s groups of max %.2f GB",
        len(paths),
        len(grouped_paths),
        max_size_bytes / 1024**3,
    )

    return grouped_paths


def weighted_bucket_sample(values: list, count: int, weights: list[float]) -> list[int]:
    """Sample bucket indices with optional weights."""

    # Use the weighted sampling approach
    keys = [random.random() * (w / sum(weights)) for w in weights]
    indices = sorted(range(len(values)), key=lambda i: keys[i], reverse=True)[:count]
    return indices


def group_paths_by_max_num_files(
    paths: list[TokensMetadataPaths],
    max_num_files: int,
) -> list[list[TokensMetadataPaths]]:
    """
    Group paths by max number of files.
    """
    counts = Counter(paths)
    logger.info(
        "Found %s unique paths from %s files; max repetition is %s",
        len(counts),
        len(paths),
        max(counts.values()),
    )

    if (m := max(counts.values())) > max_num_files:
        raise ValueError(f"One or more paths appear {m} times, exceeding max_num_files={max_num_files}")

    grouped_paths: list[list[TokensMetadataPaths]] = [[] for _ in range(max_num_files)]
    # Distribute each element across groups in round-robin fashion
    for element, count in counts.items():
        # sample count buckets out of max_num_files where we could put the element
        # we sample with weights proportional to the number of elements in the bucket,
        # so that we are more likely to sample buckets with fewer elements.
        buckets = weighted_bucket_sample(
            values=list(range(max_num_files)),
            count=count,
            weights=[1 / (len(grouped_paths[i]) + 1) for i in range(max_num_files)],
        )
        for bucket in buckets:
            grouped_paths[bucket].append(element)

    # there is still a change that some buckets might be empty; we remove them.
    grouped_paths = [group for group in grouped_paths if len(group) > 0]

    return grouped_paths


def merge_all_npys(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    max_size_bytes: int | None = None,
    max_num_files: int | None = None,
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer",
    max_workers: int | None = None,
):
    max_workers = max_workers or os.cpu_count() or 1

    if len(paths) == 0:
        raise ValueError("No paths provided")

    destination = Path(destination)

    if Path(tokenizer_name_or_path).exists():
        logger.info("Loading tokenizer from local file %s", tokenizer_name_or_path)
        tokenizer = Tokenizer.from_file(tokenizer_name_or_path)
    else:
        logger.info("Loading tokenizer from Hugging Face %s", tokenizer_name_or_path)
        tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path)

    grouped_paths: list[list[TokensMetadataPaths]]
    if max_num_files is not None:
        grouped_paths = group_paths_by_max_num_files(paths, max_num_files)
    elif max_size_bytes is not None:
        grouped_paths = group_paths_by_max_size(paths, max_size_bytes)
    else:
        raise ValueError("Either max_size_bytes or max_num_files must be provided")

    logger.info(
        "Organizing %s files into %s groups using %s workers...", len(paths), len(grouped_paths), max_workers
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, group in enumerate(grouped_paths):
            future = pool.submit(
                merge_group,
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


@dataclass
class ReshardingPrefixConfig:
    """
    Configuration for a resharding source.

    Can be used to download the files and compute file up/down sampling.
    """

    prefix: str | Path
    sample_rate: float

    def __post_init__(self):
        assert self.sample_rate > 0

    def download(self, local_prefix: str | Path) -> "ReshardingPrefixConfig":
        if urlparse(str(self.prefix)).scheme != "s3":
            return self

        logger.info("Downloading %s to %s", self.prefix, local_prefix)
        remote_prefix_no_star = re.sub(r"(/|/\*)$", "", str(self.prefix))
        local_prefix_no_trailing_slash = str(local_prefix).rstrip("/")
        cmd = ["s5cmd", "cp", "-sp", f"{remote_prefix_no_star}/*", f"{local_prefix_no_trailing_slash}/"]

        logger.info("Running command: %s", " ".join(cmd))
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"s5cmd failed with error: {result.stderr}")
            raise Exception(f"Failed to upload files using s5cmd: {result.stderr}")
        return ReshardingPrefixConfig(
            prefix=local_prefix,
            sample_rate=self.sample_rate,
        )

    def take(self) -> list[TokensMetadataPaths]:
        if urlparse(str(self.prefix)).scheme not in {"file", ""}:
            raise ValueError(
                f"Invalid protocol: {urlparse(str(self.prefix)).scheme}; "
                f"only local paths are supported; download the files first."
            )

        local_prefix = Path(self.prefix)
        paths = []
        for root, _, files in os.walk(local_prefix):
            for file in files:
                if file.endswith(".npy"):
                    npy_path = os.path.join(root, file)
                    csv_path = os.path.join(root, file.replace(".npy", ".csv.gz"))
                    paths.append(TokensMetadataPaths(npy_path, csv_path))

        new_paths = []

        # if the multiplier k is > 1, we first take ⌊k⌋ copies of each path.
        if (repetition_rate := int(math.floor(self.sample_rate))) > 0:
            new_paths.extend(paths * repetition_rate)

        # this is the remaining non-integer part of the sample rate; because the npys are actually uneven in
        # size, the proper way to do this is to use an ILP solver; however, since usually most of the npys are
        # of same size, we can just take a random sample.
        if (residual_frac := self.sample_rate - repetition_rate) > 0:
            new_paths.extend(random.sample(paths, max(1, round(residual_frac * len(paths)))))

        # sort by size
        logger.info("Taking %s paths from %s using %s sample rate", len(new_paths), len(paths), self.sample_rate)
        return new_paths

    def to_dict(self) -> dict:
        return {"prefix": str(self.prefix), "sample_rate": self.sample_rate}

    @classmethod
    def from_dict(cls, d: dict) -> "ReshardingPrefixConfig":
        return cls(**d)


@dataclass
class ReshardingConfig:
    """Base configuration for resharding."""

    source_prefixes: list[ReshardingPrefixConfig]
    destination_prefix: str
    local_tempdir: str | Path | None = None
    max_size_bytes: int | None = None
    max_num_files: int | None = None
    max_workers: int = os.cpu_count() or 1
    random_seed: int = 42
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer"

    def __post_init__(self):
        if self.max_size_bytes is not None and self.max_num_files is not None:
            raise ValueError("Cannot provide both max_size_bytes and max_num_files")
        if self.max_size_bytes is None and self.max_num_files is None:
            raise ValueError("Either max_size_bytes or max_num_files must be provided")

        if self.local_tempdir is None:
            logging.warning("No local tempdir provided; using a temporary directory")
            self.local_tempdir = Path(mkdtemp())
        else:
            self.local_tempdir = Path(self.local_tempdir)

    def to_dict(self) -> dict:
        source_prefixes_dict = [p.to_dict() for p in self.source_prefixes]
        return {**asdict(self), "source_prefixes": source_prefixes_dict}

    @classmethod
    def from_dict(cls, d: dict) -> "ReshardingConfig":
        source_prefixes = [ReshardingPrefixConfig.from_dict(p) for p in d.get("source_prefixes", [])]
        return cls(
            source_prefixes=source_prefixes,
            destination_prefix=str(d["destination_prefix"]),
            local_tempdir=(Path(p) if (p := d.get("local_tempdir")) is not None else None),
            max_size_bytes=(int(s) if (s := d.get("max_size_bytes")) is not None else None),
            max_num_files=(int(n) if (n := d.get("max_num_files")) is not None else None),
            max_workers=int(d.get("max_workers", 1)),
            random_seed=int(d.get("random_seed", 42)),
        )

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ReshardingConfig":
        if file_path == "-":
            return cls.from_dict(yaml.safe_load(sys.stdin))

        with open(file_path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))


def upload_to_s3(local_prefix: str | Path, remote_prefix: str, max_workers: int):
    """
    Upload a local directory to S3.
    """
    if urlparse(remote_prefix).scheme != "s3":
        return

    local_prefix_no_star = re.sub(r"(/|/\*)$", "", str(local_prefix))
    remote_prefix_no_trailing_slash = str(remote_prefix).rstrip("/")
    cmd = ["s5cmd", "cp", "-sp", f"{local_prefix_no_star}/*", f"{remote_prefix_no_trailing_slash}/"]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"s5cmd failed with error: {result.stderr}")
        raise Exception(f"Failed to upload files using s5cmd: {result.stderr}")


def reshard(config: ReshardingConfig):
    random.seed(config.random_seed)

    try:
        local_tempdir = Path(config.local_tempdir or mkdtemp())
        local_tempdir.mkdir(parents=True, exist_ok=True)

        local_output_dir = (
            local_tempdir / "output"
            if urlparse(config.destination_prefix).scheme == "s3"
            else Path(config.destination_prefix)
        )

        # download the files
        source_prefixes = [
            source_prefix.download(local_tempdir / f"input/{i:06d}")
            for i, source_prefix in enumerate(config.source_prefixes)
        ]

        # get repetition aware samples
        source_paths = [path for source_prefix in source_prefixes for path in source_prefix.take()]

        # make destination directory
        local_output_dir.mkdir(parents=True, exist_ok=True)

        # merge the files
        merge_all_npys(
            source_paths,
            destination=local_output_dir,
            max_size_bytes=config.max_size_bytes,
            max_num_files=config.max_num_files,
            max_workers=config.max_workers,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
        )

        # upload the files
        upload_to_s3(
            local_prefix=local_output_dir,
            remote_prefix=config.destination_prefix,
            max_workers=config.max_workers,
        )

    finally:
        shutil.rmtree(local_tempdir)


def main():
    config = ReshardingConfig.from_file(sys.argv[1])
    reshard(config)


if __name__ == "__main__":
    main()
