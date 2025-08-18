"""
# Adding tool to reshard npy files based on minimum desired size.

Given a prefix with npy and csv.gz files, this script will merge the npy files so that the output
satisfies a minimum size constraint.

## Contact info

Author: Luca Soldaini
Email:  luca@soldaini.net
"""

import argparse
import csv
from functools import cached_property, partial
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
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
    fraction: float = 1.0

    def __post_init__(self):
        assert self.npy_path.endswith(".npy")
        assert self.csv_path.endswith(".csv.gz")
        assert Path(self.npy_path).stem == Path(Path(self.csv_path).stem).stem
        assert self.fraction > 0 and self.fraction <= 1

    @property
    def size(self) -> int:
        return os.path.getsize(self.npy_path)

# class RingReader:
#     def __init__(self, paths: list[TokensMetadataPaths], ring_size: int):
#         self.paths = paths[:]
#         self.ring_size = ring_size
#         self.memmaps = []
#         self.sizes = []
#         random.shuffle(self.paths)

#     def __iter__(self):
#         while True:
#             if len(self.memmaps) == 0 and len(self.paths)  == 0:
#                 break

#             if len(self.memmaps) < self.ring_size:
#                 new_path = self.paths.pop(0)


# def merge_group_with_block_shuffle(
#     paths: list[TokensMetadataPaths],
#     destination: str | Path,
#     dtype: np.dtype,
#     block_size: int = 100_000_000,
#     ring_size: int = 8,
# ):
#     """
#     Given a list of paths, merge them into a single memmap. Adjacent sequences are periodically shuffled every
#     block_size tokens (by default, 100 million tokens). This is useful to avoid to avoid content from the same
#     file being adjacent in the resulting memmap.

#     Args:
#         paths: List of paths to merge.
#         destination: Path to the destination memmap.
#         dtype: Data type of the memmap.
#         block_size: The block size is the number of tokens to shuffle in each block; by default, we shuffle
#             (approximately) every 100 million tokens.
#         ring_size: The number of files to shuffle from.
#     """
#     npy_destination = Path(destination)
#     csv_destination = npy_destination.with_suffix(".csv.gz")
#     total_size = sum(p.size for p in paths)

#     npy_destination.parent.mkdir(parents=True, exist_ok=True)

#     target_memmap = np.memmap(npy_destination, mode="w+", shape=(total_size // dtype.itemsize,), dtype=dtype)

#     bytes_offset = row_offset = 0
#     with smart_open.open(csv_destination, "w", encoding="utf-8") as f:
#         for path in paths:
#             rw = csv.writer(f)
#             source_memmap = np.memmap(path.npy_path, mode="r", dtype=dtype, shape=(path.size // dtype.itemsize,))
#             target_memmap[bytes_offset : bytes_offset + source_memmap.shape[0]] = source_memmap
#             target_memmap.flush()

#             row_count = 0
#             with smart_open.open(path.csv_path, "r", encoding="utf-8") as g:
#                 rd = csv.reader(g)
#                 for row in rd:
#                     start, end, id_, src, idx = row
#                     rw.writerow([int(start) + bytes_offset, int(end) + bytes_offset, id_, src, int(idx)])
#                     row_count += 1

#             bytes_offset += source_memmap.shape[0]
#             row_offset += row_count
#             del source_memmap


def merge_group(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    dtype: np.dtype,
):
    """
    Given a list of paths, merge them into a single memmap.

    Args:
        paths: List of paths to merge.
        destination: Path to the destination memmap.
        dtype: Data type of the memmap.
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


def reset_random_seed(random_seed: int):
    """Function used to make sure that workers use the same random seed.

    This is needed because the random seed is not thread-/process-safe.
    """
    random.seed(random_seed)


def calculate_number_of_digits_positions_in_filenames(list_of_filenames: list) -> int:
    """
    Calculate the number of digits positions in the filenames.
    """
    # the number of digits in the file names depends on the number of groups. We take the log10 to figure out
    # how many digits we need. for example, if we have 103 groups, we need 3 digits, which is equal to log(103)
    # rounded up. Note that we always round down to closest integer, and then add always add 1. This is the
    # easiest way to use 3 digits even in cases like 100 files.
    return int(np.log10(len(list_of_filenames))) + 1


def merge_all_npys(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    random_seed: int,
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

    digits_in_npy_names = calculate_number_of_digits_positions_in_filenames(grouped_paths)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=partial(reset_random_seed, random_seed=random_seed),
    ) as pool:
        futures = []
        for i, group in enumerate(grouped_paths):
            future = pool.submit(
                merge_group,
                paths=group,
                destination=destination / f"{i:0{digits_in_npy_names}d}.npy",
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

    sample_rate: float | None = None
    target_size: int | None = None
    prefix: str | Path | None = None
    prefixes: list[str | Path] = field(default_factory=list)

    def __post_init__(self):
        if self.prefix is not None:
            if len(self.prefixes) > 0:
                raise ValueError("Cannot provide both prefix and prefixes")
            logger.warning("Deprecation warning: prefix is deprecated; use prefixes instead")
            self.prefixes.append(self.prefix)
        elif len(self.prefixes) == 0:
            raise ValueError("Either prefix or prefixes must be provided")

        if self.sample_rate is not None:
            if self.target_size is not None:
                raise ValueError("Cannot provide both sample_rate and target_size")
            if self.sample_rate <= 0:
                raise ValueError("sample_rate must be greater than 0")
        elif self.target_size is not None:
            if self.target_size <= 0:
                raise ValueError("target_size must be greater than 0")
        else:
            raise ValueError("Either sample_rate or target_size must be provided")

    def download(self, shared_local_prefix: str | Path) -> "ReshardingPrefixConfig":
        local_prefixes: list[str | Path] = []

        digits_in_local_names = calculate_number_of_digits_positions_in_filenames(self.prefixes)

        for i, remote_prefix in enumerate(self.prefixes):
            if urlparse(str(remote_prefix)).scheme != "s3":
                local_prefixes.append(Path(remote_prefix))
                continue

            local_prefix = Path(shared_local_prefix) / f"{i:0{digits_in_local_names}d}"

            logger.info("Downloading %s to %s", remote_prefix, local_prefix)
            if "*" in str(remote_prefix):
                raise ValueError("Wildcard is not supported in prefixes")

            remote_prefix_no_trailing_slash = str(remote_prefix).rstrip("/")
            local_prefix_no_trailing_slash = str(local_prefix).rstrip("/")
            cmd = [
                "s5cmd",
                "cp",
                "-sp",
                f"{remote_prefix_no_trailing_slash}/*",
                f"{local_prefix_no_trailing_slash}/",
            ]

            logger.info("Running command: %s", " ".join(cmd))
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # append to the set of prefixes where we downloaded the files.
            local_prefixes.append(local_prefix)

            if result.returncode != 0:
                print(f"s5cmd failed with error: {result.stderr}")
                raise Exception(f"Failed to upload files using s5cmd: {result.stderr}")

        return ReshardingPrefixConfig(
            prefixes=local_prefixes,
            sample_rate=self.sample_rate,
            target_size=self.target_size,
        )

    def take(self) -> list[TokensMetadataPaths]:
        paths: list[TokensMetadataPaths] = []

        for local_prefix in self.prefixes:
            if urlparse(str(local_prefix)).scheme not in {"file", ""}:
                raise ValueError(
                    f"Invalid protocol: {urlparse(str(local_prefix)).scheme}; "
                    f"only local paths are supported; download the files first."
                )

            local_prefix = Path(local_prefix)

            if not local_prefix.exists():
                raise FileNotFoundError(f"Local prefix {local_prefix} does not exist")

            for root, _, files in os.walk(local_prefix):
                for file in files:
                    if file.endswith(".npy"):
                        npy_path = os.path.join(root, file)
                        csv_path = os.path.join(root, file.replace(".npy", ".csv.gz"))
                        paths.append(TokensMetadataPaths(npy_path, csv_path))

        if len(paths) == 0:
            raise FileNotFoundError(f"No files found in {self.prefixes}")

        new_paths = []

        if self.sample_rate is None:
            # sample rate is not provided, therefore we have to calculate it from the target size.
            assert self.target_size is not None, "this should have been checked in __post_init__"

            # get current size of all the npy files in the prefixes.
            current_size = sum(p.size for p in paths)
            sample_rate = self.target_size / current_size
        else:
            # sample rate is provided, let's just use that!
            sample_rate = self.sample_rate

        # if the multiplier k is > 1, we first take ⌊k⌋ copies of each path.
        if (repetition_rate := int(math.floor(sample_rate))) > 0:
            new_paths.extend(paths * repetition_rate)

        # this is the remaining non-integer part of the sample rate; because the npys are actually uneven in
        # size, the proper way to do this is to use an ILP solver; however, since usually most of the npys are
        # of same size, we can just take a random sample.
        if (residual_frac := sample_rate - repetition_rate) > 0:
            sample = random.sample(paths, max(1, round(residual_frac * len(paths))))
            new_paths.extend(sample)

        # sort by size
        logger.info("Taking %s paths from %s using %s sample rate", len(new_paths), len(paths), sample_rate)
        return new_paths

    def to_dict(self) -> dict:
        return {
            "prefixes": [str(p) for p in self.prefixes],
            "sample_rate": self.sample_rate,
            "target_size": self.target_size,
        }

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

        random.seed(self.random_seed)

    @cached_property
    def tempdir(self) -> Path:
        if self.local_tempdir is None:
            logging.warning("No local tempdir provided; using a temporary directory")
            local_tempdir = Path(mkdtemp())
        else:
            local_tempdir = Path(self.local_tempdir)
        local_tempdir.mkdir(parents=True, exist_ok=True)
        return local_tempdir

    def to_dict(self) -> dict:
        source_prefixes_dict = [p.to_dict() for p in self.source_prefixes]
        return {**asdict(self), "source_prefixes": source_prefixes_dict}

    @classmethod
    def from_dict(cls, d: dict) -> "ReshardingConfig":
        if "source_prefixes" not in d or len(d["source_prefixes"]) == 0:
            raise ValueError("source_prefixes is required")

        source_prefixes = [ReshardingPrefixConfig.from_dict(p) for p in d["source_prefixes"]]

        return cls(
            source_prefixes=source_prefixes,
            destination_prefix=str(d["destination_prefix"]),
            local_tempdir=(Path(p) if (p := d.get("local_tempdir")) is not None else None),
            max_size_bytes=(int(s) if (s := d.get("max_size_bytes")) is not None else None),
            max_num_files=(int(n) if (n := d.get("max_num_files")) is not None else None),
            max_workers=int(d.get("max_workers", 1)),
            random_seed=int(d.get("random_seed", 42)),
            tokenizer_name_or_path=d.get("tokenizer_name_or_path", "allenai/dolma2-tokenizer"),
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



def is_remote(path: str | Path) -> bool:
    """
    Check if a path is remote.

    Returns true if the path is an S3 path, false if it is a local path, and raises an error for other schemes.
    """

    if urlparse(str(path)).scheme in {"file", ""}:
        return False
    elif urlparse(str(path)).scheme == "s3":
        return True
    else:
        raise ValueError(f"Unsupported remote scheme: {urlparse(str(path)).scheme}")



def make_local_source_prefixes(config: ReshardingConfig) -> list[ReshardingPrefixConfig]:

    """
    Create local copies of source prefixes if the destination is remote.

    If the destination prefix is remote (e.g., S3), this function downloads all source
    prefixes to a local temporary directory. If the destination is already local,
    it returns the source prefixes unchanged.

    Args:
        config: The resharding configuration containing source prefixes and destination info.

    Returns:
        A list of ReshardingPrefixConfig objects pointing to local paths.
    """
    if not is_remote(config.destination_prefix):
        return config.source_prefixes

    (local_input_tempdir := config.tempdir / "input").mkdir(parents=True, exist_ok=True)

    return [
        source_prefix.download(local_input_tempdir / f"{i:06d}")
        for i, source_prefix in enumerate(config.source_prefixes)
    ]


def make_local_output_dir(config: ReshardingConfig) -> Path:
    """
    Create and return a local output directory for resharded files.

    If the destination prefix is remote (e.g., S3), creates a temporary local
    output directory. If the destination is already local, uses the destination
    prefix directly as the output directory.

    Args:
        config: The resharding configuration containing destination prefix and tempdir info.

    Returns:
        A Path object pointing to the local output directory.
    """
    if is_remote(config.destination_prefix):
        local_output_dir = config.tempdir / "output"
    else:
        local_output_dir = Path(config.destination_prefix)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    return local_output_dir


def clean_tempdir(config: ReshardingConfig):
    """
    Clean up the temporary directory after resharding.

    Deletes all files and subdirectories in the temporary directory.

    Args:
        config: The resharding configuration containing the temporary directory.
    """
    for path in config.tempdir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def reshard(config: ReshardingConfig):
    try:
        local_source_prefixes = make_local_source_prefixes(config)
        local_output_dir = make_local_output_dir(config)

        # get repetition aware samples
        tokens_source_paths = [path for source_prefix in local_source_prefixes for path in source_prefix.take()]

        # make destination directory
        local_output_dir.mkdir(parents=True, exist_ok=True)

        # merge the files
        merge_all_npys(
            tokens_source_paths,
            destination=local_output_dir,
            random_seed=config.random_seed,
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
        clean_tempdir(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the resharding config file")
    args = parser.parse_args()

    config = ReshardingConfig.from_file(args.config)
    reshard(config)


if __name__ == "__main__":
    main()
