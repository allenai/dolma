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

    target_memmap = np.memmap(
        npy_destination, mode="w+", shape=(total_size // dtype.itemsize,), dtype=dtype
    )

    bytes_offset = row_offset = 0
    with smart_open.open(csv_destination, "w", encoding="utf-8") as f:
        for path in paths:
            rw = csv.writer(f)
            source_memmap = np.memmap(
                path.npy_path,
                mode="r",
                dtype=dtype,
                shape=(path.size // dtype.itemsize,),
            )
            target_memmap[bytes_offset : bytes_offset + source_memmap.shape[0]] = (
                source_memmap
            )
            target_memmap.flush()

            row_count = 0
            with smart_open.open(path.csv_path, "r", encoding="utf-8") as g:
                rd = csv.reader(g)
                for row in rd:
                    start, end, id_, src, idx = row
                    rw.writerow(
                        [
                            int(start) + bytes_offset,
                            int(end) + bytes_offset,
                            id_,
                            src,
                            int(idx),
                        ]
                    )
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
    counts: dict[TokensMetadataPaths, int] = {
        p: int(c) for p, c in Counter(paths).items()
    }
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
        counts = {
            path: new_count
            for path, count in counts.items()
            if (new_count := count - 1) > 0
        }

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
        raise ValueError(
            f"One or more paths appear {m} times, exceeding max_num_files={max_num_files}"
        )

    grouped_paths: list[list[TokensMetadataPaths]] = [[] for _ in range(max_num_files)]
    # Distribute each element across groups in round-robin fashion
    for element, count in counts.items():
        # sample count buckets out of max_num_files where we could put the element
        # we sample with weights proportional to the number of elements in the bucket,
        # so that we are more likely to sample buckets with fewer elements.
        weights = [1 / (len(grouped_paths[i]) + 1) for i in range(max_num_files)]
        buckets = weighted_bucket_sample(
            values=list(range(max_num_files)),
            count=count,
            weights=weights,
        )
        logger.info(
            "Distributing element %s (count=%d) to buckets %s",
            element,
            count,
            buckets,
        )
        for bucket in buckets:
            grouped_paths[bucket].append(element)

    # there is still a change that some buckets might be empty; we remove them.
    grouped_paths = [group for group in grouped_paths if len(group) > 0]

    logger.info(
        "Organized %s files into %s groups",
        len(paths),
        len(grouped_paths),
    )
    return grouped_paths


def merge_all_npys(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    max_size_bytes: int | None = None,
    max_num_files: int | None = None,
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer",
    max_workers: int | None = None,
):
    max_workers = max_workers or os.cpu_count() or 192

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
        "Organizing %s files into %s groups using %s workers...",
        len(paths),
        len(grouped_paths),
        max_workers,
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

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Merging files"
        ):
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
    Supports both a single prefix and a list of path globs.
    """

    prefix: str | Path | None = None  # Single prefix (for backward compatibility)
    paths: list[str | Path] | None = None  # List of path globs
    sample_rate: float | None = None
    target_ratio: float | None = None
    _computed_sample_rate: float | None = None

    def __post_init__(self):
        # Validate that either prefix or paths is provided
        if self.prefix is None and self.paths is None:
            raise ValueError("Must specify either 'prefix' or 'paths'")
        if self.prefix is not None and self.paths is not None:
            raise ValueError("Cannot specify both 'prefix' and 'paths'")

        # Convert single prefix to paths list for consistency
        if self.prefix is not None:
            self.paths = [self.prefix]

        if self.sample_rate is not None and self.target_ratio is not None:
            raise ValueError("Cannot specify both sample_rate and target_ratio")
        if self.sample_rate is None and self.target_ratio is None:
            raise ValueError("Must specify either sample_rate or target_ratio")
        if self.sample_rate is not None:
            assert self.sample_rate > 0
            self._computed_sample_rate = self.sample_rate
        if self.target_ratio is not None:
            assert 0 < self.target_ratio <= 1.0

    def get_sample_rate(
        self, total_source_sizes: dict[str, int] | None = None
    ) -> float:
        """Get the computed sample rate, calculating from target_ratio if needed."""
        if self._computed_sample_rate is not None:
            return self._computed_sample_rate

        if self.target_ratio is not None and total_source_sizes is not None:
            # Calculate sample_rate from target_ratio
            # sample_rate = (target_ratio * total_mix_size) / source_size
            # Sum up sizes for all paths in this source
            source_size = 0
            for path in self.paths or []:
                source_size += total_source_sizes.get(str(path), 0)

            total_mix_size = sum(total_source_sizes.values())
            if source_size > 0:
                self._computed_sample_rate = (
                    self.target_ratio * total_mix_size
                ) / source_size
            else:
                raise ValueError(f"Source size for paths {self.paths} is 0")
            return self._computed_sample_rate

        if self.sample_rate is not None:
            return self.sample_rate

        raise ValueError(
            "Cannot compute sample rate without total_source_sizes when using target_ratio"
        )

    def download(
        self, local_prefix: str | Path, skip_download: bool = False
    ) -> "ReshardingPrefixConfig":
        """Download files from remote paths to local directory."""
        if not self.paths:
            return self

        if skip_download:
            return self

        # Disallow GCS; check if any paths are remote (s3)
        schemes = {urlparse(str(path)).scheme for path in self.paths}
        if "gs" in schemes:
            raise ValueError(
                "GCS (gs://) paths are not supported. Please pre-download locally or mirror to s3://."
            )
        needs_download = "s3" in schemes

        if not needs_download:
            return self

        local_prefix = Path(local_prefix)
        local_prefix.mkdir(parents=True, exist_ok=True)

        for i, path in enumerate(self.paths):
            scheme = urlparse(str(path)).scheme

            if scheme == "s3":
                # Create a subdirectory for each path to avoid filename collisions
                path_subdir = local_prefix / f"path_{i:04d}"
                path_subdir.mkdir(parents=True, exist_ok=True)

                # Use s5cmd with glob support
                logger.info("Downloading S3 path %s to %s", path, path_subdir)
                cmd = ["s5cmd", "cp", "-u", "--sp", str(path), f"{path_subdir}/"]

                logger.info("Running command: %s", " ".join(cmd))
                result = subprocess.run(
                    cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                if result.returncode != 0:
                    print(f"s5cmd failed with error: {result.stderr}")
                    raise Exception(
                        f"Failed to download files using s5cmd: {result.stderr}"
                    )

            elif scheme in {"", "file"}:
                # Local path; nothing to do
                continue
            else:
                raise ValueError(f"Unsupported protocol: {scheme}")

        # After downloading, the paths are now local under local_prefix
        return ReshardingPrefixConfig(
            paths=[local_prefix],  # All files are now under this local directory
            sample_rate=self.sample_rate,
            target_ratio=self.target_ratio,
            _computed_sample_rate=self._computed_sample_rate,
        )

    def take(
        self, total_source_sizes: dict[str, int] | None = None
    ) -> list[TokensMetadataPaths]:
        """Collect files from all path globs and apply sampling."""
        if not self.paths:
            return []

        # Ensure all paths are local
        for path_pattern in self.paths:
            if urlparse(str(path_pattern)).scheme not in {"file", ""}:
                raise ValueError(
                    f"Invalid protocol: {urlparse(str(path_pattern)).scheme}; "
                    f"only local paths are supported; download the files first."
                )

        paths = []
        for path_pattern in self.paths:
            path_pattern_str = str(path_pattern)

            # Handle glob patterns
            if "*" in path_pattern_str:
                # Use glob to find matching files
                import glob

                matching_files = glob.glob(path_pattern_str, recursive=True)
                for npy_path in matching_files:
                    if npy_path.endswith(".npy"):
                        csv_path = npy_path.replace(".npy", ".csv.gz")
                        if os.path.exists(csv_path):
                            paths.append(TokensMetadataPaths(npy_path, csv_path))
            else:
                # Treat as a directory path and walk it
                local_path = Path(path_pattern)
                if local_path.exists():
                    for root, _, files in os.walk(local_path):
                        for file in files:
                            if file.endswith(".npy"):
                                npy_path = os.path.join(root, file)
                                csv_path = os.path.join(
                                    root, file.replace(".npy", ".csv.gz")
                                )
                                if os.path.exists(csv_path):
                                    paths.append(
                                        TokensMetadataPaths(npy_path, csv_path)
                                    )

        new_paths = []

        # Get the computed sample rate
        sample_rate = self.get_sample_rate(total_source_sizes)

        # if the multiplier k is > 1, we first take ⌊k⌋ copies of each path.
        if (repetition_rate := int(math.floor(sample_rate))) > 0:
            new_paths.extend(paths * repetition_rate)

        # this is the remaining non-integer part of the sample rate; because the npys are actually uneven in
        # size, the proper way to do this is to use an ILP solver; however, since usually most of the npys are
        # of same size, we can just take a random sample.
        if (residual_frac := sample_rate - repetition_rate) > 0:
            sample_size = min(len(paths), max(1, round(residual_frac * len(paths))))
            new_paths.extend(random.sample(paths, sample_size))

        # sort by size
        logger.info(
            "Taking %s paths from %s using %s sample rate",
            len(new_paths),
            len(paths),
            sample_rate,
        )
        return new_paths

    def to_dict(self) -> dict:
        result = {}
        # Use 'paths' if we have multiple paths, 'prefix' for single path (backward compatibility)
        if self.paths and len(self.paths) == 1 and self.prefix is not None:
            result["prefix"] = str(self.prefix)
        elif self.paths:
            result["paths"] = [str(p) for p in self.paths]

        if self.sample_rate is not None:
            result["sample_rate"] = self.sample_rate
        if self.target_ratio is not None:
            result["target_ratio"] = self.target_ratio
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "ReshardingPrefixConfig":
        # Handle both 'prefix' (single) and 'paths' (multiple)
        prefix = d.get("prefix")
        paths = d.get("paths")

        return cls(
            prefix=prefix if prefix is not None else None,
            paths=paths if paths is not None else None,
            sample_rate=d.get("sample_rate"),
            target_ratio=d.get("target_ratio"),
        )


@dataclass
class ReshardingConfig:
    """Base configuration for resharding."""

    source_prefixes: list[ReshardingPrefixConfig]
    destination_prefix: str
    local_tempdir: str | Path | None = None
    max_size_bytes: int | None = None
    max_num_files: int | None = None
    max_workers: int = os.cpu_count() or 192
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
        source_prefixes = [
            ReshardingPrefixConfig.from_dict(p) for p in d.get("source_prefixes", [])
        ]
        return cls(
            source_prefixes=source_prefixes,
            destination_prefix=str(d["destination_prefix"]),
            local_tempdir=(
                Path(p) if (p := d.get("local_tempdir")) is not None else None
            ),
            max_size_bytes=(
                int(s) if (s := d.get("max_size_bytes")) is not None else None
            ),
            max_num_files=(
                int(n) if (n := d.get("max_num_files")) is not None else None
            ),
            max_workers=int(d.get("max_workers", 192)),
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
    cmd = [
        "s5cmd",
        "cp",
        "-sp",
        f"{local_prefix_no_star}/*",
        f"{remote_prefix_no_trailing_slash}/",
    ]
    result = subprocess.run(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f"s5cmd failed with error: {result.stderr}")
        raise Exception(f"Failed to upload files using s5cmd: {result.stderr}")
    print(f"Uploaded {local_prefix} to {remote_prefix}")


def reshard(config: ReshardingConfig):
    random.seed(config.random_seed)

    try:
        local_tempdir = Path(config.local_tempdir or mkdtemp())
        local_tempdir.mkdir(parents=True, exist_ok=True)
        logger.info("Using local temp directory: %s", local_tempdir)

        local_output_dir = (
            local_tempdir / "output"
            if urlparse(config.destination_prefix).scheme == "s3"
            else Path(config.destination_prefix)
        )
        logger.info("Local output directory: %s", local_output_dir)

        # download the files
        logger.info("Processing %d source prefixes", len(config.source_prefixes))
        source_prefixes = []
        for i, source_prefix in enumerate(config.source_prefixes):
            logger.info(
                "Downloading source prefix %d/%d", i + 1, len(config.source_prefixes)
            )
            downloaded_prefix = source_prefix.download(
                local_tempdir / f"input/{i:06d}",
                skip_download=True,  ## !!!
            )
            source_prefixes.append(downloaded_prefix)
        logger.info("Finished downloading all source prefixes")

        # Calculate total source sizes if any prefix uses target_ratio
        total_source_sizes = None
        if any(sp.target_ratio is not None for sp in source_prefixes):
            total_source_sizes = {}
            for source_prefix in source_prefixes:
                # Calculate total size for all paths in this source
                for path_pattern in source_prefix.paths or []:
                    path_pattern_str = str(path_pattern)
                    total_size = 0

                    # Handle glob patterns
                    if "*" in path_pattern_str:
                        import glob

                        matching_files = glob.glob(path_pattern_str, recursive=True)
                        for npy_path in matching_files:
                            if npy_path.endswith(".npy"):
                                total_size += os.path.getsize(npy_path)
                    else:
                        # Treat as a directory path and walk it
                        local_path = Path(path_pattern)
                        if local_path.exists():
                            for root, _, files in os.walk(local_path):
                                for file in files:
                                    if file.endswith(".npy"):
                                        npy_path = os.path.join(root, file)
                                        total_size += os.path.getsize(npy_path)

                    # Store size for this specific path pattern
                    total_source_sizes[str(path_pattern)] = total_size

            # Verify target ratios sum to <= 1.0
            total_ratio = sum(
                sp.target_ratio for sp in source_prefixes if sp.target_ratio is not None
            )
            if total_ratio > 1.0:
                raise ValueError(f"Sum of target_ratios ({total_ratio}) exceeds 1.0")

        # get repetition aware samples
        source_paths = [
            path
            for source_prefix in source_prefixes
            for path in source_prefix.take(total_source_sizes)
        ]

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
