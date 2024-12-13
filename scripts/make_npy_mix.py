#!/usr/bin/env python3

import argparse
import fnmatch
import logging
import math
import random
import sys
from dataclasses import dataclass
from typing import Callable, Generator
from urllib.parse import urlparse

import boto3
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


units_map = {
    "B": 9,
    "T": 12,
    "G": 9,
    "M": 6,
    "K": 3,
}


@dataclass(frozen=True)
class SourceConfig:
    source: str
    mix_percent: float | None = None
    sample_percent: float = 1.0

    def __post_init__(self):
        if self.mix_percent is not None and (self.mix_percent < 0 or self.mix_percent > 1):
            raise ValueError("mix_percent must be between 0 and 1")
        elif self.sample_percent is not None and (self.sample_percent < 0 or self.sample_percent > 1):
            raise ValueError("sample_percent must be between 0 and 1")

    @property
    def bucket(self) -> str:
        return urlparse(self.source).netloc

    @property
    def prefix(self) -> str:
        path = urlparse(self.source).path.lstrip("/")
        for i, char in enumerate(path):
            if char in ["*", "?", "["]:
                return path[:i]
        return path

    def sample(self, total_size: int) -> tuple[list[str], int]:
        formatter = make_formatter(total_size)

        try:
            all_paths, all_sizes = map(list, zip(*self.glob))
        except ValueError:
            raise ValueError(f"No files found for source {self.source}")

        source_size = sum(all_sizes)

        if self.mix_percent is not None:
            target_size = int(round(total_size * self.mix_percent))
        else:
            target_size = int(round(source_size * self.sample_percent))

        logger.info(
            f"Sampling {formatter(target_size)} bytes from {formatter(source_size)} "
            f"from {self.source} ({target_size / total_size:.2%})"
        )

        # Randomly sample files
        running_size = 0
        selected = []
        while len(all_paths) > 0:
            idx = random.randint(0, len(all_paths) - 1)
            path = all_paths.pop(idx)
            size = all_sizes.pop(idx)
            selected.append(path)

            running_size += size
            if running_size >= target_size:
                break

        return selected, running_size

    @property
    def glob(self) -> Generator[tuple[str, int], None, None]:
        client = boto3.client("s3")

        # Use paginator to handle cases with many objects
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                path = f"s3://{self.bucket}/{obj['Key']}"
                if not path.endswith(".npy"):
                    continue

                # Use fnmatch to check if the object key matches the pattern
                if "*" not in self.source or fnmatch.fnmatch(path, self.source):
                    yield path, obj["Size"]

    @classmethod
    def from_dict(cls, data: dict) -> "SourceConfig":
        return cls(
            source=data["source"],
            mix_percent=data.get("mix_percent"),
            sample_percent=data.get("sample_percent") or 1.0,
        )


@dataclass(frozen=True)
class SamplingConfig:
    target_size: float | int | str
    sources: list[SourceConfig]
    output: str | None = None
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.target_size, str):
            # check if string is in the format "xxxS" where S is a suffix for size (e.g. G, M, K)
            try:
                self.size
            except ValueError as e:
                raise ValueError("Invalid target size format") from e

        if len(self.sources) == 0:
            raise ValueError("Must specify at least one source")

        random.seed(self.seed)

    @property
    def size(self) -> int:
        if isinstance(self.target_size, float) or isinstance(self.target_size, int):
            return int(self.target_size)

        suffix = self.target_size[-1].upper()
        try:
            size = float(self.target_size[:-1])
        except ValueError:
            raise ValueError("Invalid target size format")

        digits = units_map.get(suffix)
        if digits is None:
            raise ValueError("Invalid target size suffix")
        return int(size * 10 ** digits)

    @classmethod
    def from_yaml(cls, path: str) -> "SamplingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "SamplingConfig":
        return cls(
            target_size=data["target_size"],
            sources=[SourceConfig.from_dict(source) for source in data.get("sources", [])],
            output=data.get("output"),
        )


def make_formatter(total_size: int) -> Callable[[int], str]:
    num_digits = (math.floor(math.log10(total_size))) // 3 * 3
    suffix = {v: k for k, v in units_map.items()}.get(num_digits, f"e{num_digits}")

    def formatter(size: int, _num_digits: int = num_digits, _suffix: str = suffix) -> str:
        value = size / 10 ** _num_digits
        return f"{value:.1f}{_suffix}"

    return formatter


def main():
    parser = argparse.ArgumentParser(description="Sample files from S3 datasets")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    if args.config == "-":
        config = SamplingConfig.from_dict(yaml.safe_load(sys.stdin))
    else:
        config = SamplingConfig.from_yaml(args.config)

    formatter = make_formatter(config.size)

    total = 0
    rows = ["data:", "  paths:"]
    for source in config.sources:
        paths, size = source.sample(config.size)
        logger.info(
            f"Selected {len(paths)} files from {source.source} "
            f"({formatter(size)} bytes; {size / config.size:.2%})"
        )
        rows.append(f"\n    # {source.source} ({formatter(size)};{size / config.size:.2%})")
        rows.extend([f"    - {path}" for path in sorted(paths)])
        total += size

    logger.info(f"Total size: {formatter(config.size)} bytes requested, "
                f"{formatter(total)} bytes selected ({total / config.size:.2%})")

    output_text = "\n".join(rows)
    if config.output:
        with open(config.output, "w") as f:
            f.write(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
