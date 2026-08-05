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
import multiprocessing
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import urlparse

import boto3
import numpy as np
import smart_open
import yaml
from tqdm import tqdm

from dolma.core.loggers import get_logger
from dolma.tokenizer.document_selection import (
    DOCUMENT_SELECTION_ALGORITHM,
    create_document_selection,
)
from dolma.tokenizer.tokenizer import Tokenizer

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
RESHARDING_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TokensMetadataPaths:
    npy_path: str
    csv_path: str
    selection_path: str | None = None
    selected_uint32_values: int | None = None

    def __post_init__(self):
        assert self.npy_path.endswith(".npy")
        assert self.csv_path.endswith(".csv.gz")
        assert Path(self.npy_path).stem == Path(Path(self.csv_path).stem).stem
        if self.selection_path is None:
            assert self.selected_uint32_values is None
        else:
            assert self.selection_path.endswith(".csv.gz")
            assert self.selected_uint32_values is not None
            assert self.selected_uint32_values > 0

    @property
    def size(self) -> int:
        if self.selected_uint32_values is not None:
            return self.selected_uint32_values * np.dtype(np.uint32).itemsize
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
    if any(path.selection_path is not None for path in paths) and dtype.itemsize != 4:
        raise ValueError("Document selections require uint32 token memmaps")

    npy_destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(npy_destination) or os.path.lexists(csv_destination):
        raise FileExistsError(f"Refusing to replace an existing reshard output: {npy_destination}")

    target_memmap = np.memmap(npy_destination, mode="w+", shape=(total_size // dtype.itemsize,), dtype=dtype)

    token_offset = 0
    with smart_open.open(csv_destination, "w", encoding="utf-8") as f:
        for path in paths:
            rw = csv.writer(f)
            source_memmap = np.memmap(
                path.npy_path,
                mode="r",
                dtype=dtype,
                shape=(os.path.getsize(path.npy_path) // dtype.itemsize,),
            )
            metadata_path = path.selection_path or path.csv_path
            if path.selection_path is None:
                target_memmap[token_offset : token_offset + source_memmap.shape[0]] = source_memmap
                copy_start = token_offset
                token_offset += source_memmap.shape[0]
            else:
                copy_start = None
            with smart_open.open(metadata_path, "r", encoding="utf-8") as g:
                rd = csv.reader(g)
                for row in rd:
                    start, end, id_, src, idx = row
                    start_value = int(start)
                    end_value = int(end)
                    if path.selection_path is None:
                        assert copy_start is not None
                        output_start = copy_start + start_value
                        output_end = copy_start + end_value
                    else:
                        output_start = token_offset
                        output_end = token_offset + end_value - start_value
                        target_memmap[output_start:output_end] = source_memmap[start_value:end_value]
                        token_offset = output_end
                    rw.writerow(
                        [
                            output_start,
                            output_end,
                            id_,
                            src,
                            int(idx),
                        ]
                    )
            del source_memmap
        target_memmap.flush()
    if token_offset != total_size // dtype.itemsize:
        raise RuntimeError(
            f"Reshard output length mismatch for {npy_destination}: "
            f"wrote {token_offset}, expected {total_size // dtype.itemsize}"
        )


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


def _get_worker_rank() -> int:
    """
    Returns an index for the current worker:
    - 0 if running in a plain session (no threads/processes)
    - 1..N for threads in multithreading
    - 1..N for processes in multiprocessing
    """
    # Case 1: Detect multiprocessing worker
    # multiprocessing sets a special process name like "Process-1", "SpawnPoolWorker-1", etc.
    pname = multiprocessing.current_process().name
    if pname != "MainProcess":
        # Try to extract a trailing integer
        try:
            return int(pname.split("-")[-1])
        except ValueError:
            return 1  # fallback if no number is found

    # Case 2: Detect multithreading worker
    tname = threading.current_thread().name
    if tname != "MainThread":
        # Threads are usually named like "Thread-1", "Thread-2", etc.
        try:
            return int(tname.split("-")[-1])
        except ValueError:
            return 1  # fallback if no number is found

    # Case 3: No multiprocessing or threading
    return 0


def _worker_init(seed: int):
    """Initialize the random seed for the current worker."""
    random.seed(seed + _get_worker_rank())


def merge_all_npys(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    max_size_bytes: int | None = None,
    max_num_files: int | None = None,
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer",
    max_workers: int | None = None,
    seed: int = 42,
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
        "Organizing %s files into %s groups using %s workers...",
        len(paths),
        len(grouped_paths),
        max_workers,
    )

    init_fn = partial(_worker_init, seed=seed)

    with ThreadPoolExecutor(max_workers=max_workers, initializer=init_fn) as pool:
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
        cmd = [
            "s5cmd",
            "cp",
            "-sp",
            "--no-clobber",
            f"{remote_prefix_no_star}/*",
            f"{local_prefix_no_trailing_slash}/",
        ]

        logger.info("Running command: %s", " ".join(cmd))
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"s5cmd failed with error: {result.stderr}")
            raise Exception(f"Failed to download files using s5cmd: {result.stderr}")
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
        logger.info(
            "Taking %s paths from %s using %s sample rate",
            len(new_paths),
            len(paths),
            self.sample_rate,
        )
        return new_paths

    def to_dict(self) -> dict:
        return {"prefix": str(self.prefix), "sample_rate": self.sample_rate}

    @classmethod
    def from_dict(cls, d: dict) -> "ReshardingPrefixConfig":
        return cls(**d)


@dataclass(frozen=True)
class ReshardingManifestEntry:
    npy_uri: str
    metadata_uri: str
    repeat_count: int
    partial_target_uint32_values: int = 0
    selection_seed: int = 0
    selection_algorithm: str = DOCUMENT_SELECTION_ALGORITHM
    npy_size_bytes: int | None = None
    metadata_size_bytes: int | None = None
    npy_etag: str = ""
    metadata_etag: str = ""


@dataclass(frozen=True)
class ReshardingManifestConfig:
    """An exact, locally stored manifest of token/metadata object pairs.

    Version-one CSVs contain ``npy_uri``, ``metadata_uri``, and
    ``repeat_count``. Version two may also request a deterministic partial copy
    with ``partial_target_uint32_values`` and ``selection_seed``. Remote
    objects are downloaded exactly once into a run-owned directory.
    """

    manifest: str | Path

    def take(self, local_prefix: str | Path, max_workers: int) -> list[TokensMetadataPaths]:
        manifest = Path(self.manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Resharding manifest does not exist: {manifest}")

        local_prefix = Path(local_prefix)
        local_prefix.mkdir(parents=True, exist_ok=False)
        rows: list[ReshardingManifestEntry] = []
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            expected = {"npy_uri", "metadata_uri", "repeat_count"}
            if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                raise ValueError(f"Manifest {manifest} must contain columns {sorted(expected)}")
            for row_number, csv_row in enumerate(reader, start=2):
                npy_uri = csv_row["npy_uri"].strip()
                metadata_uri = csv_row["metadata_uri"].strip()
                try:
                    repeat_count = int(csv_row["repeat_count"])
                    partial_target = int(csv_row.get("partial_target_uint32_values") or 0)
                    selection_seed = int(csv_row.get("selection_seed") or 0)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid sampling values on {manifest}:{row_number}") from exc
                if not npy_uri.endswith(".npy") or not metadata_uri.endswith(".csv.gz"):
                    raise ValueError(f"Invalid object pair on {manifest}:{row_number}")
                if any(ord(char) < 32 for char in npy_uri + metadata_uri):
                    raise ValueError(f"Control character in object URI on {manifest}:{row_number}")
                if Path(npy_uri).stem != Path(Path(metadata_uri).stem).stem:
                    raise ValueError(f"Mismatched object pair on {manifest}:{row_number}")
                if repeat_count < 0 or partial_target < 0:
                    raise ValueError(f"Sampling values cannot be negative on {manifest}:{row_number}")
                if repeat_count == 0 and partial_target == 0:
                    raise ValueError(f"Manifest row selects no tokens on {manifest}:{row_number}")
                try:
                    npy_size_bytes = int(csv_row["npy_size_bytes"]) if csv_row.get("npy_size_bytes") else None
                    metadata_size_bytes = (
                        int(csv_row["metadata_size_bytes"]) if csv_row.get("metadata_size_bytes") else None
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid expected size on {manifest}:{row_number}") from exc
                if npy_size_bytes is not None and npy_size_bytes <= 0:
                    raise ValueError(f"npy_size_bytes must be positive on {manifest}:{row_number}")
                if metadata_size_bytes is not None and metadata_size_bytes <= 0:
                    raise ValueError(f"metadata_size_bytes must be positive on {manifest}:{row_number}")
                if urlparse(npy_uri).scheme == "s3" and (npy_size_bytes is None or metadata_size_bytes is None):
                    raise ValueError(
                        f"Remote manifest rows require npy_size_bytes and metadata_size_bytes on "
                        f"{manifest}:{row_number}"
                    )
                if partial_target:
                    if npy_size_bytes is None:
                        raise ValueError(
                            "Partial manifest rows require npy_size_bytes on " f"{manifest}:{row_number}"
                        )
                    if npy_size_bytes % np.dtype(np.uint32).itemsize:
                        raise ValueError(
                            f"Partial source size is not uint32-aligned on " f"{manifest}:{row_number}"
                        )
                    source_values = npy_size_bytes // np.dtype(np.uint32).itemsize
                    if partial_target >= source_values:
                        raise ValueError(
                            "partial_target_uint32_values must be smaller than "
                            f"the source on {manifest}:{row_number}"
                        )
                    selection_algorithm = str(csv_row.get("selection_algorithm") or DOCUMENT_SELECTION_ALGORITHM)
                    if selection_algorithm != DOCUMENT_SELECTION_ALGORITHM:
                        raise ValueError(
                            f"Unsupported selection_algorithm on "
                            f"{manifest}:{row_number}: {selection_algorithm}"
                        )
                else:
                    selection_algorithm = DOCUMENT_SELECTION_ALGORITHM
                rows.append(
                    ReshardingManifestEntry(
                        npy_uri=npy_uri,
                        metadata_uri=metadata_uri,
                        repeat_count=repeat_count,
                        partial_target_uint32_values=partial_target,
                        selection_seed=selection_seed,
                        selection_algorithm=selection_algorithm,
                        npy_size_bytes=npy_size_bytes,
                        metadata_size_bytes=metadata_size_bytes,
                        npy_etag=str(csv_row.get("npy_etag", "")).strip('"'),
                        metadata_etag=str(csv_row.get("metadata_etag", "")).strip('"'),
                    )
                )

        if not rows:
            raise ValueError(f"Resharding manifest is empty: {manifest}")

        remote_expectations: list[tuple[str, int, str]] = []
        for entry in rows:
            if urlparse(entry.npy_uri).scheme == "s3":
                assert entry.npy_size_bytes is not None
                assert entry.metadata_size_bytes is not None
                remote_expectations.extend(
                    [
                        (entry.npy_uri, entry.npy_size_bytes, entry.npy_etag),
                        (
                            entry.metadata_uri,
                            entry.metadata_size_bytes,
                            entry.metadata_etag,
                        ),
                    ]
                )
        if remote_expectations:
            client = boto3.client("s3")

            def verify_remote(expectation: tuple[str, int, str]) -> None:
                uri, expected_size, expected_etag = expectation
                parsed = urlparse(uri)
                response = client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
                actual_size = int(response["ContentLength"])
                actual_etag = str(response.get("ETag", "")).strip('"')
                if actual_size != expected_size:
                    raise RuntimeError(
                        f"Source object size changed before download: {uri}; "
                        f"expected {expected_size}, found {actual_size}"
                    )
                if expected_etag and actual_etag != expected_etag:
                    raise RuntimeError(
                        f"Source object ETag changed before download: {uri}; "
                        f"expected {expected_etag}, found {actual_etag}"
                    )

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(verify_remote, expectation) for expectation in remote_expectations]
                for future in as_completed(futures):
                    future.result()

        paths: list[TokensMetadataPaths] = []
        downloaded_pairs: list[
            tuple[
                ReshardingManifestEntry,
                Path,
                Path,
                Path,
                int | None,
                int | None,
            ]
        ] = []
        remote_commands: list[str] = []
        for index, entry in enumerate(rows):
            npy_uri = entry.npy_uri
            metadata_uri = entry.metadata_uri
            row_dir = local_prefix / f"{index:06d}"
            row_dir.mkdir(exist_ok=False)
            local_npy = row_dir / "tokens.npy"
            local_metadata = row_dir / "tokens.csv.gz"

            npy_scheme = urlparse(npy_uri).scheme
            metadata_scheme = urlparse(metadata_uri).scheme
            if npy_scheme == "s3" and metadata_scheme == "s3":
                remote_commands.extend(
                    [
                        f"cp --raw --no-clobber {shlex.quote(npy_uri)} {shlex.quote(str(local_npy))}",
                        f"cp --raw --no-clobber {shlex.quote(metadata_uri)} {shlex.quote(str(local_metadata))}",
                    ]
                )
            elif npy_scheme in {"", "file"} and metadata_scheme in {"", "file"}:
                local_npy = Path(urlparse(npy_uri).path if npy_scheme == "file" else npy_uri)
                local_metadata = Path(urlparse(metadata_uri).path if metadata_scheme == "file" else metadata_uri)
                if not local_npy.is_file() or not local_metadata.is_file():
                    raise FileNotFoundError(f"Manifest object pair does not exist: {npy_uri}, {metadata_uri}")
            else:
                raise ValueError(f"Manifest row mixes unsupported URI schemes: {npy_uri}, {metadata_uri}")

            downloaded_pairs.append(
                (
                    entry,
                    local_npy,
                    local_metadata,
                    row_dir / "selection.csv.gz",
                    entry.npy_size_bytes,
                    entry.metadata_size_bytes,
                )
            )

        if remote_commands:
            commands_path = local_prefix / "s5cmd-download-commands.txt"
            with commands_path.open("x") as f:
                f.write("\n".join(remote_commands) + "\n")
            cmd = ["s5cmd", "--numworkers", str(max_workers), "run", str(commands_path)]
            logger.info("Downloading exact manifest objects with s5cmd")
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"s5cmd manifest download failed: {result.stderr}")

            missing = [
                (npy_path, metadata_path)
                for _, npy_path, metadata_path, _, _, _ in downloaded_pairs
                if not npy_path.is_file() or not metadata_path.is_file()
            ]
            if missing:
                raise RuntimeError(f"s5cmd completed without creating {len(missing)} manifest object pairs")

        for (
            entry,
            npy_path,
            metadata_path,
            selection_path,
            expected_npy_size,
            expected_metadata_size,
        ) in downloaded_pairs:
            if expected_npy_size is not None and npy_path.stat().st_size != expected_npy_size:
                raise RuntimeError(
                    f"Downloaded NPY size does not match manifest: {npy_path}; "
                    f"expected {expected_npy_size}, found {npy_path.stat().st_size}"
                )
            if expected_metadata_size is not None and metadata_path.stat().st_size != expected_metadata_size:
                raise RuntimeError(
                    f"Downloaded metadata size does not match manifest: {metadata_path}; "
                    f"expected {expected_metadata_size}, found {metadata_path.stat().st_size}"
                )

            full_path = TokensMetadataPaths(str(npy_path), str(metadata_path))
            paths.extend([full_path] * entry.repeat_count)
            if entry.partial_target_uint32_values:
                source_values = npy_path.stat().st_size // np.dtype(np.uint32).itemsize
                selection = create_document_selection(
                    metadata_path=metadata_path,
                    selection_path=selection_path,
                    source_uint32_values=source_values,
                    target_uint32_values=entry.partial_target_uint32_values,
                    seed=entry.selection_seed,
                )
                logger.info(
                    "Selected %s uint32 values from %s for a %s-value quota " "(residual %+d; %s documents)",
                    selection.selected_uint32_values,
                    entry.npy_uri,
                    entry.partial_target_uint32_values,
                    selection.target_residual_uint32_values,
                    selection.selected_document_count,
                )
                if selection.selected_uint32_values:
                    paths.append(
                        TokensMetadataPaths(
                            str(npy_path),
                            str(metadata_path),
                            selection_path=str(selection.selection_path),
                            selected_uint32_values=selection.selected_uint32_values,
                        )
                    )

        if not paths:
            raise RuntimeError(f"Manifest document selections produced no output: {manifest}")
        return paths

    def to_dict(self) -> dict:
        return {"manifest": str(self.manifest)}

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path | None = None) -> "ReshardingManifestConfig":
        path = Path(d["manifest"])
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return cls(manifest=path)


@dataclass
class ReshardingConfig:
    """Base configuration for resharding."""

    destination_prefix: str
    source_prefixes: list[ReshardingPrefixConfig] = field(default_factory=list)
    source_manifests: list[ReshardingManifestConfig] = field(default_factory=list)
    local_tempdir: str | Path | None = None
    max_size_bytes: int | None = None
    max_num_files: int | None = None
    max_workers: int = os.cpu_count() or 1
    random_seed: int = 42
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer"
    allow_existing_destination: bool = False

    def __post_init__(self):
        if self.max_size_bytes is not None and self.max_num_files is not None:
            raise ValueError("Cannot provide both max_size_bytes and max_num_files")
        if self.max_size_bytes is None and self.max_num_files is None:
            raise ValueError("Either max_size_bytes or max_num_files must be provided")
        if not self.source_prefixes and not self.source_manifests:
            raise ValueError("At least one source_prefix or source_manifest must be provided")
        if self.allow_existing_destination:
            raise ValueError("Overwriting an existing destination is not supported")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

        if self.local_tempdir is None:
            logging.warning("No local tempdir provided; using a temporary directory")
        else:
            self.local_tempdir = Path(self.local_tempdir)

    def to_dict(self) -> dict:
        source_prefixes_dict = [p.to_dict() for p in self.source_prefixes]
        source_manifests_dict = [m.to_dict() for m in self.source_manifests]
        return {
            **asdict(self),
            "source_prefixes": source_prefixes_dict,
            "source_manifests": source_manifests_dict,
        }

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path | None = None) -> "ReshardingConfig":
        source_prefixes = [ReshardingPrefixConfig.from_dict(p) for p in d.get("source_prefixes", [])]
        source_manifests = [
            ReshardingManifestConfig.from_dict(m, base_dir=base_dir) for m in d.get("source_manifests", [])
        ]
        return cls(
            destination_prefix=str(d["destination_prefix"]),
            source_prefixes=source_prefixes,
            source_manifests=source_manifests,
            local_tempdir=(Path(p) if (p := d.get("local_tempdir")) is not None else None),
            max_size_bytes=(int(s) if (s := d.get("max_size_bytes")) is not None else None),
            max_num_files=(int(n) if (n := d.get("max_num_files")) is not None else None),
            max_workers=int(d.get("max_workers", 1)),
            random_seed=int(d.get("random_seed", 42)),
            tokenizer_name_or_path=str(d.get("tokenizer_name_or_path", "allenai/dolma2-tokenizer")),
            allow_existing_destination=bool(d.get("allow_existing_destination", False)),
        )

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ReshardingConfig":
        if file_path == "-":
            return cls.from_dict(yaml.safe_load(sys.stdin))

        path = Path(file_path)
        with path.open("r") as f:
            return cls.from_dict(yaml.safe_load(f), base_dir=path.parent)


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
        "--numworkers",
        str(max_workers),
        "cp",
        "--no-clobber",
        "-sp",
        f"{local_prefix_no_star}/*",
        f"{remote_prefix_no_trailing_slash}/",
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"s5cmd failed with error: {result.stderr}")
        raise Exception(f"Failed to upload files using s5cmd: {result.stderr}")


def destination_has_objects(destination: str | Path) -> bool:
    """Return whether a destination already contains data.

    S3 is queried with a read-only, one-key ``ListObjectsV2`` request. Local
    destinations are considered occupied when the path exists at all; this
    deliberately refuses even an empty pre-created directory.
    """

    destination = str(destination)
    parsed = urlparse(destination)
    if parsed.scheme == "s3":
        prefix = parsed.path.lstrip("/").rstrip("/")
        if not parsed.netloc or not prefix:
            raise ValueError("Refusing to materialize into an S3 bucket root")
        response = boto3.client("s3").list_objects_v2(Bucket=parsed.netloc, Prefix=f"{prefix}/", MaxKeys=1)
        return bool(response.get("Contents"))
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"Unsupported destination protocol: {parsed.scheme}")
    path = Path(parsed.path if parsed.scheme == "file" else destination)
    return os.path.lexists(path)


def reshard(config: ReshardingConfig):
    random.seed(config.random_seed)

    if destination_has_objects(config.destination_prefix):
        raise FileExistsError(f"Refusing to use existing destination: {config.destination_prefix}")

    run_tempdir: Path | None = None
    try:
        if config.local_tempdir is None:
            run_tempdir = Path(mkdtemp(prefix="dolma-reshard-"))
        else:
            temp_base = Path(config.local_tempdir)
            temp_base.mkdir(parents=True, exist_ok=True)
            run_tempdir = Path(mkdtemp(prefix="dolma-reshard-", dir=temp_base))

        local_output_dir = (
            run_tempdir / "output"
            if urlparse(config.destination_prefix).scheme == "s3"
            else Path(config.destination_prefix)
        )

        # download the files
        source_prefixes = [
            source_prefix.download(run_tempdir / f"prefix-input/{i:06d}")
            for i, source_prefix in enumerate(config.source_prefixes)
        ]

        # get repetition aware samples
        source_paths = [path for source_prefix in source_prefixes for path in source_prefix.take()]
        for i, source_manifest in enumerate(config.source_manifests):
            source_paths.extend(
                source_manifest.take(
                    run_tempdir / f"manifest-input/{i:06d}",
                    max_workers=config.max_workers,
                )
            )

        # make destination directory
        local_output_dir.mkdir(parents=True, exist_ok=False)

        # merge the files
        merge_all_npys(
            source_paths,
            destination=local_output_dir,
            max_size_bytes=config.max_size_bytes,
            max_num_files=config.max_num_files,
            max_workers=config.max_workers,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            seed=config.random_seed,
        )

        # upload the files
        upload_to_s3(
            local_prefix=local_output_dir,
            remote_prefix=config.destination_prefix,
            max_workers=config.max_workers,
        )

    finally:
        if run_tempdir is not None:
            shutil.rmtree(run_tempdir)


def main():
    config = ReshardingConfig.from_file(sys.argv[1])
    reshard(config)


if __name__ == "__main__":
    main()
