"""Reshard npy token files into larger output shards.

Merges npy files and their csv.gz metadata partners so that each output shard
satisfies a size or file-count constraint. The entry point takes a single YAML
config path, or "-" to read the config from stdin.

A config names one destination and at least one source, and may mix the two source
kinds. source_prefixes discovers every npy under a prefix by listing it.
source_manifests reads an exact CSV manifest of npy/csv.gz pairs, with per-row
repeat_count and deterministic partial selection, so a run can reproduce an exact
subset of tokens rather than whole files. Exactly one of max_size_bytes or
max_num_files must be set; ReshardingConfig holds the remaining fields.

Destinations are never overwritten: reshard() refuses to run when the destination
already contains objects, and token and metadata files are written with no-clobber
semantics.

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
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import urlparse

import boto3
import numpy as np
import smart_open
import yaml

from dolma.core.loggers import get_logger
from dolma.tokenizer.document_selection import (
    DOCUMENT_SELECTION_ALGORITHM,
    DocumentSelectionResult,
    create_document_selection,
)
from dolma.tokenizer.tokenizer import Tokenizer

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Manifests carry no version column. External planners import this constant and
# refuse to dispatch when it does not match the version they generated for. Bump it
# when the columns the parser requires, or their meaning, change.
RESHARDING_MANIFEST_SCHEMA_VERSION = 2
PROGRESS_INTERVAL_SECONDS = 10.0
# Default parallelism for a single s5cmd `cp`, shared by the manifest downloader
# and the config field that feeds it.
DEFAULT_S5CMD_DOWNLOAD_CONCURRENCY = 32
# Rows between progress checks while writing merged metadata.
METADATA_ROW_LOG_INTERVAL = 100_000


def _human_count(value: int, unit: str = "") -> str:
    for scale, suffix in ((10**12, "T"), (10**9, "B"), (10**6, "M"), (10**3, "K")):
        if abs(value) >= scale:
            rendered = f"{value / scale:.3g}{suffix}"
            return f"{rendered} {unit}".rstrip()
    return f"{value:,} {unit}".rstrip()


def _human_bytes(value: int) -> str:
    for scale, suffix in (
        (1024**4, "TiB"),
        (1024**3, "GiB"),
        (1024**2, "MiB"),
        (1024, "KiB"),
    ):
        if value >= scale:
            return f"{value / scale:.3g} {suffix}"
    return f"{value:,} B"


def _elapsed(started_at: float) -> str:
    seconds = max(0, round(time.monotonic() - started_at))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _run_s5cmd(command: list[str], phase: str) -> None:
    """Run s5cmd, forwarding its in-place progress updates to the worker log."""

    logger.info("s5cmd %s: %s", phase, shlex.join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        # TextIOWrapper's universal-newline handling converts the carriage returns
        # used by --show-progress into newlines. Flush each update so the controller
        # can retrieve it while s5cmd is still running.
        print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"s5cmd {phase} failed with exit code {returncode}; inspect the worker log")


@dataclass(frozen=True)
class TokensMetadataPaths:
    npy_path: str
    csv_path: str
    selection_path: str | None = None
    selected_uint32_values: int | None = None

    def __post_init__(self):
        # These checks raise rather than assert because asserts are stripped
        # under `python -O`.
        if not self.npy_path.endswith(".npy"):
            raise ValueError(f"Token path must end in .npy: {self.npy_path}")
        if not self.csv_path.endswith(".csv.gz"):
            raise ValueError(f"Metadata path must end in .csv.gz: {self.csv_path}")
        if Path(self.npy_path).stem != Path(Path(self.csv_path).stem).stem:
            raise ValueError(f"Mismatched token/metadata pair: {self.npy_path}, {self.csv_path}")
        if self.selection_path is None:
            if self.selected_uint32_values is not None:
                raise ValueError("selected_uint32_values requires a selection_path")
        else:
            if not self.selection_path.endswith(".csv.gz"):
                raise ValueError(f"Selection path must end in .csv.gz: {self.selection_path}")
            if self.selected_uint32_values is None:
                raise ValueError("A selection_path requires selected_uint32_values")
            if self.selected_uint32_values <= 0:
                raise ValueError(f"selected_uint32_values must be positive: {self.selected_uint32_values}")

    @property
    def size(self) -> int:
        if self.selected_uint32_values is not None:
            return self.selected_uint32_values * np.dtype(np.uint32).itemsize
        return os.path.getsize(self.npy_path)


@dataclass(frozen=True)
class MergeGroupResult:
    destination: Path
    output_bytes: int
    metadata_bytes: int
    document_count: int
    token_copy_operations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class MergeGroupProgress:
    destination: Path
    processed_values: int
    total_values: int
    document_count: int
    complete: bool = False


def merge_group(
    paths: list[TokensMetadataPaths],
    destination: str | Path,
    dtype: np.dtype,
    progress: Callable[[MergeGroupProgress], None] | None = None,
) -> MergeGroupResult:
    """Merge the given inputs into one npy memmap plus a sibling csv.gz of metadata.

    Document offsets in the merged metadata are rebased onto the output memmap. Inputs
    carrying a selection index contribute only their selected documents and require a
    uint32 dtype. Raises FileExistsError when either output path exists, and
    RuntimeError when the number of values written does not match the total computed
    from the inputs.
    """
    npy_destination = Path(destination)
    csv_destination = npy_destination.with_suffix(".csv.gz")
    total_size = sum(p.size for p in paths)
    total_values = total_size // dtype.itemsize
    if any(path.selection_path is not None for path in paths) and dtype.itemsize != 4:
        raise ValueError("Document selections require uint32 token memmaps")

    npy_destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(npy_destination) or os.path.lexists(csv_destination):
        raise FileExistsError(f"Refusing to replace an existing reshard output: {npy_destination}")

    started_at = time.monotonic()
    logger.info(
        "merge %s start · %s inputs · %s tokens (%s)",
        npy_destination.stem,
        len(paths),
        _human_count(total_values),
        _human_bytes(total_size),
    )
    target_memmap = np.memmap(npy_destination, mode="w+", shape=(total_values,), dtype=dtype)

    token_offset = 0
    completed_values = 0
    document_count = 0
    token_copy_operations = 0
    last_progress_at = started_at

    def publish_progress(*, complete: bool = False) -> None:
        if progress is not None:
            progress(
                MergeGroupProgress(
                    destination=npy_destination,
                    processed_values=completed_values + input_processed_values,
                    total_values=total_values,
                    document_count=document_count,
                    complete=complete,
                )
            )

    input_processed_values = 0
    publish_progress()
    with smart_open.open(csv_destination, "w", encoding="utf-8") as f:
        rw = csv.writer(f)
        for input_index, path in enumerate(paths, start=1):
            input_started_at = time.monotonic()
            input_documents = 0
            input_processed_values = 0
            view_type = "selected" if path.selection_path is not None else "full"
            source_memmap = np.memmap(
                path.npy_path,
                mode="r",
                dtype=dtype,
                shape=(os.path.getsize(path.npy_path) // dtype.itemsize,),
            )
            metadata_path = path.selection_path or path.csv_path
            if path.selection_path is None:
                copy_started_at = time.monotonic()
                target_memmap[token_offset : token_offset + source_memmap.shape[0]] = source_memmap
                copy_seconds = max(time.monotonic() - copy_started_at, 1e-9)
                token_copy_operations += 1
                logger.info(
                    "merge %s copy · input %s/%s full · %s at %s/s",
                    npy_destination.stem,
                    input_index,
                    len(paths),
                    _human_bytes(path.size),
                    _human_bytes(round(path.size / copy_seconds)),
                )
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
                        input_processed_values = end_value
                    else:
                        output_start = token_offset
                        output_end = token_offset + end_value - start_value
                        target_memmap[output_start:output_end] = source_memmap[start_value:end_value]
                        token_offset = output_end
                        copied_values = end_value - start_value
                        input_processed_values += copied_values
                        token_copy_operations += 1
                    input_documents += 1
                    document_count += 1
                    rw.writerow(
                        [
                            output_start,
                            output_end,
                            id_,
                            src,
                            int(idx),
                        ]
                    )
                    if document_count % METADATA_ROW_LOG_INTERVAL == 0:
                        now = time.monotonic()
                        if now - last_progress_at >= PROGRESS_INTERVAL_SECONDS:
                            publish_progress()
                            last_progress_at = now
            del source_memmap
            input_seconds = max(time.monotonic() - input_started_at, 1e-9)
            finished_input_values = input_processed_values
            completed_values += finished_input_values
            input_processed_values = 0
            publish_progress()
            logger.info(
                "merge %s input %s/%s done · %s · %s tokens · %s docs · " "%s tokens/s · %s docs/s · %s",
                npy_destination.stem,
                input_index,
                len(paths),
                view_type,
                _human_count(finished_input_values),
                _human_count(input_documents),
                _human_count(round(finished_input_values / input_seconds)),
                _human_count(round(input_documents / input_seconds)),
                _elapsed(input_started_at),
            )
        flush_started_at = time.monotonic()
        target_memmap.flush()
        logger.info(
            "merge %s flush · %s · %s",
            npy_destination.stem,
            _human_bytes(total_size),
            _elapsed(flush_started_at),
        )
    if token_offset != total_values:
        raise RuntimeError(
            f"Reshard output length mismatch for {npy_destination}: "
            f"wrote {token_offset}, expected {total_values}"
        )
    elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
    metadata_bytes = csv_destination.stat().st_size
    result = MergeGroupResult(
        destination=npy_destination,
        output_bytes=total_size,
        metadata_bytes=metadata_bytes,
        document_count=document_count,
        token_copy_operations=token_copy_operations,
        elapsed_seconds=elapsed_seconds,
    )
    publish_progress(complete=True)
    logger.info(
        "merge %s done · %s tokens (%s) · %s metadata · %s docs · " "%s tokens/s · %s docs/s · %s",
        npy_destination.stem,
        _human_count(total_values),
        _human_bytes(total_size),
        _human_bytes(metadata_bytes),
        _human_count(document_count),
        _human_count(round(total_values / elapsed_seconds)),
        _human_count(round(document_count / elapsed_seconds)),
        _elapsed(started_at),
    )
    return result


def group_paths_by_max_size(
    paths: list[TokensMetadataPaths],
    max_size_bytes: int,
) -> list[list[TokensMetadataPaths]]:
    """Group merge inputs into output shards of at most max_size_bytes each.

    Every pass over the remaining inputs starts a new group and emits one copy of each
    input, so repeated copies of an input are spread across groups.
    """
    counts: dict[TokensMetadataPaths, int] = {p: int(c) for p, c in Counter(paths).items()}
    _log_merge_input_views(counts, len(paths))

    grouped_paths: list[list[TokensMetadataPaths]] = []
    while len(counts) > 0:
        grouped_paths.append([])

        for path, _ in sorted(counts.items(), key=lambda x: -x[1]):
            if sum(p.size for p in grouped_paths[-1]) + path.size > max_size_bytes:
                grouped_paths.append([path])
            else:
                grouped_paths[-1].append(path)

        counts = {path: new_count for path, count in counts.items() if (new_count := count - 1) > 0}

    logger.info(
        "Grouped %s merge input uses into %s output shards capped at %.2f GiB",
        len(paths),
        len(grouped_paths),
        max_size_bytes / 1024**3,
    )

    return grouped_paths


def group_paths_by_max_num_files(
    paths: list[TokensMetadataPaths],
    max_num_files: int,
) -> list[list[TokensMetadataPaths]]:
    """Pack merge inputs into at most `max_num_files` size-balanced output shards.

    Largest-first bin packing: every use of every input is expanded into a flat list,
    shuffled so equal-sized views tie-break under the configured deterministic seed
    rather than by dict order, sorted by descending size, and then placed in whichever
    bucket is currently smallest, with ties broken randomly. The result is
    `max_num_files` groups of roughly equal total bytes, minus any that stayed empty,
    which happens when there are fewer inputs than buckets.

    Repeated copies of the same input are not separated: when a manifest row has
    `repeat_count > 1`, two or more copies of that shard can land in the same output
    file.
    """
    counts = Counter(paths)
    _log_merge_input_views(counts, len(paths))

    grouped_paths: list[list[TokensMetadataPaths]] = [[] for _ in range(max_num_files)]
    grouped_sizes = [0] * max_num_files
    expanded_paths = [element for element, count in counts.items() for _ in range(count)]
    random.shuffle(expanded_paths)
    expanded_paths.sort(key=lambda path: path.size, reverse=True)
    for element in expanded_paths:
        minimum_size = min(grouped_sizes)
        candidates = [index for index, size in enumerate(grouped_sizes) if size == minimum_size]
        bucket = random.choice(candidates)
        grouped_paths[bucket].append(element)
        grouped_sizes[bucket] += element.size

    grouped_paths = [group for group in grouped_paths if len(group) > 0]

    return grouped_paths


def _log_merge_input_views(
    counts: Mapping[TokensMetadataPaths, int],
    total_uses: int,
) -> None:
    logger.info(
        "Merge inputs: %s uses · %s distinct inputs · max uses of any input: %s×",
        total_uses,
        len(counts),
        max(counts.values()),
    )


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
    """Seed the global random module with `seed` offset by this worker's rank."""
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

    total_bytes = sum(path.size for path in paths)
    total_values = total_bytes // tokenizer.dtype.itemsize
    logger.info(
        "merge start · %s inputs · %s shards · %s tokens (%s) · %s workers",
        len(paths),
        len(grouped_paths),
        _human_count(total_values),
        _human_bytes(total_bytes),
        max_workers,
    )

    init_fn = partial(_worker_init, seed=seed)
    started_at = time.monotonic()
    progress_lock = threading.Lock()
    progress_by_shard: dict[str, MergeGroupProgress] = {}

    def update_progress(update: MergeGroupProgress) -> None:
        with progress_lock:
            progress_by_shard[update.destination.stem] = update

    with ThreadPoolExecutor(max_workers=max_workers, initializer=init_fn) as pool:
        futures = set()
        for i, group in enumerate(grouped_paths):
            future = pool.submit(
                merge_group,
                paths=group,
                destination=destination / f"{i:06d}.npy",
                dtype=tokenizer.dtype,
                progress=update_progress,
            )
            futures.add(future)

        pending = futures
        completed_shards = 0
        completed_metadata_bytes = 0
        completed_documents = 0
        while pending:
            done, pending = wait(
                pending,
                timeout=PROGRESS_INTERVAL_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                try:
                    result = future.result()
                except Exception:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise
                completed_shards += 1
                completed_metadata_bytes += result.metadata_bytes
                completed_documents += result.document_count
            if pending:
                elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                with progress_lock:
                    progress_snapshot = tuple(progress_by_shard.values())
                processed_values = sum(update.processed_values for update in progress_snapshot)
                processed_documents = sum(update.document_count for update in progress_snapshot)
                active_shards = sum(not update.complete for update in progress_snapshot)
                queued_shards = len(grouped_paths) - len(progress_snapshot)
                percent = 100 * processed_values / total_values
                queued = f" · {queued_shards} queued" if queued_shards else ""
                logger.info(
                    "merge %.1f%% · %s/%s tokens · %s tokens/s · %s docs/s · " "%s/%s done · %s active%s · %s",
                    percent,
                    _human_count(processed_values),
                    _human_count(total_values),
                    _human_count(round(processed_values / elapsed_seconds)),
                    _human_count(round(processed_documents / elapsed_seconds)),
                    completed_shards,
                    len(grouped_paths),
                    active_shards,
                    queued,
                    _elapsed(started_at),
                )

    elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
    logger.info(
        "merge done · %s shards · %s tokens (%s) · %s metadata · %s docs · " "%s tokens/s · %s docs/s · %s",
        len(grouped_paths),
        _human_count(total_values),
        _human_bytes(total_bytes),
        _human_bytes(completed_metadata_bytes),
        _human_count(completed_documents),
        _human_count(round(total_values / elapsed_seconds)),
        _human_count(round(completed_documents / elapsed_seconds)),
        _elapsed(started_at),
    )


@dataclass
class ReshardingPrefixConfig:
    """A source prefix and the rate at which the npy files under it are sampled.

    `download` copies an s3 prefix to a local directory; `take` lists the local
    prefix and applies `sample_rate`.
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

        # Take floor(sample_rate) copies of every path.
        if (repetition_rate := int(math.floor(self.sample_rate))) > 0:
            new_paths.extend(paths * repetition_rate)

        # The fractional remainder is drawn as a uniform random sample of files. Exact
        # token-level sampling would need an ILP solver, since the npys are uneven in
        # size; this approximation holds when most npys are of similar size.
        if (residual_frac := self.sample_rate - repetition_rate) > 0:
            new_paths.extend(random.sample(paths, max(1, round(residual_frac * len(paths)))))

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
class PreparedManifestRow:
    """One manifest row resolved to the local paths its later phases operate on.

    `npy_path` and `metadata_path` are where the two objects live once
    `ReshardingManifestConfig._plan_downloads` has run: inside the row's private
    directory under the run prefix for remote rows, or wherever the manifest pointed
    for local ones. `selection_path` is where a deterministic partial selection index
    is written, and is unused by rows that do not request one. Expected object sizes
    are not duplicated here; `entry` holds what the planner recorded.
    """

    entry: ReshardingManifestEntry
    npy_path: Path
    metadata_path: Path
    selection_path: Path


@dataclass(frozen=True)
class DocumentSelectionJob:
    manifest_index: int
    source_uri: str
    metadata_path: Path
    selection_path: Path
    source_uint32_values: int
    target_uint32_values: int
    seed: int


@dataclass(frozen=True)
class CompletedDocumentSelection:
    manifest_index: int
    result: DocumentSelectionResult


def _run_document_selection(job: DocumentSelectionJob) -> CompletedDocumentSelection:
    """Create one selection index in a process-pool worker."""

    started_at = time.monotonic()
    current_pass = ""
    pass_started_at = started_at
    source_document_count: int | None = None

    def report_progress(pass_name: str, document_count: int, covered_values: int) -> None:
        nonlocal current_pass, pass_started_at, source_document_count
        now = time.monotonic()
        if current_pass != pass_name:
            current_pass = pass_name
            pass_started_at = now
            logger.info("Document selection %s %s started", job.source_uri, pass_name)
        if document_count == 0:
            return

        pass_seconds = max(now - pass_started_at, 1e-9)
        if pass_name == "pass 1/2" and covered_values == job.source_uint32_values:
            source_document_count = document_count
        if pass_name == "pass 2/2" and source_document_count is not None:
            documents = f"{_human_count(document_count)}/{_human_count(source_document_count)} documents"
        else:
            documents = f"{_human_count(document_count)} documents"
        logger.info(
            "Document selection %s %s: %s · %s/%s values scanned · " "%s documents/s · %s values/s · elapsed %s",
            job.source_uri,
            pass_name,
            documents,
            _human_count(covered_values),
            _human_count(job.source_uint32_values),
            _human_count(round(document_count / pass_seconds)),
            _human_count(round(covered_values / pass_seconds)),
            _elapsed(pass_started_at),
        )

    result = create_document_selection(
        metadata_path=job.metadata_path,
        selection_path=job.selection_path,
        source_uint32_values=job.source_uint32_values,
        target_uint32_values=job.target_uint32_values,
        seed=job.seed,
        progress=report_progress,
    )
    elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
    scanned_documents = result.source_document_count * 2
    scanned_values = job.source_uint32_values * 2
    logger.info(
        "Document selection %s complete: %s/%s documents selected · %s values selected "
        "for %s requested (residual %+d) · %s documents/s · %s values/s · elapsed %s",
        job.source_uri,
        _human_count(result.selected_document_count),
        _human_count(result.source_document_count),
        _human_count(result.selected_uint32_values),
        _human_count(result.requested_uint32_values),
        result.target_residual_uint32_values,
        _human_count(round(scanned_documents / elapsed_seconds)),
        _human_count(round(scanned_values / elapsed_seconds)),
        _elapsed(started_at),
    )
    return CompletedDocumentSelection(
        manifest_index=job.manifest_index,
        result=result,
    )


@dataclass(frozen=True)
class ReshardingManifestConfig:
    """An exact, locally stored manifest of token/metadata object pairs.

    Version-one CSVs contain ``npy_uri``, ``metadata_uri``, and
    ``repeat_count``. Version two may also request a deterministic partial copy
    with ``partial_target_uint32_values`` and ``selection_seed``. Remote
    objects are downloaded exactly once into a run-owned directory.
    """

    manifest: str | Path

    def take(
        self,
        local_prefix: str | Path,
        max_workers: int,
        s5cmd_concurrency: int = DEFAULT_S5CMD_DOWNLOAD_CONCURRENCY,
    ) -> list[TokensMetadataPaths]:
        """Resolve this manifest into the merge inputs it names.

        Six phases, in order: parse and validate the CSV, HEAD every remote object to
        confirm it still matches the size and ETag the planner recorded, lay out one
        local directory per row, run a single batched s5cmd download, re-verify the
        downloaded sizes, and build any deterministic partial selections in a process
        pool.

        The returned list holds one entry per use of a source shard rather than one per
        shard: a row with ``repeat_count`` 3 contributes three equal entries, and a row
        that also requests a partial selection contributes one more.
        """

        manifest = Path(self.manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Resharding manifest does not exist: {manifest}")

        run_prefix = Path(local_prefix)
        run_prefix.mkdir(parents=True, exist_ok=False)

        rows = self._parse_manifest_rows(manifest)
        self._verify_remote_sources(rows, max_workers=max_workers)
        prepared, remote_commands = self._plan_downloads(
            rows, local_prefix=run_prefix, s5cmd_concurrency=s5cmd_concurrency
        )
        self._download_remote_objects(prepared, remote_commands, local_prefix=run_prefix, max_workers=max_workers)
        self._verify_downloaded_sizes(prepared)

        row_paths, selection_jobs = self._build_row_views(prepared)
        selection_views = self._run_selection_jobs(selection_jobs, prepared, max_workers=max_workers)
        for manifest_index, selection_view in selection_views.items():
            row_paths[manifest_index].append(selection_view)

        paths = [path for per_row_paths in row_paths for path in per_row_paths]

        if not paths:
            raise RuntimeError(f"Manifest document selections produced no output: {manifest}")
        selected_bytes = sum(path.size for path in paths)
        logger.info(
            "Manifest ready: %s merge input uses from %s source shards · %s planned output (%s uint32 values)",
            len(paths),
            len(rows),
            _human_bytes(selected_bytes),
            _human_count(selected_bytes // np.dtype(np.uint32).itemsize),
        )
        return paths

    @classmethod
    def _parse_manifest_rows(cls, manifest: Path) -> list[ReshardingManifestEntry]:
        """Read and validate every row of the manifest CSV.

        The header must carry at least the three version-one columns. Each row is then
        validated by `_parse_manifest_row`, which names ``manifest:<line>`` in every
        message. Line numbers start at 2 because line 1 is the header.
        """

        rows: list[ReshardingManifestEntry] = []
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            expected = {"npy_uri", "metadata_uri", "repeat_count"}
            if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                raise ValueError(f"Manifest {manifest} must contain columns {sorted(expected)}")
            for row_number, csv_row in enumerate(reader, start=2):
                rows.append(cls._parse_manifest_row(csv_row, manifest=manifest, row_number=row_number))

        if not rows:
            raise ValueError(f"Resharding manifest is empty: {manifest}")

        known_source_bytes = sum((entry.npy_size_bytes or 0) + (entry.metadata_size_bytes or 0) for entry in rows)
        planned_uint32_values = sum(
            (entry.npy_size_bytes or 0) // np.dtype(np.uint32).itemsize * entry.repeat_count
            + entry.partial_target_uint32_values
            for entry in rows
        )
        logger.info(
            "Manifest loaded: %s source shards · %s objects · %s source data · "
            "%s planned uint32 values · %s partial selections",
            len(rows),
            len(rows) * 2,
            _human_bytes(known_source_bytes),
            _human_count(planned_uint32_values),
            sum(entry.partial_target_uint32_values > 0 for entry in rows),
        )
        return rows

    @staticmethod
    def _parse_manifest_row(
        csv_row: Mapping[str, str],
        manifest: Path,
        row_number: int,
    ) -> ReshardingManifestEntry:
        """Validate one CSV row and turn it into a `ReshardingManifestEntry`.

        Sizes and ETags are optional for local rows and mandatory for ``s3://`` ones,
        which is what `_verify_remote_sources` and `_verify_downloaded_sizes` compare
        against. A row requesting a partial selection also needs a uint32-aligned
        source size strictly larger than the requested count, and must name a selection
        algorithm this build implements.
        """

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
                raise ValueError(f"Partial manifest rows require npy_size_bytes on {manifest}:{row_number}")
            if npy_size_bytes % np.dtype(np.uint32).itemsize:
                raise ValueError(f"Partial source size is not uint32-aligned on {manifest}:{row_number}")
            source_values = npy_size_bytes // np.dtype(np.uint32).itemsize
            if partial_target >= source_values:
                raise ValueError(
                    f"partial_target_uint32_values must be smaller than the source on {manifest}:{row_number}"
                )
            selection_algorithm = str(csv_row.get("selection_algorithm") or DOCUMENT_SELECTION_ALGORITHM)
            if selection_algorithm != DOCUMENT_SELECTION_ALGORITHM:
                raise ValueError(
                    f"Unsupported selection_algorithm on {manifest}:{row_number}: {selection_algorithm}"
                )
        else:
            selection_algorithm = DOCUMENT_SELECTION_ALGORITHM
        return ReshardingManifestEntry(
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

    @staticmethod
    def _verify_remote_sources(rows: list[ReshardingManifestEntry], max_workers: int) -> None:
        """HEAD every remote object and confirm its size and ETag still match.

        A source object rewritten since the manifest was planned would change which
        tokens this run emits, and for partial rows which documents get selected, so a
        mismatch aborts before anything is downloaded. Rows whose recorded ETag is
        empty are size-checked only. No-ops when the manifest is entirely local.
        """

        remote_expectations: list[tuple[str, int, str]] = []
        for entry in rows:
            if urlparse(entry.npy_uri).scheme == "s3":
                assert entry.npy_size_bytes is not None
                assert entry.metadata_size_bytes is not None
                remote_expectations.extend(
                    [
                        (entry.npy_uri, entry.npy_size_bytes, entry.npy_etag),
                        (entry.metadata_uri, entry.metadata_size_bytes, entry.metadata_etag),
                    ]
                )
        if not remote_expectations:
            return

        client = boto3.client("s3")
        validation_started_at = time.monotonic()
        logger.info(
            "Source validation started: checking size and ETag for %s objects",
            len(remote_expectations),
        )

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
            checkpoint = max(1, len(futures) // 10)
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                if completed == len(futures) or completed % checkpoint == 0:
                    logger.info(
                        "Source validation progress: %s/%s objects · elapsed %s",
                        completed,
                        len(futures),
                        _elapsed(validation_started_at),
                    )

    @staticmethod
    def _plan_downloads(
        rows: list[ReshardingManifestEntry],
        local_prefix: Path,
        s5cmd_concurrency: int,
    ) -> tuple[list[PreparedManifestRow], list[str]]:
        """Give every row a private local directory and resolve its object paths.

        Remote rows are copied into ``<local_prefix>/<index>/tokens.{npy,csv.gz}`` and
        contribute two s5cmd `cp` command lines each. Local rows are read where they
        already are and only checked for existence, so nothing is copied for them. A
        single row may not mix schemes across its two objects.

        Returns the prepared rows in manifest order, plus the s5cmd command lines,
        which are empty when there is nothing to download.
        """

        prepared: list[PreparedManifestRow] = []
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
                        (
                            "cp --show-progress --raw --no-clobber "
                            f"--concurrency {s5cmd_concurrency} "
                            f"{shlex.quote(npy_uri)} {shlex.quote(str(local_npy))}"
                        ),
                        (
                            "cp --show-progress --raw --no-clobber "
                            f"--concurrency {s5cmd_concurrency} "
                            f"{shlex.quote(metadata_uri)} {shlex.quote(str(local_metadata))}"
                        ),
                    ]
                )
            elif npy_scheme in {"", "file"} and metadata_scheme in {"", "file"}:
                local_npy = Path(urlparse(npy_uri).path if npy_scheme == "file" else npy_uri)
                local_metadata = Path(urlparse(metadata_uri).path if metadata_scheme == "file" else metadata_uri)
                if not local_npy.is_file() or not local_metadata.is_file():
                    raise FileNotFoundError(f"Manifest object pair does not exist: {npy_uri}, {metadata_uri}")
            else:
                raise ValueError(f"Manifest row mixes unsupported URI schemes: {npy_uri}, {metadata_uri}")

            prepared.append(
                PreparedManifestRow(
                    entry=entry,
                    npy_path=local_npy,
                    metadata_path=local_metadata,
                    selection_path=row_dir / "selection.csv.gz",
                )
            )
        return prepared, remote_commands

    @staticmethod
    def _download_remote_objects(
        prepared: list[PreparedManifestRow],
        remote_commands: list[str],
        local_prefix: Path,
        max_workers: int,
    ) -> None:
        """Run one batched s5cmd job and confirm it created every object.

        All command lines go through a single command file, so one s5cmd process
        handles the whole manifest. s5cmd can exit zero having skipped work, so the
        object pairs are checked afterwards. No-ops when nothing needs downloading.
        """

        if not remote_commands:
            return

        commands_path = local_prefix / "s5cmd-download-commands.txt"
        with commands_path.open("x") as f:
            f.write("\n".join(remote_commands) + "\n")
        cmd = ["s5cmd", "--stat", "--numworkers", str(max_workers), "run", str(commands_path)]
        _run_s5cmd(cmd, "source download")

        missing = [
            (row.npy_path, row.metadata_path)
            for row in prepared
            if not row.npy_path.is_file() or not row.metadata_path.is_file()
        ]
        if missing:
            raise RuntimeError(f"s5cmd completed without creating {len(missing)} manifest object pairs")

    @staticmethod
    def _verify_downloaded_sizes(prepared: list[PreparedManifestRow]) -> None:
        """Re-check the on-disk sizes against the manifest after downloading.

        `_verify_remote_sources` checks the source before the copy; this checks the
        copy itself, so a truncated or partially resumed download cannot reach the
        merge. Rows whose manifest recorded no size, only possible for local rows, are
        skipped.
        """

        for row in prepared:
            npy_path = row.npy_path
            metadata_path = row.metadata_path
            expected_npy_size = row.entry.npy_size_bytes
            expected_metadata_size = row.entry.metadata_size_bytes
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

    @staticmethod
    def _build_row_views(
        prepared: list[PreparedManifestRow],
    ) -> tuple[list[list[TokensMetadataPaths]], list[DocumentSelectionJob]]:
        """Expand each row's whole-file repeats and collect its selection job, if any.

        Returns one list of merge inputs per manifest row, holding ``repeat_count``
        copies of the full object, which may be zero for a purely partial row, plus one
        `DocumentSelectionJob` for every row that also requests a partial selection.
        The partial view is appended to its row later, once its selection index exists.
        """

        row_paths: list[list[TokensMetadataPaths]] = [[] for _ in prepared]
        selection_jobs: list[DocumentSelectionJob] = []
        for manifest_index, row in enumerate(prepared):
            entry = row.entry
            full_path = TokensMetadataPaths(str(row.npy_path), str(row.metadata_path))
            row_paths[manifest_index].extend([full_path] * entry.repeat_count)
            if entry.partial_target_uint32_values:
                source_values = row.npy_path.stat().st_size // np.dtype(np.uint32).itemsize
                selection_jobs.append(
                    DocumentSelectionJob(
                        manifest_index=manifest_index,
                        source_uri=entry.npy_uri,
                        metadata_path=row.metadata_path,
                        selection_path=row.selection_path,
                        source_uint32_values=source_values,
                        target_uint32_values=entry.partial_target_uint32_values,
                        seed=entry.selection_seed,
                    )
                )
        return row_paths, selection_jobs

    @staticmethod
    def _run_selection_jobs(
        selection_jobs: list[DocumentSelectionJob],
        prepared: list[PreparedManifestRow],
        max_workers: int,
    ) -> dict[int, TokensMetadataPaths]:
        """Build every requested partial selection index in a process pool.

        Selection is CPU-bound, two passes over a shard's metadata, hence processes
        rather than threads. Returns the partial merge-input view for each row that
        selected at least one value, keyed by manifest index; a row whose selection
        came back empty is omitted. Every row has at most one selection job, so a
        manifest index appears at most once. No-ops when no row requested a selection.
        """

        if not selection_jobs:
            return {}

        selection_views: dict[int, TokensMetadataPaths] = {}
        selection_workers = min(max_workers, len(selection_jobs))
        selection_started_at = time.monotonic()
        total_source_values = sum(job.source_uint32_values for job in selection_jobs)
        total_target_values = sum(job.target_uint32_values for job in selection_jobs)
        logger.info(
            "Document selection started: %s source shards · %s processes · %s source values per pass · "
            "%s requested values",
            len(selection_jobs),
            selection_workers,
            _human_count(total_source_values),
            _human_count(total_target_values),
        )
        completed_source_values = 0
        completed_scanned_documents = 0
        selected_values = 0
        with ProcessPoolExecutor(max_workers=selection_workers) as pool:
            futures = {pool.submit(_run_document_selection, job): job for job in selection_jobs}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                try:
                    completed = future.result()
                except Exception:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                job = futures[future]
                selection = completed.result
                if selection.selected_uint32_values:
                    source_row = prepared[completed.manifest_index]
                    selection_views[completed.manifest_index] = TokensMetadataPaths(
                        str(source_row.npy_path),
                        str(source_row.metadata_path),
                        selection_path=str(selection.selection_path),
                        selected_uint32_values=selection.selected_uint32_values,
                    )
                completed_source_values += job.source_uint32_values
                completed_scanned_documents += selection.source_document_count * 2
                selected_values += selection.selected_uint32_values
                wall_seconds = max(time.monotonic() - selection_started_at, 1e-9)
                logger.info(
                    "Document selection progress: %s/%s source shards · %s/%s values selected · "
                    "%s documents/s · %s values/s across both passes · elapsed %s",
                    completed_count,
                    len(selection_jobs),
                    _human_count(selected_values),
                    _human_count(total_target_values),
                    _human_count(round(completed_scanned_documents / wall_seconds)),
                    _human_count(round((completed_source_values * 2) / wall_seconds)),
                    _elapsed(selection_started_at),
                )
        wall_seconds = max(time.monotonic() - selection_started_at, 1e-9)
        logger.info(
            "Document selection complete: %s source shards · %s/%s values selected · "
            "%s documents/s · %s values/s across both passes · elapsed %s",
            len(selection_jobs),
            _human_count(selected_values),
            _human_count(total_target_values),
            _human_count(round(completed_scanned_documents / wall_seconds)),
            _human_count(round((completed_source_values * 2) / wall_seconds)),
            _elapsed(selection_started_at),
        )
        return selection_views

    def to_dict(self) -> dict:
        return {"manifest": str(self.manifest)}

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path | None = None) -> "ReshardingManifestConfig":
        path = Path(d["manifest"])
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return cls(manifest=path)


@dataclass(kw_only=True)
class ReshardingConfig:
    """Base configuration for resharding.

    Fields are keyword-only: `source_prefixes` is one of two alternative source lists
    and is not the first field, so positional construction would bind arguments to the
    wrong fields.
    """

    destination_prefix: str
    source_prefixes: list[ReshardingPrefixConfig] = field(default_factory=list)
    source_manifests: list[ReshardingManifestConfig] = field(default_factory=list)
    local_tempdir: str | Path | None = None
    max_size_bytes: int | None = None
    max_num_files: int | None = None
    max_workers: int = os.cpu_count() or 1
    s5cmd_download_concurrency: int = DEFAULT_S5CMD_DOWNLOAD_CONCURRENCY
    random_seed: int = 42
    tokenizer_name_or_path: str = "allenai/dolma2-tokenizer"
    # Tombstone: resharding always refuses a populated destination, and the field
    # exists so that a config setting it True is rejected rather than ignored.
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
        if self.s5cmd_download_concurrency <= 0:
            raise ValueError("s5cmd_download_concurrency must be positive")

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
            s5cmd_download_concurrency=int(
                d.get("s5cmd_download_concurrency", DEFAULT_S5CMD_DOWNLOAD_CONCURRENCY)
            ),
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
    """Upload a local directory to S3 with no-clobber semantics.

    No-ops when `remote_prefix` is not an s3 URI.
    """
    if urlparse(remote_prefix).scheme != "s3":
        return

    local_prefix_no_star = re.sub(r"(/|/\*)$", "", str(local_prefix))
    remote_prefix_no_trailing_slash = str(remote_prefix).rstrip("/")
    cmd = [
        "s5cmd",
        "--stat",
        "--numworkers",
        str(max_workers),
        "cp",
        "--no-clobber",
        "--show-progress",
        f"{local_prefix_no_star}/*",
        f"{remote_prefix_no_trailing_slash}/",
    ]
    _run_s5cmd(cmd, "output upload")


def destination_has_objects(destination: str | Path) -> bool:
    """Return whether a destination already contains data.

    S3 is queried with a read-only, one-key ``ListObjectsV2`` request. Local
    destinations count as occupied when the path exists at all, including an empty
    pre-created directory. Raises ValueError for an S3 bucket root or an unsupported
    scheme.
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
    started_at = time.monotonic()
    logger.info(
        "Reshard started: %s source manifests · %s source prefixes · %s workers · destination=%s",
        len(config.source_manifests),
        len(config.source_prefixes),
        config.max_workers,
        config.destination_prefix,
    )

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

        disk = shutil.disk_usage(run_tempdir)
        logger.info(
            "Working directory ready: %s · %s free of %s",
            run_tempdir,
            _human_bytes(disk.free),
            _human_bytes(disk.total),
        )

        local_output_dir = (
            run_tempdir / "output"
            if urlparse(config.destination_prefix).scheme == "s3"
            else Path(config.destination_prefix)
        )

        logger.info("Source preparation started")
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
                    s5cmd_concurrency=config.s5cmd_download_concurrency,
                )
            )

        planned_bytes = sum(path.size for path in source_paths)
        logger.info(
            "Source preparation complete: %s merge input uses · %s distinct token/metadata "
            "views · %s planned output (%s uint32 values) · elapsed %s",
            len(source_paths),
            len(set(source_paths)),
            _human_bytes(planned_bytes),
            _human_count(planned_bytes // np.dtype(np.uint32).itemsize),
            _elapsed(started_at),
        )

        local_output_dir.mkdir(parents=True, exist_ok=False)

        merge_all_npys(
            source_paths,
            destination=local_output_dir,
            max_size_bytes=config.max_size_bytes,
            max_num_files=config.max_num_files,
            max_workers=config.max_workers,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            seed=config.random_seed,
        )

        output_files = [path for path in local_output_dir.rglob("*") if path.is_file()]
        output_bytes = sum(path.stat().st_size for path in output_files)
        logger.info(
            "Local output ready: %s files · %s · elapsed %s",
            len(output_files),
            _human_bytes(output_bytes),
            _elapsed(started_at),
        )

        upload_to_s3(
            local_prefix=local_output_dir,
            remote_prefix=config.destination_prefix,
            max_workers=config.max_workers,
        )
        logger.info(
            "Reshard complete: %s files · %s · total elapsed %s",
            len(output_files),
            _human_bytes(output_bytes),
            _elapsed(started_at),
        )

    finally:
        if run_tempdir is not None:
            logger.info("Removing run working directory: %s", run_tempdir)
            shutil.rmtree(run_tempdir)


def main():
    config = ReshardingConfig.from_file(sys.argv[1])
    reshard(config)


if __name__ == "__main__":
    main()
