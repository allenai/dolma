"""Diagnostics helper for resharding configs without downloading data.

This script mirrors the sampling logic in ``reshard.py`` but stops after computing
basic statistics about the sampled file set. It only relies on remote file
metadata (file names and sizes) so we can estimate repetition counts without
copying the full data locally.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlparse

from dolma.core.loggers import get_logger
from dolma.tokenizer.reshard import ReshardingConfig, ReshardingPrefixConfig

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class TokensMetadataInfo:
    """Container mirroring ``TokensMetadataPaths`` but using cached sizes."""

    npy_path: str
    csv_path: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.npy_path.endswith(".npy"):
            raise ValueError(f"Invalid npy path: {self.npy_path}")
        if not self.csv_path.endswith(".csv.gz"):
            raise ValueError(f"Invalid csv.gz path: {self.csv_path}")
        # Ensure the base filenames match so we mimic the original invariant.
        npy_stem = Path(self.npy_path).stem
        csv_stem = Path(Path(self.csv_path).stem).stem
        if npy_stem != csv_stem:
            raise ValueError(f"Mismatched stems: {self.npy_path} vs {self.csv_path}")

    @property
    def size(self) -> int:
        return self.size_bytes


@dataclass
class PrefixSamplingDiagnostics:
    """Summary of requested vs realized sampling for a prefix."""

    prefix: ReshardingPrefixConfig
    requested_rate: float
    actual_rate: float
    available_bytes: int
    sampled_bytes: int

    @property
    def delta(self) -> float:
        return self.actual_rate - self.requested_rate

    @property
    def delta_pct(self) -> float | None:
        if self.requested_rate == 0:
            return None
        return self.delta / self.requested_rate


def _list_local_files(path_pattern: str) -> List[TokensMetadataInfo]:
    """Enumerate .npy/.csv.gz pairs for a local path or glob."""
    infos: list[TokensMetadataInfo] = []

    if "*" in path_pattern:
        import glob

        for npy_path in sorted(glob.glob(path_pattern, recursive=True)):
            if not npy_path.endswith(".npy"):
                continue
            csv_path = npy_path[:-4] + ".csv.gz"
            if not os.path.exists(csv_path):
                logger.warning("Missing CSV companion for %s", npy_path)
                continue
            infos.append(
                TokensMetadataInfo(
                    npy_path=npy_path,
                    csv_path=csv_path,
                    size_bytes=os.path.getsize(npy_path),
                )
            )
        return infos

    path = Path(path_pattern)
    if path.is_file() and path.suffix == ".npy":
        csv_path = path.with_suffix(".csv.gz")
        if not csv_path.exists():
            logger.warning("Missing CSV companion for %s", path)
        else:
            infos.append(
                TokensMetadataInfo(
                    npy_path=str(path),
                    csv_path=str(csv_path),
                    size_bytes=path.stat().st_size,
                )
            )
        return infos

    if not path.exists():
        logger.warning("Local path %s does not exist", path)
        return infos

    for root, _, files in os.walk(path):
        for file_name in sorted(files):
            if not file_name.endswith(".npy"):
                continue
            npy_path = Path(root) / file_name
            csv_path = npy_path.with_suffix(".csv.gz")
            if not csv_path.exists():
                logger.warning("Missing CSV companion for %s", npy_path)
                continue
            infos.append(
                TokensMetadataInfo(
                    npy_path=str(npy_path),
                    csv_path=str(csv_path),
                    size_bytes=npy_path.stat().st_size,
                )
            )
    return infos


def _list_s3_files(path_pattern: str) -> List[TokensMetadataInfo]:
    """Use ``s5cmd ls`` to list .npy files and sizes on S3."""
    pattern = path_pattern
    if not re.search(r"[\*\?]", pattern) and not pattern.endswith(".npy"):
        pattern = re.sub(r"/+$", "", pattern) + "/*.npy"

    cmd = ["s5cmd", "ls", pattern]
    logger.info("Listing S3 objects with: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    infos: list[TokensMetadataInfo] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("total "):
            continue
        try:
            metadata, path = line.rsplit(maxsplit=1)
        except ValueError:
            continue
        if not path.endswith(".npy"):
            continue
        size_str = metadata.split()[-1]
        try:
            size = int(size_str)
        except ValueError:
            logger.debug("Could not parse size from line: %s", line)
            continue
        infos.append(
            TokensMetadataInfo(
                npy_path=path,
                csv_path=path[:-4] + ".csv.gz",
                size_bytes=size,
            )
        )
    return infos


def _list_files(path_pattern: str) -> List[TokensMetadataInfo]:
    scheme = urlparse(path_pattern).scheme
    if scheme in {"", "file"}:
        return _list_local_files(path_pattern)
    if scheme == "s3":
        return _list_s3_files(path_pattern)

    raise ValueError(f"Unsupported scheme for path {path_pattern}")


def _collect_source_file_map(
    prefixes: Iterable[ReshardingPrefixConfig],
) -> Dict[str, List[TokensMetadataInfo]]:
    mapping: dict[str, list[TokensMetadataInfo]] = {}
    for prefix in prefixes:
        for path in prefix.paths or []:
            path_str = str(path)
            mapping[path_str] = _list_files(path_str)
            if not mapping[path_str]:
                logger.warning("No files found for %s", path_str)
    return mapping


def _compute_total_source_sizes(
    prefixes: Iterable[ReshardingPrefixConfig],
    mapping: Dict[str, List[TokensMetadataInfo]],
) -> Dict[str, int] | None:
    if not any(prefix.target_ratio is not None for prefix in prefixes):
        return None

    totals: dict[str, int] = {}
    for prefix in prefixes:
        for path in prefix.paths or []:
            infos = mapping.get(str(path), [])
            totals[str(path)] = sum(info.size for info in infos)
    return totals


def _sample_paths_for_prefix(
    prefix: ReshardingPrefixConfig,
    mapping: Dict[str, List[TokensMetadataInfo]],
    total_source_sizes: Dict[str, int] | None,
) -> tuple[List[TokensMetadataInfo], PrefixSamplingDiagnostics | None]:
    sample_rate = prefix.get_sample_rate(total_source_sizes)
    paths: list[TokensMetadataInfo] = []
    for path in prefix.paths or []:
        paths.extend(mapping.get(str(path), []))

    paths = sorted(paths, key=lambda p: (p.npy_path, p.csv_path))
    if not paths:
        logger.warning("No data available for prefix %s", prefix)
        return [], None

    available_bytes = sum(info.size for info in paths)
    if available_bytes == 0:
        logger.warning("Total size for prefix %s is 0 bytes", prefix)
        diagnostics = PrefixSamplingDiagnostics(
            prefix=prefix,
            requested_rate=sample_rate,
            actual_rate=0.0,
            available_bytes=0,
            sampled_bytes=0,
        )
        return [], diagnostics

    new_paths: list[TokensMetadataInfo] = []
    repetition_rate = int(math.floor(sample_rate))
    if repetition_rate > 0:
        new_paths.extend(paths * repetition_rate)

    residual_frac = sample_rate - repetition_rate
    if residual_frac > 0:
        sample_size = min(len(paths), max(1, round(residual_frac * len(paths))))
        new_paths.extend(random.sample(paths, sample_size))

    logger.info(
        "Prefix %s: Taking %d paths from %d available paths using %.6f sample rate",
        prefix.paths[0] if prefix.paths else "unknown",
        len(new_paths),
        len(paths),
        sample_rate,
    )
    sampled_bytes = sum(info.size for info in new_paths)
    actual_rate = sampled_bytes / available_bytes if available_bytes else 0.0
    diagnostics = PrefixSamplingDiagnostics(
        prefix=prefix,
        requested_rate=sample_rate,
        actual_rate=actual_rate,
        available_bytes=available_bytes,
        sampled_bytes=sampled_bytes,
    )
    return new_paths, diagnostics


def run_diagnostics(config_path: str) -> None:
    config = ReshardingConfig.from_file(config_path)
    random.seed(config.random_seed)
    mapping = _collect_source_file_map(config.source_prefixes)
    total_source_sizes = _compute_total_source_sizes(config.source_prefixes, mapping)

    sampled_paths: list[TokensMetadataInfo] = []
    prefix_diagnostics: list[PrefixSamplingDiagnostics] = []
    for prefix in config.source_prefixes:
        sampled, diagnostics = _sample_paths_for_prefix(
            prefix, mapping, total_source_sizes
        )
        sampled_paths.extend(sampled)
        if diagnostics is not None:
            prefix_diagnostics.append(diagnostics)

    if not sampled_paths:
        logger.warning("No sampled paths collected; nothing to report")
        return

    counts = Counter(sampled_paths)
    unique_paths = len(counts)
    total_paths = len(sampled_paths)
    max_repetition = max(counts.values()) if counts else 0

    logger.info(
        "Found %d unique paths from %d sampled files; max repetition is %d",
        unique_paths,
        total_paths,
        max_repetition,
    )

    for i, diag in enumerate(prefix_diagnostics, start=1):
        delta_pct = diag.delta_pct
        delta_pct_str = f"{delta_pct * 100:.2f}%" if delta_pct is not None else "n/a"
        logger.info(
            "Prefix %d (%s) requested rate %.6f -> actual %.6f (Δ=%.6f, %s) "
            "using %d/%d bytes",
            i,
            diag.prefix.prefix if hasattr(diag.prefix, "prefix") else diag.prefix,
            diag.requested_rate,
            diag.actual_rate,
            diag.delta,
            delta_pct_str,
            diag.sampled_bytes,
            diag.available_bytes,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print resharding diagnostics without downloading files",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML resharding config",
    )
    args = parser.parse_args()
    run_diagnostics(args.config)


if __name__ == "__main__":
    main()
