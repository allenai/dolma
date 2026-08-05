"""Shared implementation for the Dolma 3.5 resharding preparation scripts.

This module has no command-line interface. Use the plainly named sibling
scripts for each workflow phase. Nothing here materializes token data.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence
from urllib.parse import unquote, urlparse

import boto3
import yaml

UINT32_BYTES = 4
DOCUMENT_SELECTION_ALGORITHM = "document_hash_bucket_v1"
EXECUTION_UNIT_INDEX_WIDTH = 8
EXECUTION_LAYOUT_SCHEMA_VERSION = 1
DESTINATION_LAYOUT = "build-scoped-category-output-v1"
DEFAULT_TARGET = 14_000_000_000_000
DEFAULT_REGION = "us-east-1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_PATH = REPOSITORY_ROOT / "runs/dolma3p5-resharding/14t"
PREPARATION_PHASES = (
    "01-plan",
    "02-preflight",
    "03-output-validation",
)
PLAN_STAGES = ("resolution", "inventory", "execution")
PLAN_ROOT_ARTIFACTS = {"report.html"}
LEGACY_PREPARATION_PHASES = (
    "02-inventory",
    "03-proposal",
    "04-preflight",
    "05-output-validation",
)
PRESERVED_BUILD_METADATA = {".DS_Store"}
DEFAULT_SETTINGS: dict[str, Any] = {
    "target_uint32_values": DEFAULT_TARGET,
    "direct_s3_bucket": "ai2-llm",
    "s5cmd_numworkers": 64,
    "s5cmd_retry_count": 10,
    "inventory_max_workers": 32,
    "max_listing_catalog_objects": 5_000,
    "max_listing_overfetch_ratio": 8.0,
    "minimum_catalog_prefix_components": 4,
    "random_seed": 42,
    "max_workers_per_reshard": 8,
    "tokenizer_name_or_path": "allenai/dolma2-tokenizer",
    "maximum_expected_upsample_rate": None,
    "max_materialized_unit_target_residual_fraction": 0.001,
    "max_materialized_total_target_residual_fraction": 0.00001,
}


class PreparationError(RuntimeError):
    """A user-actionable preparation or validation failure."""


def normalize_region(region: str | None) -> str:
    """Return the configured region, falling back to the workflow default."""

    normalized = (region or "").strip()
    return normalized or DEFAULT_REGION


@dataclass(frozen=True)
class S3Object:
    bucket: str
    key: str
    size_bytes: int
    etag: str = ""
    last_modified: str = ""
    storage_class: str = ""
    source: str = "listing"

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class S5cmdRunResult:
    returncode: int
    stderr: str
    output_records: int
    elapsed_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_label(value: int, singular: str, plural: str | None = None) -> str:
    label = singular if value == 1 else plural or f"{singular}s"
    return f"{value:,} {label}"


def _slug(value: str, max_length: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.").lower()
    if not value:
        value = "unnamed"
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{value[: max_length - 9]}-{suffix}"


def _validate_preparation_build(path: Path) -> dict[str, Any]:
    """Verify that an existing directory is owned by this preparation workflow."""

    if path.is_symlink() or not path.is_dir():
        raise PreparationError(f"Preparation build is not a real directory: {path}")
    manifest_path = path / "build.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PreparationError(
            "Refusing to replace an unrecognized directory. Expected this "
            f"workflow's build.json marker in: {path}"
        )
    try:
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid preparation build marker: {manifest_path}") from exc
    build_id = manifest.get("build_id")
    mix_sha256 = manifest.get("mix_sha256")
    catalog_sha256 = manifest.get("catalog_sha256")
    hashes_are_valid = all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in (mix_sha256, catalog_sha256)
    )
    expected_build_id = ""
    if hashes_are_valid:
        seed = f"{mix_sha256}:{catalog_sha256}".encode()
        expected_build_id = f"dolma3p5-14t-{hashlib.sha256(seed).hexdigest()[:12]}"
    if manifest.get("schema_version") != 1 or not isinstance(build_id, str) or build_id != expected_build_id:
        raise PreparationError(f"Refusing to replace an unrecognized preparation build: {path}")
    recognized_phases = {*PREPARATION_PHASES, *LEGACY_PREPARATION_PHASES}
    unknown = sorted(
        child.name
        for child in path.iterdir()
        if child.name not in {"build.json", *recognized_phases, *PRESERVED_BUILD_METADATA}
    )
    if unknown:
        raise PreparationError(
            "Refusing to reset a preparation build containing unknown top-level " f"entries: {', '.join(unknown)}"
        )
    for phase_name in recognized_phases:
        phase = path / phase_name
        if phase.exists() and (phase.is_symlink() or not phase.is_dir()):
            raise PreparationError(f"Refusing to replace an unsafe preparation phase path: {phase}")
    return manifest


def _validate_execution_layout(build: Path) -> dict[str, Any]:
    """Refuse stale execution artifacts before checking or writing destinations."""

    manifest = _validate_preparation_build(build)
    layout_path = build / "01-plan/execution/dataset-layout.json"
    if layout_path.is_symlink() or not layout_path.is_file():
        raise PreparationError(
            "Execution layout is missing. Rerun scripts/dolma3p5_resharding/plan.py before preflight "
            "or materialization."
        )
    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid execution layout: {layout_path}") from exc
    expected = {
        "build_id": manifest["build_id"],
        "layout": DESTINATION_LAYOUT,
        "execution_unit_index_width": EXECUTION_UNIT_INDEX_WIDTH,
        "execution_unit_id_width": EXECUTION_UNIT_INDEX_WIDTH,
    }
    mismatches = [name for name, value in expected.items() if layout.get(name) != value]
    dataset_root = layout.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root.endswith(f'/{manifest["build_id"]}'):
        mismatches.append("dataset_root")
    if mismatches:
        found_layout = layout.get("layout", "missing")
        raise PreparationError(
            "Execution plan uses an obsolete destination layout "
            f"(layout={found_layout}). Rerun "
            "scripts/dolma3p5_resharding/plan.py before preflight or materialization."
        )
    return layout


def _remove_generated_phase(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise PreparationError(f"Refusing to replace an unsafe preparation phase path: {path}")
    shutil.rmtree(path)


def _reset_preparation_build(path: Path) -> None:
    """Reset only a recognized local preparation build, never source/token data."""

    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return
    if path.is_symlink() or not path.is_dir():
        raise PreparationError(f"Preparation output is not a real directory: {path}")
    if not any(path.iterdir()):
        return
    _validate_preparation_build(path)
    for phase_name in dict.fromkeys((*PREPARATION_PHASES, *LEGACY_PREPARATION_PHASES)):
        _remove_generated_phase(path / phase_name)
    (path / "build.json").unlink()


def _reset_preparation_phase(build: Path, phase_name: str, *downstream_phase_names: str) -> Path:
    """Replace generated local phases after verifying the build ownership marker."""

    _validate_preparation_build(build)
    names = (phase_name, *downstream_phase_names)
    if any(name not in PREPARATION_PHASES for name in names):
        raise ValueError(f"Unknown preparation phase: {names}")
    for name in names:
        _remove_generated_phase(build / name)
    phase = build / phase_name
    phase.mkdir(exist_ok=False)
    return phase


def _reset_plan_stage(build: Path, stage_name: str, *downstream_stage_names: str) -> Path:
    """Replace generated plan stages while preserving earlier reviewed stages."""

    _validate_preparation_build(build)
    plan_root = build / "01-plan"
    if plan_root.is_symlink() or not plan_root.is_dir():
        raise PreparationError(f"Plan output is not a real directory: {plan_root}")
    unknown = sorted(
        child.name
        for child in plan_root.iterdir()
        if child.name not in {*PLAN_STAGES, *PLAN_ROOT_ARTIFACTS, *PRESERVED_BUILD_METADATA}
    )
    if unknown:
        raise PreparationError("Refusing to reset a plan containing unknown entries: " + ", ".join(unknown))
    names = (stage_name, *downstream_stage_names)
    if any(name not in PLAN_STAGES for name in names):
        raise ValueError(f"Unknown plan stage: {names}")
    for artifact_name in PLAN_ROOT_ARTIFACTS:
        artifact = plan_root / artifact_name
        if artifact.exists():
            if artifact.is_symlink() or not artifact.is_file():
                raise PreparationError(f"Refusing to replace an unsafe plan artifact: {artifact}")
            artifact.unlink()
    for name in names:
        _remove_generated_phase(plan_root / name)
    for phase_name in ("02-preflight", "03-output-validation"):
        _remove_generated_phase(build / phase_name)
    stage = plan_root / stage_name
    stage.mkdir(exist_ok=False)
    return stage


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(value)
    except FileExistsError as exc:
        raise PreparationError(f"Refusing to replace existing artifact: {path}") from exc


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("x", newline="", encoding="utf-8")
    except FileExistsError as exc:
        raise PreparationError(f"Refusing to replace existing artifact: {path}") from exc
    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_settings(path: Path | None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if path is not None:
        with path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise PreparationError(f"Settings must be a YAML mapping: {path}")
        unknown = sorted(set(loaded) - set(DEFAULT_SETTINGS))
        if unknown:
            raise PreparationError(f"Unknown settings: {', '.join(unknown)}")
        settings.update(loaded)
    if int(settings["target_uint32_values"]) <= 0:
        raise PreparationError("target_uint32_values must be positive")
    for name in (
        "max_materialized_unit_target_residual_fraction",
        "max_materialized_total_target_residual_fraction",
    ):
        value = float(settings[name])
        if not 0 < value < 1:
            raise PreparationError(f"{name} must be between zero and one")
    maximum_expected_upsample_rate = settings["maximum_expected_upsample_rate"]
    if maximum_expected_upsample_rate is not None and float(maximum_expected_upsample_rate) <= 1:
        raise PreparationError("maximum_expected_upsample_rate must be greater than one")
    return settings


def _load_catalog(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise PreparationError(f"Catalog row {line_number} has fewer than two columns")
            bucket, encoded_key = row[0].strip(), row[1].strip()
            if line_number == 1 and bucket.lower() == "bucket" and encoded_key.lower() in {"key", "path"}:
                continue
            key = unquote(encoded_key)
            if not bucket:
                raise PreparationError(f"Invalid catalog object on row {line_number}: {row[:2]}")
            if any(ord(char) < 32 for char in bucket + key):
                raise PreparationError(f"Control character in catalog object on row {line_number}")
            if not key.endswith(".npy"):
                # The reference file is nominally an NPY catalog but currently
                # contains at least one metadata row. Metadata never defines
                # membership, so ignore it here and verify the derived partner
                # against S3 during inventory.
                continue
            rows.append({"bucket": bucket, "key": key, "catalog_line": str(line_number)})
    if not rows:
        raise PreparationError(f"Catalog contains no NPY objects: {path}")
    return rows


def _catalog_pattern(yaml_path: str) -> str:
    relative = yaml_path.removeprefix("dolma3p5_pool/")
    return relative if relative.startswith("preprocessed/") else f"preprocessed/{relative}"


def _direct_s3_pattern(yaml_path: str, bucket: str) -> tuple[str, str]:
    if not yaml_path.startswith("preprocessed/"):
        raise PreparationError(f"Unsupported direct path: {yaml_path}")
    return bucket, yaml_path


@lru_cache(maxsize=None)
def _compiled_path_pattern(pattern: str) -> re.Pattern[str]:
    # YAML paths use ``*`` for expansion. Treat every other character
    # literally so language names containing ``+``, ``[``, or ``?`` cannot be
    # reinterpreted as local glob syntax.
    expression = re.escape(unquote(pattern)).replace(r"\*", ".*")
    return re.compile(expression)


def _matches_key(key: str, pattern: str) -> bool:
    return _compiled_path_pattern(pattern).fullmatch(unquote(key)) is not None


def _literal_prefix(pattern: str) -> str:
    wildcard_offset = pattern.find("*")
    prefix = pattern[:wildcard_offset] if wildcard_offset >= 0 else pattern
    if not prefix.endswith("/"):
        prefix = prefix.rsplit("/", 1)[0] + "/"
    return prefix


def _common_directory_prefix(keys: Sequence[str]) -> str:
    if not keys:
        raise ValueError("Cannot find a common prefix for no keys")
    directories = [key.rsplit("/", 1)[0] for key in keys]
    common = posixpath.commonpath(directories)
    return common.rstrip("/") + "/"


def _pair_metadata_key(npy_key: str) -> str:
    if not npy_key.endswith(".npy"):
        raise ValueError(npy_key)
    return npy_key[:-4] + ".csv.gz"


def _build_id(mix_path: Path, catalog_path: Path) -> str:
    seed = f"{_sha256(mix_path)}:{_sha256(catalog_path)}".encode()
    return f"dolma3p5-14t-{hashlib.sha256(seed).hexdigest()[:12]}"


def plan_build(args: argparse.Namespace) -> None:
    mix_path = args.mix.resolve()
    catalog_path = args.catalog.resolve()
    output = args.output.resolve()
    settings = _load_settings(args.settings.resolve() if args.settings else None)

    with mix_path.open(encoding="utf-8") as f:
        document = yaml.safe_load(f)
    if not isinstance(document, dict) or not isinstance(document.get("mix"), list):
        raise PreparationError("Mix YAML must contain a top-level 'mix' list")
    catalog = _load_catalog(catalog_path)
    catalog_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in catalog:
        catalog_by_bucket[row["bucket"]].append(row)

    normalized_mix: list[dict[str, Any]] = []
    normalized_paths: list[dict[str, Any]] = []
    catalog_matches: list[dict[str, Any]] = []
    direct_patterns: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    target_total = int(settings["target_uint32_values"])
    effective_total = 0.0

    for mix_index, mix in enumerate(document["mix"]):
        mix_name = str(mix["name"])
        mix_weight = float(mix["weight"])
        categories = mix.get("categories")
        if not isinstance(categories, list) or not categories:
            failures.append(
                {
                    "mix_name": mix_name,
                    "category_name": "",
                    "path": "",
                    "reason": "no categories",
                }
            )
            continue
        category_weight_sum = sum(float(category["weight"]) for category in categories)
        if not math.isclose(category_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
            failures.append(
                {
                    "mix_name": mix_name,
                    "category_name": "",
                    "path": "",
                    "reason": f"category weights sum to {category_weight_sum}",
                }
            )

        for category_index, category in enumerate(categories):
            category_name = str(category["name"])
            category_weight = float(category["weight"])
            effective_weight = mix_weight * category_weight
            effective_total += effective_weight
            active = effective_weight > 0
            leaf_id = f"{mix_index:03d}:{category_index:02d}"
            normalized_mix.append(
                {
                    "leaf_id": leaf_id,
                    "mix_index": mix_index,
                    "mix_name": mix_name,
                    "mix_weight": f"{mix_weight:.16g}",
                    "category_index": category_index,
                    "category_name": category_name,
                    "category_weight": f"{category_weight:.16g}",
                    "effective_weight": f"{effective_weight:.16g}",
                    "target_uint32_values": round(target_total * effective_weight),
                    "repetition_factor_raw": category.get("repetition_factor", ""),
                    "active": str(active).lower(),
                }
            )

            seen_paths: Counter[str] = Counter()
            for path_index, yaml_path_value in enumerate(category.get("paths", [])):
                yaml_path = str(yaml_path_value)
                if any(ord(char) < 32 for char in yaml_path):
                    failures.append(
                        {
                            "mix_name": mix_name,
                            "category_name": category_name,
                            "path": repr(yaml_path),
                            "reason": "control character in YAML path",
                        }
                    )
                    continue
                seen_paths[yaml_path] += 1
                drop_tcl_duplicate = (
                    mix_name == "the-stack-v2:Tcl"
                    and category_name == "high"
                    and "/quality_p95/" in yaml_path
                    and seen_paths[yaml_path] > 1
                )
                if seen_paths[yaml_path] > 1:
                    duplicates.append(
                        {
                            "leaf_id": leaf_id,
                            "mix_name": mix_name,
                            "category_name": category_name,
                            "path": yaml_path,
                            "occurrence": seen_paths[yaml_path],
                            "action": "dropped" if drop_tcl_duplicate else "retained",
                        }
                    )
                if drop_tcl_duplicate:
                    corrections.append(
                        {
                            "type": "drop_duplicate",
                            "mix_name": mix_name,
                            "category_name": category_name,
                            "path": yaml_path,
                            "reason": "explicitly approved duplicate Tcl quality_p95 correction",
                        }
                    )
                    continue

                path_id = f"{leaf_id}:{path_index:03d}"
                resolution_route = (
                    "catalog"
                    if yaml_path.startswith("dolma3p5_pool/")
                    else "direct_s3" if yaml_path.startswith("preprocessed/") else "unsupported"
                )
                normalized_paths.append(
                    {
                        "path_id": path_id,
                        "leaf_id": leaf_id,
                        "mix_index": mix_index,
                        "mix_name": mix_name,
                        "category_index": category_index,
                        "category_name": category_name,
                        "active": str(active).lower(),
                        "yaml_path": yaml_path,
                    }
                )
                if resolution_route == "catalog":
                    pattern = _catalog_pattern(yaml_path)
                    matched = [row for row in catalog if _matches_key(row["key"], pattern)]
                    if not matched:
                        failures.append(
                            {
                                "mix_name": mix_name,
                                "category_name": category_name,
                                "path": yaml_path,
                                "reason": "no matching object in reference catalog",
                            }
                        )
                    for match in matched:
                        catalog_matches.append(
                            {
                                "path_id": path_id,
                                "leaf_id": leaf_id,
                                "mix_index": mix_index,
                                "mix_name": mix_name,
                                "category_index": category_index,
                                "category_name": category_name,
                                "active": str(active).lower(),
                                "yaml_path": yaml_path,
                                "bucket": match["bucket"],
                                "key": match["key"],
                                "catalog_line": match["catalog_line"],
                            }
                        )
                elif resolution_route == "direct_s3":
                    bucket, pattern = _direct_s3_pattern(yaml_path, str(settings["direct_s3_bucket"]))
                    direct_patterns.append(
                        {
                            "path_id": path_id,
                            "leaf_id": leaf_id,
                            "mix_index": mix_index,
                            "mix_name": mix_name,
                            "category_index": category_index,
                            "category_name": category_name,
                            "active": str(active).lower(),
                            "yaml_path": yaml_path,
                            "bucket": bucket,
                            "key_pattern": pattern,
                            "listing_prefix": _literal_prefix(pattern),
                        }
                    )
                else:
                    failures.append(
                        {
                            "mix_name": mix_name,
                            "category_name": category_name,
                            "path": yaml_path,
                            "reason": "unsupported path root",
                        }
                    )

    mix_weight_sum = sum(float(mix["weight"]) for mix in document["mix"])
    if not math.isclose(mix_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        failures.append(
            {
                "mix_name": "",
                "category_name": "",
                "path": "",
                "reason": f"mix weights sum to {mix_weight_sum}",
            }
        )
    if not math.isclose(effective_total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        failures.append(
            {
                "mix_name": "",
                "category_name": "",
                "path": "",
                "reason": f"effective weights sum to {effective_total}",
            }
        )

    listing_groups: dict[tuple[str, str], dict[str, Any]] = {}
    by_leaf_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in catalog_matches:
        by_leaf_bucket[(match["leaf_id"], match["bucket"])].append(match)
    for (leaf_id, bucket), matches in by_leaf_bucket.items():
        prefix = _common_directory_prefix([row["key"] for row in matches])
        prefix_depth = len([part for part in prefix.split("/") if part])
        if prefix_depth < int(settings["minimum_catalog_prefix_components"]):
            # Shallow common prefixes are split by YAML expression rather than
            # risking a broad S3 namespace listing.
            by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for match in matches:
                by_path[match["path_id"]].append(match)
            subgroups = list(by_path.values())
        else:
            subgroups = [matches]
        for subgroup in subgroups:
            subgroup_prefix = _common_directory_prefix([row["key"] for row in subgroup])
            required = {(row["bucket"], row["key"]) for row in subgroup}
            estimated = sum(1 for row in catalog_by_bucket[bucket] if row["key"].startswith(subgroup_prefix))
            overfetch = estimated / max(1, len(required))
            if (
                estimated > int(settings["max_listing_catalog_objects"])
                or overfetch > float(settings["max_listing_overfetch_ratio"])
            ) and len(subgroup) > 1:
                final_groups = [[row] for row in subgroup]
            else:
                final_groups = [subgroup]
            for final_group in final_groups:
                final_prefix = _common_directory_prefix([row["key"] for row in final_group])
                key = (bucket, final_prefix)
                entry = listing_groups.setdefault(
                    key,
                    {
                        "bucket": bucket,
                        "listing_prefix": final_prefix,
                        "leaf_ids": set(),
                    },
                )
                entry["leaf_ids"].update(row["leaf_id"] for row in final_group)

    for row in direct_patterns:
        key = (row["bucket"], row["listing_prefix"])
        entry = listing_groups.setdefault(
            key,
            {
                "bucket": row["bucket"],
                "listing_prefix": row["listing_prefix"],
                "leaf_ids": set(),
            },
        )
        entry["leaf_ids"].add(row["leaf_id"])

    listing_plan: list[dict[str, Any]] = []
    for listing_id, ((bucket, prefix), entry) in enumerate(sorted(listing_groups.items())):
        estimated = sum(1 for row in catalog_by_bucket[bucket] if row["key"].startswith(prefix))
        required = {
            (row["bucket"], row["key"])
            for row in catalog_matches
            if row["bucket"] == bucket and row["key"].startswith(prefix)
        }
        listing_plan.append(
            {
                "listing_id": listing_id,
                "bucket": bucket,
                "listing_prefix": prefix,
                "leaf_ids": ";".join(sorted(entry["leaf_ids"])),
                "required_catalog_npy_count": len(required),
                "estimated_catalog_npy_count": estimated,
                "estimated_overfetch_ratio": f"{estimated / max(1, len(required)):.6f}",
            }
        )

    build_id = _build_id(mix_path, catalog_path)
    manifest = {
        "schema_version": 1,
        "build_id": build_id,
        "created_at": _utc_now(),
        "mix_path": str(mix_path),
        "mix_sha256": _sha256(mix_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "settings": settings,
        "counts": {
            "mix_entries": len(document["mix"]),
            "category_entries": len(normalized_mix),
            "active_categories": sum(row["active"] == "true" for row in normalized_mix),
            "normalized_paths": len(normalized_paths),
            "catalog_matches": len(catalog_matches),
            "direct_patterns": len(direct_patterns),
            "listing_commands": len(listing_plan),
            "corrections": len(corrections),
            "failures": len(failures),
        },
    }
    _reset_preparation_build(output)
    plan_root = output / "01-plan"
    plan_root.mkdir(exist_ok=False)
    phase = plan_root / "resolution"
    phase.mkdir(exist_ok=False)
    _write_json(output / "build.json", manifest)
    _write_csv(phase / "normalized-mix.csv", normalized_mix, list(normalized_mix[0]))
    _write_csv(phase / "normalized-paths.csv", normalized_paths, list(normalized_paths[0]))
    _write_csv(
        phase / "catalog-matches.csv",
        catalog_matches,
        [
            "path_id",
            "leaf_id",
            "mix_index",
            "mix_name",
            "category_index",
            "category_name",
            "active",
            "yaml_path",
            "bucket",
            "key",
            "catalog_line",
        ],
    )
    _write_csv(
        phase / "direct-s3-patterns.csv",
        direct_patterns,
        [
            "path_id",
            "leaf_id",
            "mix_index",
            "mix_name",
            "category_index",
            "category_name",
            "active",
            "yaml_path",
            "bucket",
            "key_pattern",
            "listing_prefix",
        ],
    )
    _write_csv(
        phase / "listing-plan.csv",
        listing_plan,
        list(listing_plan[0]) if listing_plan else [],
    )
    _write_csv(
        phase / "corrections.csv",
        corrections,
        ["type", "mix_name", "category_name", "path", "reason"],
    )
    _write_csv(
        phase / "duplicate-paths.csv",
        duplicates,
        ["leaf_id", "mix_name", "category_name", "path", "occurrence", "action"],
    )
    _write_csv(
        phase / "resolution-failures.csv",
        failures,
        ["mix_name", "category_name", "path", "reason"],
    )

    commands = [
        "ls --etag --storage-class " + shlex.quote("s3://" + row["bucket"] + "/" + row["listing_prefix"] + "*")
        for row in listing_plan
    ]
    _write_text(
        phase / "bulk-listing-commands.txt",
        "\n".join(commands) + ("\n" if commands else ""),
    )
    _render_plan_report(
        phase,
        normalized_mix=normalized_mix,
        normalized_paths=normalized_paths,
        catalog_matches=catalog_matches,
        direct_patterns=direct_patterns,
    )

    catalog_summary = _count_label(len(catalog_matches), "catalog NPY match", "catalog NPY matches")
    pattern_summary = _count_label(len(direct_patterns), "direct S3 pattern")
    correction_summary = _count_label(len(corrections), "correction")
    failure_summary = _count_label(len(failures), "blocking failure")
    print(f"Plan summary: {catalog_summary}, {pattern_summary}, " f"{correction_summary}, {failure_summary}")
    if failures:
        raise PreparationError(
            f"Plan contains {len(failures)} validation failure(s); inspect {phase / 'resolution-failures.csv'}"
        )
    print(f"Created preparation plan: {output}")


def _load_build(build: Path) -> dict[str, Any]:
    build = build.resolve()
    with (build / "build.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    mix_path = Path(manifest["mix_path"])
    catalog_path = Path(manifest["catalog_path"])
    if _sha256(mix_path) != manifest["mix_sha256"]:
        raise PreparationError(f"Mix YAML changed since plan creation: {mix_path}")
    if _sha256(catalog_path) != manifest["catalog_sha256"]:
        raise PreparationError(f"Catalog changed since plan creation: {catalog_path}")
    settings = dict(DEFAULT_SETTINGS)
    settings.update(manifest.get("settings", {}))
    manifest["settings"] = settings
    return manifest


def _list_prefix(client: Any, bucket: str, prefix: str) -> list[S3Object]:
    objects: list[S3Object] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            objects.append(
                S3Object(
                    bucket=bucket,
                    key=item["Key"],
                    size_bytes=int(item["Size"]),
                    etag=str(item.get("ETag", "")).strip('"'),
                    last_modified=str(item.get("LastModified", "")),
                    storage_class=str(item.get("StorageClass", "")),
                    source="listing",
                )
            )
    return objects


def _head_object(client: Any, bucket: str, key: str) -> S3Object:
    response = client.head_object(Bucket=bucket, Key=key)
    return S3Object(
        bucket=bucket,
        key=key,
        size_bytes=int(response["ContentLength"]),
        etag=str(response.get("ETag", "")).strip('"'),
        last_modified=str(response.get("LastModified", "")),
        storage_class=str(response.get("StorageClass", "")),
        source="head",
    )


def _inventory_status(step: int, message: str) -> None:
    print(f"[inventory {step}/4] {message}", flush=True)


def _count_newlines(path: Path, offset: int) -> tuple[int, int]:
    count = 0
    with path.open("rb") as f:
        f.seek(offset)
        while chunk := f.read(1024 * 1024):
            count += chunk.count(b"\n")
        return f.tell(), count


def _run_s5cmd_inventory(
    command: Sequence[str],
    raw_output: Path,
    environment: dict[str, str],
    status: Callable[[str], None],
    status_interval_seconds: float = 10.0,
) -> S5cmdRunResult:
    stderr_path = raw_output.with_name("collector-stderr.txt")
    started = time.monotonic()
    output_records = 0
    output_offset = 0
    with raw_output.open("x") as stdout, stderr_path.open("x") as stderr:
        process = subprocess.Popen(
            list(command),
            text=True,
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        while True:
            try:
                returncode = process.wait(timeout=status_interval_seconds)
                break
            except subprocess.TimeoutExpired:
                output_offset, new_records = _count_newlines(raw_output, output_offset)
                output_records += new_records
                status(
                    f"Running: {output_records:,} JSON records received, "
                    f"{time.monotonic() - started:,.0f}s elapsed"
                )
    output_offset, new_records = _count_newlines(raw_output, output_offset)
    output_records += new_records
    del output_offset
    stderr_text = stderr_path.read_text()
    stderr_path.unlink()
    return S5cmdRunResult(
        returncode=returncode,
        stderr=stderr_text,
        output_records=output_records,
        elapsed_seconds=time.monotonic() - started,
    )


def collect_inventory(args: argparse.Namespace) -> None:
    build = args.build.resolve()
    manifest = _load_build(build)
    region = normalize_region(args.region)
    if shutil.which("s5cmd") is None:
        raise PreparationError("s5cmd is required for inventory collection and was not found on PATH")
    phase = _reset_plan_stage(build, "inventory", "execution")
    listing_plan = _read_csv(build / "01-plan/resolution/listing-plan.csv")

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    client = session.client("s3", region_name=region)
    max_workers = args.max_workers or int(manifest["settings"]["inventory_max_workers"])
    listed: dict[tuple[str, str], S3Object] = {}
    errors: list[dict[str, str]] = []
    raw_output = phase / "raw-listings.jsonl"
    commands_path = build / "01-plan/resolution/bulk-listing-commands.txt"
    command = [
        "s5cmd",
        "--json",
        "--numworkers",
        str(manifest["settings"]["s5cmd_numworkers"]),
        "--retry-count",
        str(manifest["settings"]["s5cmd_retry_count"]),
        "run",
        str(commands_path),
    ]
    environment = os.environ.copy()
    if args.profile:
        environment["AWS_PROFILE"] = args.profile
    environment["AWS_REGION"] = region
    environment["AWS_DEFAULT_REGION"] = region
    _inventory_status(
        1,
        f"Bulk listing: {len(listing_plan):,} commands, "
        f"{int(manifest['settings']['s5cmd_numworkers']):,} s5cmd workers, "
        f"{int(manifest['settings']['s5cmd_retry_count']):,} retries",
    )
    result = _run_s5cmd_inventory(
        command,
        raw_output,
        environment,
        lambda message: _inventory_status(1, message),
    )
    _write_json(
        phase / "collector.json",
        {
            "collector": "s5cmd",
            "command": command,
            "command_count": len(listing_plan),
            "output_records": result.output_records,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "raw_output": raw_output.name,
            "read_only": True,
        },
    )
    listed, parse_errors = _parse_s5cmd_jsonl(raw_output)
    errors.extend(
        {
            "operation": "parse",
            "bucket": "",
            "key": row["line"],
            "error": row["error"],
        }
        for row in parse_errors
    )
    if listing_plan and not listed and not errors:
        errors.append(
            {
                "operation": "parse",
                "bucket": "",
                "key": str(raw_output),
                "error": "collector returned no sized objects",
            }
        )
    if result.returncode != 0 or errors:
        _write_csv(
            phase / "inventory-errors.csv",
            errors,
            ["operation", "bucket", "key", "error"],
        )
        if result.stderr:
            _write_text(phase / "collector-error.txt", result.stderr)
        _inventory_status(
            1,
            f"FAILED: s5cmd exit {result.returncode}; {len(errors):,} listing errors. "
            f"Inspect {phase / 'inventory-errors.csv'} and {raw_output}",
        )
        raise PreparationError(f"S3 listing failed; inspect {phase / 'inventory-errors.csv'}")
    _inventory_status(
        1,
        f"Complete: {len(listed):,} unique objects parsed in " f"{result.elapsed_seconds:,.1f}s",
    )
    summary = _finalize_inventory(
        build,
        phase,
        listed,
        client=client,
        max_workers=max_workers,
        status=_inventory_status,
    )
    _inventory_status(
        4,
        f"PASS: {_human_token_count(summary['source_uint32_values'])} source "
        f"→ {_human_token_count(summary['target_uint32_values'])} target; "
        f"{summary['source_family_count']:,} source families, "
        f"{summary['subcategory_count']:,} subcategories, "
        f"{summary['category_count']:,} categories, "
        f"{summary['lower_group_count']:,} lower groups.",
    )


def _walk_json(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
    elif isinstance(value, str) and value.startswith("{"):
        try:
            yield from _walk_json(json.loads(value))
        except json.JSONDecodeError:
            return


def _first_value(nodes: Sequence[dict[str, Any]], names: set[str]) -> Any:
    normalized = {re.sub(r"[^a-z0-9]", "", name.lower()) for name in names}
    for node in nodes:
        for key, value in node.items():
            if re.sub(r"[^a-z0-9]", "", key.lower()) in normalized and value not in {
                None,
                "",
            }:
                return value
    return None


def _parse_s5cmd_jsonl(
    path: Path,
) -> tuple[dict[tuple[str, str], S3Object], list[dict[str, str]]]:
    objects: dict[tuple[str, str], S3Object] = {}
    errors: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": str(line_number), "error": f"invalid JSON: {exc}"})
                continue
            if record.get("error"):
                errors.append({"line": str(line_number), "error": str(record["error"])})
                continue
            nodes = list(_walk_json(record))
            uri = _first_value(nodes, {"source", "url", "uri", "path"})
            key_value = _first_value(nodes, {"key", "name"})
            bucket_value = _first_value(nodes, {"bucket"})
            if isinstance(key_value, str) and key_value.startswith("s3://"):
                uri = key_value
            elif not (isinstance(uri, str) and uri.startswith("s3://")) and bucket_value and key_value:
                uri = f"s3://{bucket_value}/{key_value}"
            if not isinstance(uri, str) or not uri.startswith("s3://") or any(c in uri for c in "*?["):
                continue
            size = _first_value(nodes, {"size", "size_bytes", "content_length", "contentlength"})
            try:
                size_int = int(size)
            except (TypeError, ValueError):
                continue
            parsed = urlparse(uri)
            key = parsed.path.lstrip("/")
            objects[(parsed.netloc, key)] = S3Object(
                bucket=parsed.netloc,
                key=key,
                size_bytes=size_int,
                etag=str(_first_value(nodes, {"etag", "e_tag"}) or "").strip('"'),
                last_modified=str(_first_value(nodes, {"last_modified", "lastmodified", "modtime"}) or ""),
                storage_class=str(_first_value(nodes, {"storage_class", "storageclass"}) or ""),
                source="s5cmd",
            )
    return objects, errors


def _finalize_inventory(
    build: Path,
    phase: Path,
    listed: dict[tuple[str, str], S3Object],
    client: Any,
    max_workers: int,
    status: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    emit = status or (lambda _step, _message: None)
    catalog_matches = _read_csv(build / "01-plan/resolution/catalog-matches.csv")
    direct_patterns = _read_csv(build / "01-plan/resolution/direct-s3-patterns.csv")
    normalized_paths = _read_csv(build / "01-plan/resolution/normalized-paths.csv")
    emit(
        2,
        f"Resolving required objects for {len(normalized_paths):,} YAML paths",
    )
    membership: list[dict[str, Any]] = []
    resolution_failures: list[dict[str, str]] = []

    catalog_buckets_by_path: dict[str, set[str]] = defaultdict(set)
    catalog_lines_by_path: dict[str, list[str]] = defaultdict(list)
    for row in catalog_matches:
        catalog_buckets_by_path[row["path_id"]].add(row["bucket"])
        catalog_lines_by_path[row["path_id"]].append(row["catalog_line"])
    normalized_paths_by_id = {row["path_id"]: row for row in normalized_paths}
    for path_id, buckets in catalog_buckets_by_path.items():
        path = normalized_paths_by_id[path_id]
        key_pattern = _catalog_pattern(path["yaml_path"])
        matched = [
            obj
            for (bucket, key), obj in listed.items()
            if bucket in buckets and key.endswith(".npy") and _matches_key(key, key_pattern)
        ]
        if not matched:
            resolution_failures.append(
                {
                    "path_id": path_id,
                    "yaml_path": path["yaml_path"],
                    "reason": "catalog-resolved YAML pattern returned no NPY objects",
                }
            )
        for obj in sorted(matched, key=lambda item: (item.bucket, item.key)):
            membership.append(
                {
                    **path,
                    "bucket": obj.bucket,
                    "key": obj.key,
                    "catalog_line": ";".join(catalog_lines_by_path[path_id]),
                }
            )
    for pattern in direct_patterns:
        matches = [
            obj
            for (bucket, key), obj in listed.items()
            if bucket == pattern["bucket"] and key.endswith(".npy") and _matches_key(key, pattern["key_pattern"])
        ]
        if not matches:
            resolution_failures.append(
                {
                    "path_id": pattern["path_id"],
                    "yaml_path": pattern["yaml_path"],
                    "reason": "YAML pattern returned no NPY objects",
                }
            )
        for obj in sorted(matches, key=lambda item: item.key):
            membership.append(
                {
                    "path_id": pattern["path_id"],
                    "leaf_id": pattern["leaf_id"],
                    "mix_index": pattern["mix_index"],
                    "mix_name": pattern["mix_name"],
                    "category_index": pattern["category_index"],
                    "category_name": pattern["category_name"],
                    "active": pattern["active"],
                    "yaml_path": pattern["yaml_path"],
                    "bucket": obj.bucket,
                    "key": obj.key,
                    "catalog_line": "",
                }
            )

    required_keys: set[tuple[str, str]] = set()
    for row in membership:
        required_keys.add((row["bucket"], row["key"]))
        required_keys.add((row["bucket"], _pair_metadata_key(row["key"])))

    missing = sorted(required_keys - set(listed))
    emit(
        2,
        f"Resolved {len(membership):,} NPY memberships requiring "
        f"{len(required_keys):,} unique NPY/metadata objects",
    )
    head_errors: list[dict[str, str]] = []
    if missing:
        head_total = len(missing)
        completed_heads = 0
        progress_interval = max(1, math.ceil(head_total / 10))
        emit(
            2,
            f"Checking {head_total:,} objects absent from bulk results with "
            f"{max_workers:,} concurrent HeadObject requests",
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_head_object, client, bucket, key): (bucket, key) for bucket, key in missing}
            for future in as_completed(futures):
                bucket, key = futures[future]
                try:
                    obj = future.result()
                    listed[(bucket, key)] = obj
                except Exception as exc:
                    head_errors.append(
                        {
                            "operation": "head",
                            "bucket": bucket,
                            "key": key,
                            "error": repr(exc),
                        }
                    )
                completed_heads += 1
                if completed_heads == head_total or completed_heads % progress_interval == 0:
                    emit(
                        2,
                        f"Exact checks: {completed_heads:,}/{head_total:,} complete",
                    )
        missing = sorted(required_keys - set(listed))
    else:
        emit(2, "All required objects were present in the bulk results")

    missing_rows = [
        {
            "bucket": bucket,
            "key": key,
            "object_type": "npy" if key.endswith(".npy") else "metadata",
        }
        for bucket, key in missing
    ]
    required_rows: list[dict[str, Any]] = []
    invalid_sizes: list[dict[str, Any]] = []
    for row in membership:
        npy = listed.get((row["bucket"], row["key"]))
        metadata_key = _pair_metadata_key(row["key"])
        metadata = listed.get((row["bucket"], metadata_key))
        if npy is None or metadata is None:
            continue
        if npy.size_bytes <= 0 or npy.size_bytes % UINT32_BYTES:
            invalid_sizes.append(
                {
                    "bucket": npy.bucket,
                    "key": npy.key,
                    "size_bytes": npy.size_bytes,
                    "reason": "non-positive or not divisible by four",
                }
            )
        required_rows.append(
            {
                **row,
                "npy_uri": npy.uri,
                "metadata_uri": metadata.uri,
                "npy_size_bytes": npy.size_bytes,
                "estimated_uint32_values": npy.size_bytes // UINT32_BYTES,
                "npy_etag": npy.etag,
                "npy_last_modified": npy.last_modified,
                "metadata_size_bytes": metadata.size_bytes,
                "metadata_etag": metadata.etag,
                "inventory_source": npy.source,
            }
        )

    all_objects = [
        {
            "bucket": obj.bucket,
            "key": obj.key,
            "size_bytes": obj.size_bytes,
            "etag": obj.etag,
            "last_modified": obj.last_modified,
            "storage_class": obj.storage_class,
            "source": obj.source,
            "required": str((obj.bucket, obj.key) in required_keys).lower(),
        }
        for obj in sorted(listed.values(), key=lambda item: (item.bucket, item.key))
    ]
    required_fields = [
        "path_id",
        "leaf_id",
        "mix_index",
        "mix_name",
        "category_index",
        "category_name",
        "active",
        "yaml_path",
        "bucket",
        "key",
        "npy_uri",
        "metadata_uri",
        "npy_size_bytes",
        "estimated_uint32_values",
        "npy_etag",
        "npy_last_modified",
        "metadata_size_bytes",
        "metadata_etag",
        "inventory_source",
    ]
    _write_csv(
        phase / "normalized-s3-inventory.csv",
        all_objects,
        list(all_objects[0]) if all_objects else [],
    )
    _write_csv(phase / "required-objects.csv", required_rows, required_fields)
    _write_csv(phase / "missing-objects.csv", missing_rows, ["bucket", "key", "object_type"])
    _write_csv(
        phase / "invalid-npy-sizes.csv",
        invalid_sizes,
        ["bucket", "key", "size_bytes", "reason"],
    )
    _write_csv(
        phase / "path-resolution-failures.csv",
        resolution_failures,
        ["path_id", "yaml_path", "reason"],
    )
    _write_csv(phase / "head-errors.csv", head_errors, ["operation", "bucket", "key", "error"])
    original_by_leaf: dict[str, dict[str, int]] = defaultdict(dict)
    for row in required_rows:
        original_by_leaf[row["leaf_id"]][row["npy_uri"]] = int(row["estimated_uint32_values"])
    normalized_mix = _read_csv(build / "01-plan/resolution/normalized-mix.csv")
    original_total = sum(sum(objects.values()) for objects in original_by_leaf.values())
    target_total = sum(int(row["target_uint32_values"]) for row in normalized_mix)
    maximum_expected_upsample_rate = _load_build(build)["settings"].get("maximum_expected_upsample_rate")
    sampling_rate_rows: list[dict[str, Any]] = []
    sampling_rate_limit_failures = 0
    for row in normalized_mix:
        if row["active"] != "true":
            continue
        original = sum(original_by_leaf[row["leaf_id"]].values())
        target = int(row["target_uint32_values"])
        sample_rate = target / original if original else None
        exceeds_limit = (
            maximum_expected_upsample_rate is not None
            and sample_rate is not None
            and sample_rate > float(maximum_expected_upsample_rate)
        )
        sampling_rate_limit_failures += int(exceeds_limit)
        sampling_rate_rows.append(
            {
                "leaf_id": row["leaf_id"],
                "mix_name": row["mix_name"],
                "category_name": row["category_name"],
                "original_uint32_values": original,
                "target_uint32_values": target,
                "sample_rate": f"{sample_rate:.12g}" if sample_rate is not None else "",
                "maximum_expected_upsample_rate": (
                    maximum_expected_upsample_rate if maximum_expected_upsample_rate is not None else ""
                ),
                "status": (
                    "above_expected_maximum"
                    if exceeds_limit
                    else "within_expected_range" if maximum_expected_upsample_rate is not None else "not_checked"
                ),
            }
        )
    sampling_rate_rows.sort(
        key=lambda row: (float(row["sample_rate"] or "inf"), row["leaf_id"]),
        reverse=True,
    )
    _write_csv(
        phase / "sampling-rate-audit.csv",
        sampling_rate_rows,
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "original_uint32_values",
            "target_uint32_values",
            "sample_rate",
            "maximum_expected_upsample_rate",
            "status",
        ],
    )
    _, _, sampling_ratio = _sampling_change(original_total, target_total)
    sampling_rate, _ = _sampling_rate_label(original_total, target_total)
    summary = {
        "created_at": _utc_now(),
        "listed_objects": len(listed),
        "required_membership_rows": len(required_rows),
        "unique_required_objects_including_metadata": len(required_keys),
        "missing_objects": len(missing),
        "path_resolution_failures": len(resolution_failures),
        "invalid_npy_sizes": len(invalid_sizes),
        "head_errors": len(head_errors),
        "sampling_rate_limit_failures": sampling_rate_limit_failures,
        "maximum_expected_upsample_rate": maximum_expected_upsample_rate,
        "source_uint32_values": original_total,
        "original_uint32_values": original_total,
        "target_uint32_values": target_total,
        "token_delta": target_total - original_total,
        "sampling_ratio": sampling_ratio,
        "sampling_rate": sampling_rate,
    }
    validation_details = (
        f"Validation: {len(required_rows):,} NPY memberships, "
        f"{len(missing):,} missing objects, "
        f"{len(resolution_failures):,} unresolved YAML patterns, "
        f"{len(invalid_sizes):,} invalid NPY sizes, {len(head_errors):,} HEAD errors"
    )
    if maximum_expected_upsample_rate is not None:
        validation_details += f", {sampling_rate_limit_failures:,} sampling-rate violations"
    emit(3, validation_details)
    emit(
        3,
        f"Estimated source tokens: {_human_token_count(original_total)} " f"({original_total:,} uint32 values)",
    )
    emit(4, f"Finalizing inventory report and summary in {phase}")
    detail_metadata = _render_inventory_report(
        phase,
        normalized_mix=normalized_mix,
        normalized_paths=normalized_paths,
        required_rows=required_rows,
        all_objects=all_objects,
        missing_rows=missing_rows,
        resolution_failures=resolution_failures,
        invalid_sizes=invalid_sizes,
        sampling_rate_rows=sampling_rate_rows,
    )
    summary.update(detail_metadata)
    _write_json(
        phase / "inventory-summary.json",
        summary,
    )
    if resolution_failures or missing or invalid_sizes or head_errors or sampling_rate_limit_failures:
        emit(4, f"FAILED: inventory validation did not pass. Inspect {phase}")
        raise PreparationError(f"Inventory validation failed; inspect artifacts in {phase}")
    return summary


def _allocate_object_sampling(target: int, sizes: Sequence[int]) -> tuple[list[int], list[int], int]:
    """Allocate every shard the same rate, using partial quotas for the residual.

    A 0.30 rate assigns roughly 30% of every shard, rather than selecting 30%
    of the shard files. Largest-remainder apportionment keeps the aggregate
    target exact while each per-shard quota differs from its ideal by less than
    one uint32 value.
    """

    if target <= 0 or not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Allocation requires a positive target and positive object sizes")
    available = sum(sizes)
    base = target // available
    repetitions = [base for _ in sizes]
    residual = target - base * available
    partial_targets: list[int] = []
    remainders: list[tuple[int, int]] = []
    for index, size in enumerate(sizes):
        partial, remainder = divmod(residual * size, available)
        partial_targets.append(partial)
        remainders.append((remainder, index))
    undistributed = residual - sum(partial_targets)
    for _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[:undistributed]:
        partial_targets[index] += 1

    for index, size in enumerate(sizes):
        if partial_targets[index] == size:
            repetitions[index] += 1
            partial_targets[index] = 0
        elif partial_targets[index] > size:
            raise AssertionError("Proportional partial target exceeds its source")
        residual_allocation = (repetitions[index] - base) * size + partial_targets[index]
        if abs(residual_allocation * available - residual * size) >= available:
            raise AssertionError("Per-shard sampling quota is not proportional")
        if residual * size >= available and residual_allocation <= 0:
            raise AssertionError("A shard with a positive proportional quota was dropped")
    planned = sum(size * repeat + partial for size, repeat, partial in zip(sizes, repetitions, partial_targets))
    if planned != target:
        raise AssertionError(f"Allocation did not preserve target: {planned} != {target}")
    return repetitions, partial_targets, planned


def _execution_unit_sizes(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    def partial_metadata_bytes(row: dict[str, Any]) -> int:
        partial_target = int(row.get("partial_target_uint32_values", 0))
        if not partial_target:
            return 0
        return math.ceil(int(row["metadata_size_bytes"]) * partial_target / int(row["estimated_uint32_values"]))

    input_npy_bytes = sum(int(row["npy_size_bytes"]) for row in rows)
    input_metadata_bytes = sum(int(row["metadata_size_bytes"]) for row in rows)
    output_npy_bytes = sum(
        int(row["npy_size_bytes"]) * int(row["repeat_count"])
        + int(row.get("partial_target_uint32_values", 0)) * UINT32_BYTES
        for row in rows
    )
    estimated_output_metadata_bytes = sum(
        int(row["metadata_size_bytes"]) * int(row["repeat_count"]) + partial_metadata_bytes(row) for row in rows
    )
    estimated_selection_index_bytes = sum(partial_metadata_bytes(row) for row in rows)
    return {
        "input_npy_bytes": input_npy_bytes,
        "input_metadata_bytes": input_metadata_bytes,
        "output_npy_bytes": output_npy_bytes,
        "estimated_output_metadata_bytes": estimated_output_metadata_bytes,
        "estimated_selection_index_bytes": estimated_selection_index_bytes,
        "estimated_peak_local_bytes": input_npy_bytes
        + input_metadata_bytes
        + output_npy_bytes
        + estimated_output_metadata_bytes
        + estimated_selection_index_bytes,
    }


def _partition_object_uses(
    rows: Sequence[dict[str, Any]], max_unit_working_bytes: int
) -> list[list[dict[str, Any]]]:
    """Partition one category into deterministic worker-sized units."""

    if max_unit_working_bytes <= 0:
        raise ValueError("max_unit_working_bytes must be positive")
    positive = [
        dict(row)
        for row in rows
        if int(row["repeat_count"]) > 0 or int(row.get("partial_target_uint32_values", 0)) > 0
    ]
    if not positive:
        raise ValueError("An execution-unit partition requires a positive object use")

    units: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            units.append(current)
            current = []

    for row in sorted(positive, key=lambda item: item["npy_uri"]):
        remaining = int(row["repeat_count"])
        partial_target = int(row.get("partial_target_uint32_values", 0))
        input_bytes = int(row["npy_size_bytes"]) + int(row["metadata_size_bytes"])
        output_bytes_per_repeat = input_bytes
        maximum_repeats_alone = (max_unit_working_bytes - input_bytes) // output_bytes_per_repeat
        if maximum_repeats_alone < 1:
            raise PreparationError(
                "One source object cannot fit in an execution unit with one output copy: "
                f"{row['npy_uri']} requires at least {input_bytes + output_bytes_per_repeat} bytes, "
                f"limit is {max_unit_working_bytes}"
            )

        whole_row = dict(row)
        whole_row["repeat_count"] = remaining
        if _execution_unit_sizes([whole_row])["estimated_peak_local_bytes"] <= max_unit_working_bytes:
            candidate = [*current, whole_row]
            if current and _execution_unit_sizes(candidate)["estimated_peak_local_bytes"] > max_unit_working_bytes:
                flush()
            current.append(whole_row)
            continue

        flush()
        while remaining or partial_target:
            chunk = dict(row)
            chunk["repeat_count"] = 0
            chunk["partial_target_uint32_values"] = 0
            if partial_target:
                chunk["partial_target_uint32_values"] = partial_target
                partial_target = 0
            partial_sizes = _execution_unit_sizes([chunk])
            if partial_sizes["estimated_peak_local_bytes"] > max_unit_working_bytes:
                raise PreparationError(
                    "One source object's partial-document selection cannot fit in "
                    f"an execution unit: {row['npy_uri']} requires at least "
                    f"{partial_sizes['estimated_peak_local_bytes']} bytes, limit is "
                    f"{max_unit_working_bytes}"
                )
            remaining_bytes = max_unit_working_bytes - partial_sizes["estimated_peak_local_bytes"]
            repeat_count = min(
                remaining,
                max(0, remaining_bytes // output_bytes_per_repeat),
            )
            if repeat_count == 0 and not chunk["partial_target_uint32_values"]:
                repeat_count = min(remaining, maximum_repeats_alone)
            chunk["repeat_count"] = repeat_count
            current.append(chunk)
            remaining -= repeat_count
            if remaining or partial_target:
                flush()

    flush()
    return units


def _self_contained_launcher(
    unit_id: str,
    config_name: str,
    config_text: str,
    manifest_name: str,
    manifest_text: str,
) -> str:
    config_payload = base64.b64encode(config_text.encode()).decode("ascii")
    manifest_payload = base64.b64encode(manifest_text.encode()).decode("ascii")
    return f"""#!/usr/bin/env bash
set -euo pipefail

python_bin="${{DOLMA_PYTHON:-$HOME/.venv/bin/python}}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=$(command -v python3.12 || command -v python3 || command -v python)
fi

status_root="${{DOLMA_STATUS_ROOT:-$HOME/dolma3p5-resharding-status}}"
mkdir -p "$status_root"
status_path="$status_root/{unit_id}.status"
log_path="$status_root/{unit_id}.log"
exec > >(tee -a "$log_path") 2>&1

unit_root=$(mktemp -d "${{TMPDIR:-/tmp}}/dolma3p5-{unit_id}.XXXXXX")
finish() {{
  exit_code=$?
  rm -rf -- "$unit_root"
  if [[ $exit_code -eq 0 ]]; then
    printf 'succeeded\n' > "$status_path"
  else
    printf 'failed %s\n' "$exit_code" > "$status_path"
  fi
  trap - EXIT
  exit "$exit_code"
}}
trap finish EXIT
printf 'running\n' > "$status_path"
mkdir -p "$unit_root/config" "$unit_root/manifests"

"$python_bin" - <<'PY'
from dolma.tokenizer.reshard import RESHARDING_MANIFEST_SCHEMA_VERSION

if RESHARDING_MANIFEST_SCHEMA_VERSION != 2:
    raise RuntimeError(
        "Worker Dolma runtime does not match the reviewed manifest-resharding schema"
    )
PY

"$python_bin" - "$unit_root/config/{config_name}" "$unit_root/manifests/{manifest_name}" <<'PY'
import base64
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode("{config_payload}", validate=True))
pathlib.Path(sys.argv[2]).write_bytes(base64.b64decode("{manifest_payload}", validate=True))
PY

"$python_bin" -m dolma.tokenizer.reshard "$unit_root/config/{config_name}"
"""


def _validate_destination_root(destination_root: str) -> str:
    parsed = urlparse(destination_root)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise PreparationError("destination-root must be an s3:// URI")
    prefix = parsed.path.strip("/")
    components = prefix.split("/")
    if len(components) < 2:
        raise PreparationError("destination-root must contain at least two path components below the bucket")
    if any(component in {"", ".", ".."} for component in components):
        raise PreparationError("destination-root cannot contain empty, '.' or '..' components")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", prefix):
        raise PreparationError("destination-root must use only letters, digits, '.', '_', '-', and '/'")
    return f"s3://{parsed.netloc}/{prefix}"


def _source_relative_directory(source_key: str) -> str:
    """Return a source directory relative to its top-level storage prefix."""

    components = tuple(source_key.split("/"))
    if len(components) < 3 or any(component in {"", ".", ".."} for component in components):
        raise PreparationError(f"Source object has an unsafe path: {source_key}")
    directory = components[1:-1]
    return "/".join(directory)


def _source_root_uri(bucket: str, source_key: str) -> str:
    top_level = source_key.split("/", 1)[0]
    if not bucket or not top_level or top_level in {".", ".."}:
        raise PreparationError(f"Cannot derive source root for s3://{bucket}/{source_key}")
    return f"s3://{bucket}/{top_level}"


def _common_path_prefix(paths: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    common: list[str] = []
    for components in zip(*paths):
        if len(set(components)) != 1:
            break
        common.append(components[0])
    return tuple(common)


def _common_path_suffix(
    paths: Sequence[tuple[str, ...]],
    prefix_length: int,
) -> tuple[str, ...]:
    common_reversed: list[str] = []
    available = min(len(path) - prefix_length for path in paths)
    for offset in range(1, available + 1):
        components = {path[-offset] for path in paths}
        if len(components) != 1:
            break
        common_reversed.append(components.pop())
    return tuple(reversed(common_reversed))


def _category_output_directory(
    objects: Sequence[dict[str, str]],
    category_name: str,
) -> str:
    """Build one source-shaped output directory for a YAML category."""

    if not category_name or category_name in {".", ".."} or "/" in category_name:
        raise PreparationError(f"Category name cannot be used in an output path: {category_name!r}")
    directories = sorted(
        {
            tuple(_source_relative_directory(row["key"]).split("/"))
            for row in objects
        }
    )
    if len(directories) == 1:
        return "/".join(directories[0])
    common_prefix = _common_path_prefix(directories)
    if not common_prefix:
        raise PreparationError(
            f"Category {category_name!r} has source paths without a common output prefix"
        )
    common_suffix = _common_path_suffix(directories, len(common_prefix))
    return "/".join((*common_prefix, category_name, *common_suffix))


def propose_configs(args: argparse.Namespace) -> None:
    build = args.build.resolve()
    manifest = _load_build(build)
    inventory_phase = build / "01-plan/inventory"
    if not (inventory_phase / "inventory-summary.json").is_file():
        raise PreparationError("Inventory is missing; rerun scripts/dolma3p5_resharding/plan.py")
    with (inventory_phase / "inventory-summary.json").open() as f:
        inventory_summary = json.load(f)
    blocking = sum(
        int(inventory_summary.get(name, 0))
        for name in (
            "missing_objects",
            "path_resolution_failures",
            "direct_resolution_failures",
            "invalid_npy_sizes",
            "head_errors",
            "sampling_rate_limit_failures",
        )
    )
    if blocking:
        raise PreparationError(
            "Inventory contains blocking validation failures; inspect "
            f"{inventory_phase / 'inventory-summary.json'} and "
            f"{inventory_phase / 'sampling-rate-audit.csv'}"
        )

    phase = _reset_plan_stage(build, "execution")
    configs_dir = phase / "config"
    manifests_dir = phase / "manifests"
    plots_dir = phase / "plots"
    launcher_dir = phase / "launcher-scripts"
    configs_dir.mkdir()
    manifests_dir.mkdir()
    plots_dir.mkdir()
    launcher_dir.mkdir()
    destination_root = _validate_destination_root(args.destination_root)
    local_temp_root = Path(args.local_temp_root)
    if not local_temp_root.is_absolute():
        raise PreparationError("local-temp-root must be an absolute path")
    max_unit_working_bytes = int(args.max_unit_working_bytes)
    if max_unit_working_bytes <= 0:
        raise PreparationError("max-unit-working-bytes must be positive")

    normalized_mix = _read_csv(build / "01-plan/resolution/normalized-mix.csv")
    inventory = _read_csv(inventory_phase / "required-objects.csv")
    by_leaf: dict[str, dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in inventory:
        if row["active"] != "true":
            continue
        object_id = (row["bucket"], row["key"])
        by_leaf[row["leaf_id"]][object_id] = row
        memberships[object_id].add(row["leaf_id"])
    source_roots = sorted(
        {
            _source_root_uri(row["bucket"], row["key"])
            for row in inventory
            if row["active"] == "true"
        }
    )
    overlaps = [
        {"bucket": bucket, "key": key, "leaf_ids": ";".join(sorted(leaves))}
        for (bucket, key), leaves in memberships.items()
        if len(leaves) > 1
    ]
    _write_csv(phase / "cross-leaf-overlaps.csv", overlaps, ["bucket", "key", "leaf_ids"])
    if overlaps:
        raise PreparationError(
            f"Found {len(overlaps)} exact NPY object(s) assigned to multiple active categories; inspect cross-leaf-overlaps.csv"
        )

    allocation_rows: list[dict[str, Any]] = []
    object_use_rows: list[dict[str, Any]] = []
    config_index: list[dict[str, Any]] = []
    category_execution_rows: list[dict[str, Any]] = []
    local_unit_commands: list[str] = []
    next_execution_unit_index = 0

    settings = manifest["settings"]
    total_planned = 0
    manifest_fields = [
        "npy_uri",
        "metadata_uri",
        "repeat_count",
        "partial_target_uint32_values",
        "selection_seed",
        "selection_algorithm",
        "npy_size_bytes",
        "estimated_uint32_values",
        "npy_etag",
        "metadata_size_bytes",
        "metadata_etag",
        "leaf_id",
        "mix_name",
        "category_name",
        "source_directory",
    ]
    active_leaves = sorted(
        (row for row in normalized_mix if row["active"] == "true"),
        key=lambda row: (int(row["mix_index"]), int(row["category_index"])),
    )
    for leaf in active_leaves:
        mix_index = int(leaf["mix_index"])
        category_index = int(leaf["category_index"])
        mix_name = leaf["mix_name"]
        category_name = leaf["category_name"]
        objects = sorted(
            by_leaf.get(leaf["leaf_id"], {}).values(),
            key=lambda row: (row["bucket"], row["key"]),
        )
        if not objects:
            raise PreparationError(f"Active category has no inventoried objects: {leaf['leaf_id']}")
        sizes = [int(row["estimated_uint32_values"]) for row in objects]
        target = int(leaf["target_uint32_values"])
        repetitions, partial_targets, planned = _allocate_object_sampling(target, sizes)
        available = sum(sizes)
        total_planned += planned
        allocation_rows.append(
            {
                "leaf_id": leaf["leaf_id"],
                "mix_index": mix_index,
                "mix_name": mix_name,
                "category_index": category_index,
                "category_name": category_name,
                "effective_weight": leaf["effective_weight"],
                "target_uint32_values": target,
                "available_uint32_values": available,
                "planned_uint32_values": planned,
                "token_change_from_original": planned - available,
                "token_change_percent_from_original": f"{(planned - available) / available:.12g}",
                "ideal_sample_rate": f"{target / available:.12g}",
                "effective_sample_rate": f"{planned / available:.12g}",
                "target_residual_uint32_values": planned - target,
                "target_residual_bps_of_total": f"{10_000 * (planned - target) / int(settings['target_uint32_values']):.12g}",
                "target_residual_fraction": f"{(planned - target) / target:.12g}",
                "unique_object_count": len(objects),
                "selected_object_count": sum(
                    repeat > 0 or partial > 0 for repeat, partial in zip(repetitions, partial_targets)
                ),
                "dropped_object_count": sum(
                    repeat == 0 and partial == 0 for repeat, partial in zip(repetitions, partial_targets)
                ),
                "repeated_object_count": sum(
                    size * repeat + partial > size
                    for size, repeat, partial in zip(sizes, repetitions, partial_targets)
                ),
                "partial_object_count": sum(value > 0 for value in partial_targets),
                "total_object_uses": sum(repetitions) + sum(value > 0 for value in partial_targets),
                "minimum_repetition": min(repetitions),
                "maximum_repetition": max(repetitions),
            }
        )
        leaf_object_uses: list[dict[str, Any]] = []
        for obj, repeat_count, partial_target in zip(objects, repetitions, partial_targets):
            selection_seed = int(settings["random_seed"]) + int(
                hashlib.sha256(f"{leaf['leaf_id']}\0{obj['npy_uri']}".encode()).hexdigest()[:16],
                16,
            )
            object_use = {
                "leaf_id": leaf["leaf_id"],
                "mix_index": mix_index,
                "mix_name": mix_name,
                "category_index": category_index,
                "category_name": category_name,
                "source_directory": obj["npy_uri"].rsplit("/", 1)[0],
                "source_layout_prefix": _source_relative_directory(obj["key"]),
                "path_id": obj["path_id"],
                "lower_group": _path_subgroup(obj["yaml_path"]),
                "yaml_path": obj["yaml_path"],
                "npy_uri": obj["npy_uri"],
                "metadata_uri": obj["metadata_uri"],
                "npy_size_bytes": obj["npy_size_bytes"],
                "estimated_uint32_values": obj["estimated_uint32_values"],
                "npy_etag": obj["npy_etag"],
                "metadata_size_bytes": obj["metadata_size_bytes"],
                "metadata_etag": obj["metadata_etag"],
                "repeat_count": repeat_count,
                "partial_target_uint32_values": partial_target,
                "selection_seed": selection_seed,
                "selection_algorithm": DOCUMENT_SELECTION_ALGORITHM,
                "planned_uint32_values": int(obj["estimated_uint32_values"]) * repeat_count + partial_target,
            }
            object_use_rows.append(object_use)
            if repeat_count > 0 or partial_target > 0:
                leaf_object_uses.append(object_use)

        units = _partition_object_uses(leaf_object_uses, max_unit_working_bytes)
        if len(units) > 10**EXECUTION_UNIT_INDEX_WIDTH:
            raise PreparationError(
                f"Category {leaf['leaf_id']} needs {len(units):,} execution units, which exceeds "
                f"the {EXECUTION_UNIT_INDEX_WIDTH}-digit destination counter"
            )
        source_layout_prefix = _category_output_directory(objects, category_name)
        unit_planned_values = [_execution_unit_sizes(unit)["output_npy_bytes"] // UINT32_BYTES for unit in units]
        unit_targets = unit_planned_values

        unit_peak_bytes: list[int] = []
        unit_input_bytes: list[int] = []
        dataset_root = f"{destination_root}/{manifest['build_id']}"
        for unit_index, (unit_rows, unit_target) in enumerate(zip(units, unit_targets)):
            unit_number = unit_index + 1
            if next_execution_unit_index >= 10**EXECUTION_UNIT_INDEX_WIDTH:
                raise PreparationError(
                    f"Execution plan exceeds the {EXECUTION_UNIT_INDEX_WIDTH}-digit unit ID counter"
                )
            unit_id = f"{next_execution_unit_index:0{EXECUTION_UNIT_INDEX_WIDTH}d}"
            next_execution_unit_index += 1
            unit_sizes = _execution_unit_sizes(unit_rows)
            unit_peak_bytes.append(unit_sizes["estimated_peak_local_bytes"])
            unit_input_bytes.append(unit_sizes["input_npy_bytes"] + unit_sizes["input_metadata_bytes"])
            unit_planned = unit_sizes["output_npy_bytes"] // UINT32_BYTES
            unit_max_repeat = max(int(row["repeat_count"]) for row in unit_rows)
            unit_partial_objects = sum(int(row.get("partial_target_uint32_values", 0)) > 0 for row in unit_rows)
            manifest_path = manifests_dir / f"{unit_id}.csv"
            _write_csv(manifest_path, unit_rows, manifest_fields)
            if unit_planned < 10_000_000_000:
                shard_floor = 2
            elif unit_planned < 100_000_000_000:
                shard_floor = 4
            else:
                shard_floor = 8
            max_num_files = max(shard_floor, unit_max_repeat)
            destination_index = f"{unit_index:0{EXECUTION_UNIT_INDEX_WIDTH}d}"
            destination = f"{dataset_root}/{source_layout_prefix}/{destination_index}"
            config = {
                "destination_prefix": destination,
                "source_manifests": [{"manifest": f"../manifests/{manifest_path.name}"}],
                "local_tempdir": str(local_temp_root / manifest["build_id"] / unit_id),
                "max_num_files": max_num_files,
                "max_workers": min(int(settings["max_workers_per_reshard"]), max_num_files),
                "random_seed": int(settings["random_seed"])
                + int(hashlib.sha256(unit_id.encode()).hexdigest()[:16], 16),
                "tokenizer_name_or_path": str(settings["tokenizer_name_or_path"]),
                "allow_existing_destination": False,
            }
            config_path = configs_dir / f"{unit_id}.yaml"
            config_text = yaml.safe_dump(config, sort_keys=False)
            _write_text(config_path, config_text)
            launcher_path = launcher_dir / f"{unit_id}.sh"
            _write_text(
                launcher_path,
                _self_contained_launcher(
                    unit_id=unit_id,
                    config_name=config_path.name,
                    config_text=config_text,
                    manifest_name=manifest_path.name,
                    manifest_text=manifest_path.read_text(),
                ),
            )
            launcher_path.chmod(0o755)
            unit_row = {
                "unit_id": unit_id,
                "leaf_id": leaf["leaf_id"],
                "mix_index": mix_index,
                "mix_name": mix_name,
                "category_index": category_index,
                "category_name": category_name,
                "source_directories": ";".join(
                    sorted({row["source_directory"] for row in unit_rows})
                ),
                "source_directory_count": len({row["source_directory"] for row in unit_rows}),
                "source_layout_prefix": source_layout_prefix,
                "destination_index": destination_index,
                "unit_index": unit_number,
                "unit_count_for_category": len(units),
                "config_path": str(config_path.relative_to(build)),
                "manifest_path": str(manifest_path.relative_to(build)),
                "launcher_path": str(launcher_path.relative_to(build)),
                "destination_prefix": destination,
                "target_uint32_values": unit_target,
                "planned_uint32_values": unit_planned,
                "target_residual_uint32_values": unit_planned - unit_target,
                "category_target_uint32_values": target,
                "category_planned_uint32_values": planned,
                "input_npy_bytes": unit_sizes["input_npy_bytes"],
                "input_metadata_bytes": unit_sizes["input_metadata_bytes"],
                "output_npy_bytes": unit_sizes["output_npy_bytes"],
                "estimated_output_metadata_bytes": unit_sizes["estimated_output_metadata_bytes"],
                "estimated_selection_index_bytes": unit_sizes["estimated_selection_index_bytes"],
                "estimated_peak_local_bytes": unit_sizes["estimated_peak_local_bytes"],
                "max_unit_working_bytes": max_unit_working_bytes,
                "working_budget_utilization": f"{unit_sizes['estimated_peak_local_bytes'] / max_unit_working_bytes:.12g}",
                "unique_object_count": len(unit_rows),
                "partial_object_count": unit_partial_objects,
                "allowed_materialized_target_residual_uint32_values": (
                    math.ceil(unit_target * float(settings["max_materialized_unit_target_residual_fraction"]))
                    if unit_partial_objects
                    else 0
                ),
                "maximum_repetition": unit_max_repeat,
                "max_num_files": max_num_files,
            }
            config_index.append(unit_row)
            local_unit_commands.append(f"python -m dolma.tokenizer.reshard {shlex.quote(str(config_path))}")

        category_execution_rows.append(
            {
                "leaf_id": leaf["leaf_id"],
                "mix_index": mix_index,
                "mix_name": mix_name,
                "category_index": category_index,
                "category_name": category_name,
                "target_uint32_values": target,
                "planned_uint32_values": planned,
                "execution_unit_count": len(units),
                "largest_estimated_peak_local_bytes": max(unit_peak_bytes),
                "total_input_bytes_across_units": sum(unit_input_bytes),
                "max_unit_working_bytes": max_unit_working_bytes,
            }
        )

    allocation_fields = [
        "leaf_id",
        "mix_index",
        "mix_name",
        "category_index",
        "category_name",
        "effective_weight",
        "target_uint32_values",
        "available_uint32_values",
        "planned_uint32_values",
        "token_change_from_original",
        "token_change_percent_from_original",
        "ideal_sample_rate",
        "effective_sample_rate",
        "target_residual_uint32_values",
        "target_residual_bps_of_total",
        "target_residual_fraction",
        "unique_object_count",
        "selected_object_count",
        "dropped_object_count",
        "repeated_object_count",
        "partial_object_count",
        "total_object_uses",
        "minimum_repetition",
        "maximum_repetition",
    ]
    _write_csv(phase / "category-allocation.csv", allocation_rows, allocation_fields)
    _write_csv(phase / "planned-object-uses.csv", object_use_rows, list(object_use_rows[0]))
    _write_csv(phase / "config-index.csv", config_index, list(config_index[0]))
    _write_csv(
        phase / "category-execution-summary.csv",
        category_execution_rows,
        list(category_execution_rows[0]),
    )
    _write_text(
        phase / "execution-units.jsonl",
        "\n".join(json.dumps(row, sort_keys=True) for row in config_index) + "\n",
    )
    _write_text(
        phase / "LOCAL-UNIT-COMMANDS.txt",
        "# INERT REVIEW ARTIFACT. These commands were not run.\n"
        "# Each command is for debugging one unit only; do not run this entire file on one machine.\n"
        + "\n".join(local_unit_commands)
        + "\n",
    )
    _write_json(
        phase / "dataset-layout.json",
        {
            "schema_version": EXECUTION_LAYOUT_SCHEMA_VERSION,
            "build_id": manifest["build_id"],
            "dataset_root": f"{destination_root}/{manifest['build_id']}",
            "layout": DESTINATION_LAYOUT,
            "source_roots": source_roots,
            "execution_unit_index_width": EXECUTION_UNIT_INDEX_WIDTH,
            "execution_unit_id_width": EXECUTION_UNIT_INDEX_WIDTH,
            "category_count": len(category_execution_rows),
            "execution_unit_count": len(config_index),
            "nominal_target_uint32_values": int(settings["target_uint32_values"]),
            "target_uint32_values": sum(int(row["target_uint32_values"]) for row in allocation_rows),
            "planned_uint32_values": total_planned,
            "max_unit_working_bytes": max_unit_working_bytes,
            "destination_prefixes_file": "dataset-prefixes.txt",
        },
    )
    _write_json(
        phase / "runtime-requirements.json",
        {
            "required_python_module": "dolma.tokenizer.reshard",
            "required_python_modules": [
                "dolma.tokenizer.reshard",
                "dolma.tokenizer.document_selection",
            ],
            "required_resharding_manifest_schema_version": 2,
            "document_selection_algorithm": DOCUMENT_SELECTION_ALGORITHM,
            "launcher_runtime_check": True,
        },
    )
    _write_text(
        phase / "dataset-prefixes.txt",
        "\n".join(row["destination_prefix"] for row in config_index) + "\n",
    )

    _validate_proposal(build, phase, config_index, allocation_rows, object_use_rows)
    report_totals = _render_report(
        build,
        phase,
        allocation_rows,
        object_use_rows,
        inventory,
        config_index,
        category_execution_rows,
    )
    inventoried_source = int(
        inventory_summary["source_uint32_values"]
        if "source_uint32_values" in inventory_summary
        else inventory_summary["original_uint32_values"]
    )
    if report_totals["source_uint32_values"] != inventoried_source:
        raise PreparationError(
            "Proposal source total does not match the source inventory: "
            f"{report_totals['source_uint32_values']:,} != {inventoried_source:,}"
        )
    target_residual = total_planned - report_totals["target_uint32_values"]
    source_shards_with_document_selection = sum(
        int(row.get("partial_target_uint32_values", 0)) > 0 for row in object_use_rows
    )
    _write_json(
        phase / "proposal-summary.json",
        {
            "created_at": _utc_now(),
            "build_id": manifest["build_id"],
            "destination_root": destination_root,
            "dataset_root": f"{destination_root}/{manifest['build_id']}",
            "destination_layout": DESTINATION_LAYOUT,
            "source_roots": source_roots,
            "execution_unit_index_width": EXECUTION_UNIT_INDEX_WIDTH,
            "execution_unit_id_width": EXECUTION_UNIT_INDEX_WIDTH,
            "category_count": len(category_execution_rows),
            "execution_unit_count": len(config_index),
            "config_count": len(config_index),
            "max_unit_working_bytes": max_unit_working_bytes,
            "largest_estimated_peak_local_bytes": max(
                int(row["estimated_peak_local_bytes"]) for row in config_index
            ),
            "nominal_target_uint32_values": int(settings["target_uint32_values"]),
            "target_uint32_values": report_totals["target_uint32_values"],
            "source_uint32_values": report_totals["source_uint32_values"],
            "planned_uint32_values": total_planned,
            "token_change_from_source": total_planned - report_totals["source_uint32_values"],
            "target_residual_uint32_values": target_residual,
            "nominal_target_residual_uint32_values": total_planned - int(settings["target_uint32_values"]),
            "source_shards_with_document_selection": source_shards_with_document_selection,
            "document_selection_algorithm": DOCUMENT_SELECTION_ALGORITHM,
            "max_materialized_unit_target_residual_fraction": float(
                settings["max_materialized_unit_target_residual_fraction"]
            ),
            "max_materialized_total_target_residual_fraction": float(
                settings["max_materialized_total_target_residual_fraction"]
            ),
            "materialization_executed": False,
            "report_artifact": "../report.html",
        },
    )
    _combine_plan_reports(build)
    print(
        f"Execution plan ready: {_human_token_count(total_planned)} tokens across "
        f"{len(config_index):,} execution units; "
        f"target residual: {target_residual:,} tokens.\n"
        f"Review: {build / '01-plan/report.html'}"
    )


def _validate_proposal(
    build: Path,
    phase: Path,
    config_index: Sequence[dict[str, Any]],
    allocations: Sequence[dict[str, Any]],
    object_uses: Sequence[dict[str, Any]],
) -> None:
    failures: list[dict[str, str]] = []
    destinations = [row["destination_prefix"] for row in config_index]
    for destination, count in Counter(destinations).items():
        if count != 1:
            failures.append({"check": "unique_destination", "detail": destination})
    unit_ids = [row["unit_id"] for row in config_index]
    for unit_id, count in Counter(unit_ids).items():
        if count != 1:
            failures.append({"check": "unique_unit_id", "detail": unit_id})
        if not re.fullmatch(rf"[0-9]{{{EXECUTION_UNIT_INDEX_WIDTH}}}", unit_id):
            failures.append({"check": "numeric_unit_id", "detail": unit_id})
    expected_unit_ids = [f"{index:0{EXECUTION_UNIT_INDEX_WIDTH}d}" for index in range(len(config_index))]
    if unit_ids != expected_unit_ids:
        failures.append(
            {
                "check": "sequential_unit_ids",
                "detail": f"expected {len(expected_unit_ids):,} globally sequential IDs",
            }
        )
    for row in config_index:
        destination_index = row.get("destination_index", "")
        if not re.fullmatch(rf"[0-9]{{{EXECUTION_UNIT_INDEX_WIDTH}}}", destination_index):
            failures.append(
                {
                    "check": "destination_index",
                    "detail": row["unit_id"],
                }
            )
        expected_destination_suffix = f'/{row["source_layout_prefix"]}/{destination_index}'
        if not row["destination_prefix"].endswith(expected_destination_suffix):
            failures.append(
                {
                    "check": "category_output_destination",
                    "detail": row["destination_prefix"],
                }
            )
        config_path = build / row["config_path"]
        with config_path.open() as f:
            config = yaml.safe_load(f)
        if config.get("destination_prefix") != row["destination_prefix"]:
            failures.append(
                {
                    "check": "config_destination_matches_index",
                    "detail": str(config_path),
                }
            )
        if config.get("allow_existing_destination") is not False:
            failures.append({"check": "no_existing_destination", "detail": str(config_path)})
        manifest_path = config_path.parent / config["source_manifests"][0]["manifest"]
        if not manifest_path.resolve().is_file():
            failures.append({"check": "manifest_exists", "detail": str(manifest_path)})
        else:
            manifest_rows = _read_csv(manifest_path.resolve())
            manifest_source_directories = sorted(
                {manifest_row["npy_uri"].rsplit("/", 1)[0] for manifest_row in manifest_rows}
            )
            indexed_source_directories = row["source_directories"].split(";")
            if manifest_source_directories != indexed_source_directories or len(
                manifest_source_directories
            ) != int(row["source_directory_count"]):
                failures.append(
                    {
                        "check": "unit_source_directories_match_manifest",
                        "detail": str(manifest_path),
                    }
                )
            for manifest_row in manifest_rows:
                partial_target = int(manifest_row.get("partial_target_uint32_values", 0))
                if partial_target and (
                    manifest_row.get("selection_algorithm") != DOCUMENT_SELECTION_ALGORITHM
                    or not manifest_row.get("selection_seed")
                ):
                    failures.append(
                        {
                            "check": "partial_selection_manifest",
                            "detail": str(manifest_path),
                        }
                    )
        launcher_path = build / row["launcher_path"]
        if not launcher_path.is_file() or not os.access(launcher_path, os.X_OK):
            failures.append({"check": "launcher_is_executable", "detail": str(launcher_path)})
        if int(row["estimated_peak_local_bytes"]) > int(row["max_unit_working_bytes"]):
            failures.append(
                {
                    "check": "unit_within_working_budget",
                    "detail": row["unit_id"],
                }
            )
    active_leaf_ids = {
        row["leaf_id"]
        for row in _read_csv(build / "01-plan/resolution/normalized-mix.csv")
        if row["active"] == "true"
    }
    allocated_leaf_ids = {row["leaf_id"] for row in allocations}
    for leaf_id in sorted(active_leaf_ids - allocated_leaf_ids):
        failures.append({"check": "active_leaf_allocated", "detail": leaf_id})
    units_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in config_index:
        units_by_leaf[row["leaf_id"]].append(row)
    allocation_by_leaf = {row["leaf_id"]: row for row in allocations}
    uses_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in object_uses:
        uses_by_leaf[row["leaf_id"]].append(row)
    for leaf_id in sorted(active_leaf_ids):
        units = units_by_leaf.get(leaf_id, [])
        if not units:
            failures.append({"check": "category_has_execution_unit", "detail": leaf_id})
            continue
        planned = sum(int(row["planned_uint32_values"]) for row in units)
        target = sum(int(row["target_uint32_values"]) for row in units)
        allocation = allocation_by_leaf.get(leaf_id)
        if allocation is None:
            continue
        if planned != int(allocation["planned_uint32_values"]):
            failures.append({"check": "unit_planned_sum", "detail": f"{leaf_id}: {planned}"})
        if target != int(allocation["target_uint32_values"]):
            failures.append({"check": "unit_target_sum", "detail": f"{leaf_id}: {target}"})
        if int(allocation["target_residual_uint32_values"]) != 0:
            failures.append(
                {
                    "check": "exact_proposal_target",
                    "detail": (f"{leaf_id}: " f"{allocation['target_residual_uint32_values']}"),
                }
            )
        object_planned = sum(int(row["planned_uint32_values"]) for row in uses_by_leaf.get(leaf_id, []))
        if object_planned != planned:
            failures.append(
                {
                    "check": "object_planned_sum",
                    "detail": f"{leaf_id}: {object_planned} != {planned}",
                }
            )
    for row in object_uses:
        source = int(row["estimated_uint32_values"])
        repeat_count = int(row["repeat_count"])
        partial_target = int(row.get("partial_target_uint32_values", 0))
        planned = int(row["planned_uint32_values"])
        if not 0 <= partial_target < source:
            failures.append(
                {
                    "check": "valid_partial_target",
                    "detail": f"{row['npy_uri']}: {partial_target} of {source}",
                }
            )
        if planned != source * repeat_count + partial_target:
            failures.append(
                {
                    "check": "object_sampling_arithmetic",
                    "detail": f"{row['npy_uri']}: {planned}",
                }
            )
    planned_uses = {(row["leaf_id"], row["npy_uri"]) for row in object_uses}
    required_uses = {
        (row["leaf_id"], row["npy_uri"])
        for row in _read_csv(build / "01-plan/inventory/required-objects.csv")
        if row["active"] == "true"
    }
    for leaf_id, uri in sorted(required_uses - planned_uses):
        failures.append({"check": "required_object_considered", "detail": f"{leaf_id}: {uri}"})
    _write_csv(phase / "validation-failures.csv", failures, ["check", "detail"])
    _write_json(
        phase / "validation-summary.json",
        {
            "passed": not failures,
            "failure_count": len(failures),
            "config_count": len(config_index),
            "execution_unit_count": len(config_index),
            "active_leaf_count": len(active_leaf_ids),
            "planned_object_rows": len(object_uses),
        },
    )
    if failures:
        raise PreparationError(f"Proposal validation failed; inspect {phase / 'validation-failures.csv'}")


def _filter_execution_units(
    rows: Sequence[dict[str, str]],
    *,
    category: str | None = None,
    unit: str | None = None,
) -> list[dict[str, str]]:
    if category is not None and unit is not None:
        raise PreparationError("Choose either a category or an execution unit, not both")
    if unit is not None:
        selected = [row for row in rows if row["unit_id"] == unit]
        if not selected:
            raise PreparationError(f"Unknown execution-unit ID: {unit}")
        return selected
    if category is None:
        return list(rows)
    selected = [row for row in rows if row["leaf_id"] == category]
    if not selected:
        selected = [row for row in rows if row["mix_name"] == category]
    if not selected:
        selected = [
            row
            for row in rows
            if f'{row["mix_name"]}::{row["category_name"]}' == category
        ]
    if not selected:
        raise PreparationError(
            f"Unknown category selector: {category}. "
            "Use materialize.py --list-categories FILTER to find the exact selector."
        )
    return selected


def _unit_selection_digest(rows: Sequence[dict[str, str]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(row["unit_id"] for row in rows)).encode()
    ).hexdigest()


def preflight_build(args: argparse.Namespace) -> None:
    """Re-inventory selected inputs and verify that selected destinations are empty."""

    build = args.build.resolve()
    manifest = _load_build(build)
    region = normalize_region(args.region)
    proposal_summary = build / "01-plan/execution/proposal-summary.json"
    if not proposal_summary.is_file():
        raise PreparationError("Proposal is missing; run propose first")
    _validate_execution_layout(build)
    phase = _reset_preparation_phase(build, "02-preflight", "03-output-validation")
    all_config_index = _read_csv(build / "01-plan/execution/config-index.csv")
    category = getattr(args, "category", None)
    unit = getattr(args, "unit", None)
    config_index = _filter_execution_units(all_config_index, category=category, unit=unit)
    selection_scope = "unit" if unit is not None else "category" if category is not None else "all"
    selection_value = unit if unit is not None else category if category is not None else "all"

    full_listing_plan = _read_csv(build / "01-plan/resolution/listing-plan.csv")
    approved_inventory = _read_csv(build / "01-plan/inventory/normalized-s3-inventory.csv")
    approved_all = {
        (row["bucket"], row["key"]): row
        for row in approved_inventory
        if row["required"] == "true"
    }
    if selection_scope == "all":
        approved = approved_all
    else:
        selected_identities: set[tuple[str, str]] = set()
        manifest_root = (build / "01-plan/execution/manifests").resolve()
        for config_row in config_index:
            manifest_path = (build / config_row["manifest_path"]).resolve()
            if manifest_path.parent != manifest_root or manifest_path.is_symlink() or not manifest_path.is_file():
                raise PreparationError(f"Unsafe or missing execution manifest: {manifest_path}")
            for manifest_row in _read_csv(manifest_path):
                for field in ("npy_uri", "metadata_uri"):
                    parsed = urlparse(manifest_row[field])
                    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
                        raise PreparationError(f"Unsupported source URI in {manifest_path}: {manifest_row[field]}")
                    selected_identities.add((parsed.netloc, parsed.path.lstrip("/")))
        missing_approved = selected_identities - set(approved_all)
        if missing_approved:
            raise PreparationError(
                f"Selected manifests contain {len(missing_approved):,} objects outside the approved inventory"
            )
        approved = {identity: approved_all[identity] for identity in selected_identities}

    listing_plan = [
        row
        for row in full_listing_plan
        if any(
            bucket == row["bucket"] and key.startswith(row["listing_prefix"])
            for bucket, key in approved
        )
    ]

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    client = session.client("s3", region_name=region)
    max_workers = args.max_workers or int(manifest["settings"]["inventory_max_workers"])
    current: dict[tuple[str, str], S3Object] = {}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_list_prefix, client, row["bucket"], row["listing_prefix"]): row for row in listing_plan
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                for obj in future.result():
                    current[(obj.bucket, obj.key)] = obj
            except Exception as exc:
                errors.append(
                    {
                        "operation": "list",
                        "bucket": row["bucket"],
                        "key": row["listing_prefix"],
                        "error": repr(exc),
                    }
                )

    missing = sorted(set(approved) - set(current))
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_head_object, client, bucket, key): (bucket, key) for bucket, key in missing}
            for future in as_completed(futures):
                bucket, key = futures[future]
                try:
                    obj = future.result()
                    current[(bucket, key)] = obj
                except Exception as exc:
                    errors.append(
                        {
                            "operation": "head",
                            "bucket": bucket,
                            "key": key,
                            "error": repr(exc),
                        }
                    )

    drift_rows: list[dict[str, Any]] = []
    for identity, expected in sorted(approved.items()):
        actual = current.get(identity)
        if actual is None:
            status = "missing"
        elif actual.size_bytes != int(expected["size_bytes"]):
            status = "size_changed"
        elif expected["etag"] and actual.etag != expected["etag"]:
            status = "etag_changed"
        elif expected["last_modified"] and _normalize_timestamp(actual.last_modified) != _normalize_timestamp(
            expected["last_modified"]
        ):
            status = "last_modified_changed"
        else:
            status = "unchanged"
        drift_rows.append(
            {
                "bucket": identity[0],
                "key": identity[1],
                "status": status,
                "expected_size_bytes": expected["size_bytes"],
                "actual_size_bytes": actual.size_bytes if actual else "",
                "expected_etag": expected["etag"],
                "actual_etag": actual.etag if actual else "",
                "expected_last_modified": expected["last_modified"],
                "actual_last_modified": actual.last_modified if actual else "",
            }
        )

    destination_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for row in config_index:
            parsed = urlparse(row["destination_prefix"])
            prefix = parsed.path.lstrip("/").rstrip("/") + "/"
            future = pool.submit(
                client.list_objects_v2,
                Bucket=parsed.netloc,
                Prefix=prefix,
                MaxKeys=1,
            )
            futures[future] = (row, parsed.netloc, prefix)
        for future in as_completed(futures):
            row, bucket, prefix = futures[future]
            try:
                response = future.result()
                contents = response.get("Contents", [])
                destination_rows.append(
                    {
                        "unit_id": row["unit_id"],
                        "leaf_id": row["leaf_id"],
                        "mix_name": row["mix_name"],
                        "category_name": row["category_name"],
                        "source_directories": row["source_directories"],
                        "destination_prefix": row["destination_prefix"],
                        "status": "occupied" if contents else "empty",
                        "first_existing_key": contents[0]["Key"] if contents else "",
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "operation": "destination_list",
                        "bucket": bucket,
                        "key": prefix,
                        "error": repr(exc),
                    }
                )

    drift_fields = [
        "bucket",
        "key",
        "status",
        "expected_size_bytes",
        "actual_size_bytes",
        "expected_etag",
        "actual_etag",
        "expected_last_modified",
        "actual_last_modified",
    ]
    _write_csv(phase / "input-drift.csv", drift_rows, drift_fields)
    _write_csv(
        phase / "destination-status.csv",
        sorted(destination_rows, key=lambda row: row["destination_prefix"]),
        [
            "unit_id",
            "leaf_id",
            "mix_name",
            "category_name",
            "source_directories",
            "destination_prefix",
            "status",
            "first_existing_key",
        ],
    )
    _write_csv(phase / "preflight-errors.csv", errors, ["operation", "bucket", "key", "error"])
    drifted = sum(row["status"] != "unchanged" for row in drift_rows)
    occupied = sum(row["status"] != "empty" for row in destination_rows)
    passed = not errors and not drifted and not occupied and len(destination_rows) == len(config_index)
    summary = {
        "created_at": _utc_now(),
        "selection_scope": selection_scope,
        "selection_value": selection_value,
        "selected_execution_units": len(config_index),
        "total_execution_units": len(all_config_index),
        "selected_unit_ids_sha256": _unit_selection_digest(config_index),
        "approved_input_objects": len(approved),
        "drifted_input_objects": drifted,
        "destinations_checked": len(destination_rows),
        "occupied_destinations": occupied,
        "errors": len(errors),
        "passed": passed,
        "materialization_executed": False,
    }
    _write_json(phase / "preflight-summary.json", summary)
    if not passed:
        raise PreparationError(f"Preflight failed; inspect artifacts in {phase}")
    print(
        f"Preflight passed for {len(config_index):,} selected execution unit(s) "
        f"without materializing data: {phase}"
    )


def verify_output(args: argparse.Namespace) -> None:
    """Verify materialized outputs using only S3 object listings and sizes."""

    build = args.build.resolve()
    manifest = _load_build(build)
    region = normalize_region(args.region)
    _validate_execution_layout(build)
    settings = manifest["settings"]
    config_index = _read_csv(build / "01-plan/execution/config-index.csv")
    if not config_index:
        raise PreparationError("Proposal config index is empty or missing")
    phase = _reset_preparation_phase(build, "03-output-validation")
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    client = session.client("s3", region_name=region)
    max_workers = args.max_workers or int(manifest["settings"]["inventory_max_workers"])
    output_objects: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    unexpected_rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for row in config_index:
            parsed = urlparse(row["destination_prefix"])
            prefix = parsed.path.lstrip("/").rstrip("/") + "/"
            future = pool.submit(_list_prefix, client, parsed.netloc, prefix)
            futures[future] = (row, parsed.netloc, prefix)
        for future in as_completed(futures):
            row, bucket, prefix = futures[future]
            try:
                objects = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "operation": "list_output",
                        "bucket": bucket,
                        "key": prefix,
                        "error": repr(exc),
                    }
                )
                continue
            object_map = {obj.key: obj for obj in objects}
            npys = [obj for obj in objects if obj.key.endswith(".npy")]
            metadata = {obj.key for obj in objects if obj.key.endswith(".csv.gz")}
            unexpected = [obj for obj in objects if not obj.key.endswith((".npy", ".csv.gz"))]
            missing_metadata = [
                _pair_metadata_key(obj.key) for obj in npys if _pair_metadata_key(obj.key) not in metadata
            ]
            orphan_metadata = [key for key in metadata if key[: -len(".csv.gz")] + ".npy" not in object_map]
            invalid_npys = [obj for obj in npys if obj.size_bytes <= 0 or obj.size_bytes % UINT32_BYTES]
            actual = sum(obj.size_bytes // UINT32_BYTES for obj in npys)
            predicted = int(row["planned_uint32_values"])
            allowed_residual = int(row["allowed_materialized_target_residual_uint32_values"])
            status = (
                "passed"
                if npys
                and not missing_metadata
                and not orphan_metadata
                and not invalid_npys
                and not unexpected
                and abs(actual - predicted) <= allowed_residual
                else "failed"
            )
            validation_rows.append(
                {
                    "unit_id": row["unit_id"],
                    "leaf_id": row["leaf_id"],
                    "mix_index": row["mix_index"],
                    "mix_name": row["mix_name"],
                    "category_index": row["category_index"],
                    "category_name": row["category_name"],
                    "destination_prefix": row["destination_prefix"],
                    "target_uint32_values": row["target_uint32_values"],
                    "predicted_uint32_values": predicted,
                    "actual_uint32_values": actual,
                    "actual_minus_predicted": actual - predicted,
                    "allowed_target_residual_uint32_values": allowed_residual,
                    "npy_count": len(npys),
                    "metadata_count": len(metadata),
                    "missing_metadata_count": len(missing_metadata),
                    "orphan_metadata_count": len(orphan_metadata),
                    "invalid_npy_count": len(invalid_npys),
                    "unexpected_object_count": len(unexpected),
                    "status": status,
                }
            )
            for obj in objects:
                output_objects.append(
                    {
                        "unit_id": row["unit_id"],
                        "mix_name": row["mix_name"],
                        "category_name": row["category_name"],
                        "destination_prefix": row["destination_prefix"],
                        "bucket": obj.bucket,
                        "key": obj.key,
                        "size_bytes": obj.size_bytes,
                        "etag": obj.etag,
                        "last_modified": obj.last_modified,
                        "object_type": (
                            "npy"
                            if obj.key.endswith(".npy")
                            else "metadata" if obj.key.endswith(".csv.gz") else "unexpected"
                        ),
                    }
                )
            for key in missing_metadata:
                unexpected_rows.append(
                    {
                        "mix_name": row["mix_name"],
                        "key": key,
                        "reason": "missing metadata partner",
                    }
                )
            for key in orphan_metadata:
                unexpected_rows.append(
                    {
                        "mix_name": row["mix_name"],
                        "key": key,
                        "reason": "orphan metadata",
                    }
                )
            for obj in invalid_npys:
                unexpected_rows.append(
                    {
                        "mix_name": row["mix_name"],
                        "key": obj.key,
                        "reason": "invalid NPY size",
                    }
                )
            for obj in unexpected:
                unexpected_rows.append(
                    {
                        "mix_name": row["mix_name"],
                        "key": obj.key,
                        "reason": "unexpected output object type",
                    }
                )

    validation_rows.sort(key=lambda row: row["unit_id"])
    _write_csv(
        phase / "output-inventory.csv",
        output_objects,
        [
            "unit_id",
            "mix_name",
            "category_name",
            "destination_prefix",
            "bucket",
            "key",
            "size_bytes",
            "etag",
            "last_modified",
            "object_type",
        ],
    )
    validation_fields = [
        "unit_id",
        "leaf_id",
        "mix_index",
        "mix_name",
        "category_index",
        "category_name",
        "destination_prefix",
        "target_uint32_values",
        "predicted_uint32_values",
        "actual_uint32_values",
        "actual_minus_predicted",
        "allowed_target_residual_uint32_values",
        "npy_count",
        "metadata_count",
        "missing_metadata_count",
        "orphan_metadata_count",
        "invalid_npy_count",
        "unexpected_object_count",
        "status",
    ]
    _write_csv(phase / "output-validation.csv", validation_rows, validation_fields)
    _write_csv(
        phase / "output-problems.csv",
        unexpected_rows,
        ["mix_name", "key", "reason"],
    )
    _write_csv(phase / "output-errors.csv", errors, ["operation", "bucket", "key", "error"])
    failed = sum(row["status"] != "passed" for row in validation_rows)
    target_total = sum(int(row["target_uint32_values"]) for row in validation_rows)
    actual_total = sum(int(row["actual_uint32_values"]) for row in validation_rows)
    aggregate_residual = actual_total - target_total
    allowed_aggregate_residual = math.ceil(
        target_total * float(settings["max_materialized_total_target_residual_fraction"])
    )
    aggregate_within_bound = abs(aggregate_residual) <= allowed_aggregate_residual
    passed = not errors and not failed and len(validation_rows) == len(config_index) and aggregate_within_bound
    _write_json(
        phase / "output-summary.json",
        {
            "created_at": _utc_now(),
            "destinations_expected": len(config_index),
            "destinations_checked": len(validation_rows),
            "failed_destinations": failed,
            "errors": len(errors),
            "target_uint32_values": target_total,
            "predicted_uint32_values": sum(int(row["predicted_uint32_values"]) for row in validation_rows),
            "actual_uint32_values": actual_total,
            "materialized_target_residual_uint32_values": aggregate_residual,
            "allowed_materialized_target_residual_uint32_values": (allowed_aggregate_residual),
            "aggregate_target_residual_within_bound": aggregate_within_bound,
            "passed": passed,
        },
    )
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plots.mkdir(exist_ok=False)
    plot_data.mkdir(exist_ok=False)
    _write_csv(
        plot_data / "target-predicted-actual.csv",
        validation_rows,
        [
            "unit_id",
            "leaf_id",
            "mix_index",
            "mix_name",
            "category_name",
            "target_uint32_values",
            "predicted_uint32_values",
            "actual_uint32_values",
            "actual_minus_predicted",
            "allowed_target_residual_uint32_values",
            "status",
        ],
    )
    points = [
        (
            int(row["predicted_uint32_values"]) / 1e9,
            int(row["actual_uint32_values"]) / 1e9,
            row["unit_id"],
        )
        for row in validation_rows
    ]
    _write_text(
        plots / "predicted-vs-actual.svg",
        _svg_scatter(
            "Predicted versus actual materialized token counts",
            points,
            "Predicted (billions of tokens)",
            "Actual (billions of tokens)",
        ),
    )
    failed_rows = [row for row in validation_rows if row["status"] != "passed"]
    status_rows = [
        {
            "state": "passed destinations",
            "count": len(validation_rows) - len(failed_rows),
        },
        {"state": "failed destinations", "count": len(failed_rows)},
        {"state": "request errors", "count": len(errors)},
    ]
    _write_csv(plot_data / "output-status.csv", status_rows, ["state", "count"])
    _write_text(
        plots / "output-status.svg",
        _svg_bar_chart(
            "Materialized output validation",
            [row["state"] for row in status_rows],
            [row["count"] for row in status_rows],
            "destinations",
        ),
    )
    _write_text(
        phase / "report.html",
        '<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 output validation</title></head><body>'
        "<h1>Post-materialization size validation</h1>"
        "<p>Token counts are estimated from output object sizes. No arrays or metadata rows were read.</p>"
        '<img src="plots/predicted-vs-actual.svg" alt="Predicted versus actual sizes">'
        '<img src="plots/output-status.svg" alt="Output validation status">'
        "</body></html>\n",
    )
    if not passed:
        raise PreparationError(f"Output validation failed; inspect artifacts in {phase}")
    print(f"Output validation passed using size-only checks: {phase}")


def _svg_bar_chart(
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    unit: str,
    width: int = 1100,
    value_labels: Sequence[str] | None = None,
    summary: str | None = None,
) -> str:
    if value_labels is not None and len(value_labels) != len(values):
        raise ValueError("value_labels must have the same length as values")
    row_height = 24
    label_x = 12
    margin_left = 390
    margin_right = 240 if value_labels is not None else 140
    height = 70 + row_height * len(labels)
    plot_width = width - margin_left - margin_right
    maximum = max(values, default=1.0) or 1.0
    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 50 + index * row_height
        bar_width = max(0.0, plot_width * value / maximum)
        displayed_value = value_labels[index] if value_labels is not None else f"{value:.4g} {unit}"
        rows.append(
            f'<text x="{label_x}" y="{y + 14}">{html.escape(label[:52])}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="16" />'
            f'<text x="{margin_left + plot_width + 8}" y="{y + 14}">{html.escape(displayed_value)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title)}"><title>{html.escape(title)}</title><style>'
        "text{font:12px sans-serif;fill:#222}rect{fill:#356cb6}"
        ".summary{font-size:14px;font-weight:600}</style>"
        + (f'<text class="summary" x="{label_x}" y="24">{html.escape(summary)}</text>' if summary else "")
        + "".join(rows)
        + "</svg>\n"
    )


def _human_token_count(value: int) -> str:
    for scale, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:,}"


def _sampling_change(original: int, target: int) -> tuple[str, str, float | None]:
    if original <= 0:
        if target <= 0:
            return "no source tokens", "sampling-same", None
        return "no source tokens", "sampling-unknown", None
    ratio = target / original
    if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=0.0005):
        return "1.00× unchanged", "sampling-same", ratio
    direction = "upsample" if ratio > 1 else "downsample"
    css_class = "sampling-up" if ratio > 1 else "sampling-down"
    return f"{_format_multiplier(ratio)} {direction}", css_class, ratio


def _sampling_rate_label(original: int, target: int) -> tuple[str, str]:
    label, css_class, _ = _sampling_change(original, target)
    return label, css_class


def _format_multiplier(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def _apportion_by_size(total: int, sizes: Sequence[int]) -> list[int]:
    denominator = sum(sizes)
    if total <= 0 or denominator <= 0:
        return [0 for _ in sizes]
    apportioned = [total * size // denominator for size in sizes]
    remainder = total - sum(apportioned)
    order = sorted(
        range(len(sizes)),
        key=lambda index: (-(total * sizes[index] % denominator), index),
    )
    for index in order[:remainder]:
        apportioned[index] += 1
    return apportioned


def _comparison_bars(original: int, target: int) -> str:
    maximum = max(original, target, 1)
    original_width = 100 * original / maximum
    target_width = 100 * target / maximum
    return (
        '<span class="comparison-bars" aria-hidden="true">'
        '<span class="comparison-track"><span class="original-fill" '
        f'style="width:{original_width:.8f}%"></span></span>'
        '<span class="comparison-track"><span class="target-fill" '
        f'style="width:{target_width:.8f}%"></span></span></span>'
    )


def _mix_plot_value_label(value: int, total: int) -> str:
    percent = 100 * value / total if total else 0.0
    return f"{percent:.2f}% · {_human_token_count(value)} tokens"


def _summary_metrics(metrics: Sequence[tuple[str, str]]) -> str:
    return (
        '<div class="summary-metrics">'
        + "".join(
            '<div class="summary-metric">'
            f'<span class="summary-label">{html.escape(label)}</span>'
            f'<span class="summary-value">{html.escape(value)}</span>'
            "</div>"
            for label, value in metrics
        )
        + "</div>"
    )


def _svg_scatter(
    title: str,
    points: Sequence[tuple[float, float, str]],
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 760,
) -> str:
    left, top, right, bottom = 90, 40, 40, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max([max(x, y) for x, y, _ in points] or [1.0]) or 1.0
    circles = []
    for x, y, label in points:
        px = left + plot_w * x / maximum
        py = top + plot_h * (1 - y / maximum)
        circles.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4"><title>{html.escape(label)}: target={x:.6g}, proposed={y:.6g}</title></circle>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">
<title>{html.escape(title)}</title><style>text{{font:13px sans-serif;fill:#222}}line{{stroke:#777;stroke-width:1}}circle{{fill:#b34747;fill-opacity:.7}}</style>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke-dasharray="5 4"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>
{"".join(circles)}
<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">{html.escape(x_label)}</text>
<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>
</svg>\n"""


def _path_subgroup(yaml_path: str) -> str:
    parts = [part for part in yaml_path.split("/") if part]
    for index, part in enumerate(parts):
        if part == "allenai" and index:
            return unquote(parts[index - 1])
    candidates = [part for part in parts if "*" not in part and not part.endswith((".npy", ".csv.gz"))]
    return unquote(candidates[-1]) if candidates else yaml_path


def _source_uri_details(source_uris: Iterable[str], empty_message: str) -> str:
    resolved_sources = sorted(set(source_uris)) or [empty_message]
    return "".join(f'<span class="path-detail-uri">{html.escape(uri)}</span>' for uri in resolved_sources)


def _split_mix_name(mix_name: str) -> tuple[str, str]:
    source_family, separator, subcategory = mix_name.partition(":")
    return source_family, subcategory if separator else "default"


def _interactive_report_style() -> str:
    return """
<style>
:root{color-scheme:light dark;--muted:#536965;--surface-hover:#eaf3f1;--surface-selected:#dceeea;--detail:#edf6f4;--track:#d2e1de;--series:#14786f;--original:#71817e;--up:#14786f;--down:#a35f16;--same:#536965;--code:#e2efec}
@media(prefers-color-scheme:dark){:root{--muted:#a7bbb7;--surface-hover:#172522;--surface-selected:#1b312d;--detail:#142420;--track:#2a403c;--series:#5cc8bb;--original:#91a29f;--up:#5cc8bb;--down:#e5a456;--same:#a7bbb7;--code:#1b312d}}
*{box-sizing:border-box}body{font:14px/1.45 system-ui,sans-serif;max-width:1120px;margin:0 auto;padding:32px 24px 72px;color:CanvasText;background:Canvas}h1{margin:0;font-size:26px;line-height:1.2}h2{margin:0;font-size:20px;line-height:1.3;overflow-wrap:anywhere}.chart-total{margin:8px 0 24px;color:var(--muted);font-variant-numeric:tabular-nums}.summary-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px 32px;margin:18px 10px 26px;font-variant-numeric:tabular-nums}.summary-label,.mix-metric-label{display:block;margin-bottom:2px;color:var(--muted)}.summary-value{display:block;font-size:18px;font-weight:500}.mix-chart{display:grid;gap:2px}
.mix-row,.subcategory-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 20px;width:100%;padding:12px 10px;border:0;border-radius:8px;background:transparent;color:inherit;text-align:left;font:inherit;cursor:pointer}.mix-row:hover,.subcategory-row:hover{background:var(--surface-hover)}.mix-row.is-selected,.subcategory-row.is-selected{background:var(--surface-selected)}.mix-name{min-width:0;overflow-wrap:anywhere;font-weight:500}.mix-value{white-space:nowrap;color:var(--muted);font-variant-numeric:tabular-nums}.bar-track{grid-column:1/-1;display:block;height:6px;overflow:hidden;background:var(--track);border-radius:999px}.bar-fill{display:block;height:100%;background:var(--series);border-radius:inherit}
.mix-row.has-metrics,.subcategory-row.has-metrics{grid-template-columns:1fr;gap:8px}.mix-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:5px 24px;color:var(--muted);font-variant-numeric:tabular-nums}.mix-metric-value{color:CanvasText}.subcategory-list{display:grid;gap:2px}.subcategory-detail{padding:20px 10px 26px}.subcategory-detail .detail-head{margin-bottom:18px}
.mix-detail{padding:22px 18px 28px;border-radius:10px;background:var(--detail)}.mix-detail[hidden],.subcategory-detail[hidden]{display:none}.detail-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 20px;align-items:end;margin-bottom:20px}.detail-total{color:var(--muted);font-variant-numeric:tabular-nums;text-align:right}.category-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px 32px}.category{min-width:0}.category-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:baseline}.category-name{font-weight:500;overflow-wrap:anywhere}.sampling{font-variant-numeric:tabular-nums;white-space:nowrap}.sampling-up{color:var(--up)}.sampling-down{color:var(--down)}.sampling-same,.sampling-unknown{color:var(--same)}.category-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px 24px;margin:4px 0 7px;color:var(--muted);font-variant-numeric:tabular-nums}.category-metrics.proposal-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.category-metric{min-width:0}.category-bar{height:4px;overflow:hidden;margin-top:7px;background:var(--track);border-radius:999px}.category-bar-fill{display:block;height:100%;background:var(--series);border-radius:inherit}.comparison-bars{display:grid;grid-template-rows:3px 3px;gap:3px}.comparison-track{display:block;overflow:hidden;background:var(--track);border-radius:999px}.original-fill,.target-fill{display:block;height:100%;border-radius:inherit}.original-fill{background:var(--original)}.target-fill{background:var(--series)}.path-list{display:grid;gap:2px;margin-top:8px}.path-detail summary{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:5px 12px;padding:8px 6px;border-radius:6px;cursor:pointer;list-style-position:inside}.path-detail summary:hover{background:var(--surface-hover)}.path-name{overflow-wrap:anywhere}.path-stat{color:var(--muted);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.path-sampling{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px 24px;color:var(--muted);font-variant-numeric:tabular-nums}.path-metric{min-width:0}.path-sampling .comparison-bars,.path-use-summary{grid-column:1/-1}.path-chevron{display:inline-block;margin-left:7px;transition:transform .12s ease}.path-detail[open] .path-chevron{transform:rotate(90deg)}.path-detail code{display:block;margin:2px 6px 10px;padding:11px 12px;border-radius:6px;background:var(--code);font:12px/1.45 ui-monospace,monospace;overflow-wrap:anywhere}.path-detail-metrics{display:flex;flex-wrap:wrap;gap:3px 24px}.path-detail-metric{white-space:nowrap}.path-detail-uri{display:block;margin-top:7px}.audit-warning{margin:8px 0 24px;padding:16px 18px;border-left:4px solid var(--down);border-radius:6px;background:var(--detail)}.audit-warning h2{font-size:17px}.audit-warning p{margin:6px 0 10px}.audit-warning ul{margin:0;padding-left:20px;font-variant-numeric:tabular-nums}.supporting-plots{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:32px}.supporting-plots img{display:block;width:100%;height:auto}
.repetition-line{display:flex;flex-wrap:wrap;gap:5px 12px;margin-top:7px;color:var(--muted);font-variant-numeric:tabular-nums}
@media(prefers-reduced-motion:reduce){.path-chevron{transition:none}}
@media(max-width:760px){body{padding:24px 16px 48px}.summary-metrics{margin-left:6px;margin-right:6px}.mix-row,.subcategory-row{grid-template-columns:1fr;gap:7px}.mix-value{white-space:normal}.bar-track{grid-column:1}.detail-head{grid-template-columns:1fr}.detail-total{white-space:normal;text-align:left}.category-grid{grid-template-columns:1fr}.category-metrics,.category-metrics.proposal-metrics,.path-sampling{grid-template-columns:1fr}.supporting-plots{grid-template-columns:1fr}}
</style>
"""


def _interactive_chart_rows(
    rows: Sequence[dict[str, Any]],
    value_field: str,
    percent_field: str,
    detail_prefix: str,
    total: int,
    row_class: str = "mix-row",
) -> str:
    maximum = max((float(row[percent_field]) for row in rows), default=1.0) or 1.0
    output: list[str] = []
    for index, row in enumerate(rows):
        detail_id = f"{detail_prefix}-{index}"
        row["detail_id"] = detail_id
        percent = float(row[percent_field])
        relative_width = 100 * percent / maximum
        tokens = int(row[value_field])
        value_label = str(row.get("value_label") or _mix_plot_value_label(tokens, total))
        metric_columns = row.get("metric_columns")
        if metric_columns:
            metrics = (
                '<span class="mix-metrics">'
                + "".join(
                    '<span class="mix-metric">'
                    f'<span class="mix-metric-label">{html.escape(str(metric["label"]))}</span>'
                    f'<span class="mix-metric-value">{html.escape(str(metric["value"]))}</span>'
                    "</span>"
                    for metric in metric_columns
                )
                + "</span>"
            )
            rendered_row_class = f"{row_class} has-metrics"
            value = metrics
        else:
            rendered_row_class = row_class
            value = f'<span class="mix-value">{html.escape(value_label)}</span>'
        output.append(
            f'<button type="button" class="{rendered_row_class}" data-detail="{detail_id}" '
            f'aria-controls="{detail_id}" aria-expanded="false">'
            f'<span class="mix-name">{html.escape(str(row.get("display_name", row["mix_name"])))}</span>'
            + value
            + f'<span class="bar-track" aria-hidden="true"><span class="bar-fill" style="width:{relative_width:.8f}%"></span></span>'
            "</button>"
        )
    return "".join(output)


def _interactive_report_script() -> str:
    return """
<script>
const setAccordionState = (button, detail, expanded) => {
  button.classList.toggle('is-selected', expanded);
  button.setAttribute('aria-expanded', String(expanded));
  detail.hidden = !expanded;
};
document.querySelectorAll('.mix-row').forEach((button) => {
  button.addEventListener('click', () => {
    const target = document.getElementById(button.dataset.detail);
    const opening = button.getAttribute('aria-expanded') !== 'true';
    document.querySelectorAll('.mix-row').forEach((row) => {
      const detail = document.getElementById(row.dataset.detail);
      if (detail) setAccordionState(row, detail, false);
    });
    document.querySelectorAll('.subcategory-row').forEach((row) => {
      const detail = document.getElementById(row.dataset.detail);
      if (detail) setAccordionState(row, detail, false);
    });
    if (opening) {
      button.insertAdjacentElement('afterend', target);
      setAccordionState(button, target, true);
      target.scrollIntoView({block: 'nearest'});
    }
  });
});
document.querySelectorAll('.subcategory-row').forEach((button) => {
  button.addEventListener('click', () => {
    const family = button.closest('.mix-detail');
    const target = document.getElementById(button.dataset.detail);
    const opening = button.getAttribute('aria-expanded') !== 'true';
    family.querySelectorAll('.subcategory-row').forEach((row) => {
      const detail = document.getElementById(row.dataset.detail);
      if (detail) setAccordionState(row, detail, false);
    });
    if (opening) {
      button.insertAdjacentElement('afterend', target);
      setAccordionState(button, target, true);
      target.scrollIntoView({block: 'nearest'});
    }
  });
});
</script>
"""


def _human_byte_count(value: int) -> str:
    for scale, suffix in (
        (1_000_000_000_000, "TB"),
        (1_000_000_000, "GB"),
        (1_000_000, "MB"),
        (1_000, "KB"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:.3g} {suffix}"
    return f"{value:,} B"


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _proposal_artifact_href(path: Any) -> str:
    relative = Path(str(path)).as_posix()
    prefix = "01-plan/execution/"
    if relative.startswith(prefix):
        relative = relative[len(prefix) :]
    return html.escape(relative, quote=True)


def _execution_proposal_style() -> str:
    return """
<style>
:root{color-scheme:light dark;--muted:#536965;--surface:#edf6f4;--surface-hover:#e4f0ed;--track:#d2e1de;--source:#71817e;--output:#218f84;--selection:#d28a32;--link:#126a63}
@media(prefers-color-scheme:dark){:root{--muted:#a7bbb7;--surface:#142420;--surface-hover:#1a2d29;--track:#2a403c;--source:#91a29f;--output:#5cc8bb;--selection:#e5a456;--link:#74d7cb}}
*{box-sizing:border-box}body{max-width:1240px;margin:0 auto;padding:34px 26px 72px;background:Canvas;color:CanvasText;font:14px/1.45 system-ui,sans-serif}h1{margin:0;font-size:28px;line-height:1.2}h2{margin:34px 0 14px;font-size:20px}.lede{max-width:820px;margin:8px 0 0;color:var(--muted)}.execution-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px 28px;margin:20px 0}.execution-metric{min-width:0}.metric-label{display:block;color:var(--muted)}.metric-value{display:block;margin-top:2px;font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}.storage-note{margin:0 0 16px;color:var(--muted)}.utilization-bands{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.utilization-band{padding:12px 14px;border-radius:8px;background:var(--surface)}.utilization-band strong{display:block;font-size:18px;font-variant-numeric:tabular-nums}.utilization-band span{color:var(--muted)}.split-list{display:grid;gap:6px}.split-category{display:grid;grid-template-columns:minmax(0,1fr) repeat(3,minmax(115px,auto));gap:12px 24px;align-items:center;padding:11px 14px;border-radius:8px;background:var(--surface)}.split-name{overflow-wrap:anywhere;font-weight:600}.split-value{font-variant-numeric:tabular-nums}.split-value span{display:block;color:var(--muted);font-size:12px}.unit-heading{display:flex;flex-wrap:wrap;gap:10px 20px;align-items:end;justify-content:space-between}.unit-heading h2{margin-bottom:0}.visible-count{color:var(--muted);font-variant-numeric:tabular-nums}.unit-controls{display:grid;grid-template-columns:minmax(220px,1fr) auto;gap:10px 20px;margin:14px 0}.unit-controls input[type=search]{width:100%;padding:9px 11px;border:0;border-radius:7px;background:var(--surface);color:inherit;font:inherit}.unit-controls label{display:flex;gap:8px;align-items:center;color:var(--muted)}.unit-list{display:grid;gap:6px}.execution-unit{border:0;border-radius:9px;background:var(--surface)}.execution-unit[hidden]{display:none}.execution-unit summary{display:grid;grid-template-columns:minmax(0,1fr) minmax(190px,260px);gap:10px 28px;padding:14px 16px;cursor:pointer;list-style-position:inside}.execution-unit summary:hover{background:var(--surface-hover);border-radius:9px}.unit-title{min-width:0;overflow-wrap:anywhere;font-weight:600}.unit-position{display:block;margin:2px 0 0 18px;color:var(--muted);font-size:12px;font-weight:400}.unit-disk{font-variant-numeric:tabular-nums}.unit-disk strong,.unit-disk span{display:block}.unit-disk span{color:var(--muted);font-size:12px}.unit-body{padding:2px 16px 17px}.unit-metrics{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px 24px;margin:6px 0 16px}.unit-metric{min-width:0}.unit-metric span{display:block;color:var(--muted);font-size:12px}.unit-metric strong{display:block;margin-top:2px;font-weight:600;font-variant-numeric:tabular-nums}.disk-breakdown{display:flex;height:9px;overflow:hidden;border-radius:999px;background:var(--track)}.disk-segment{display:block;height:100%}.disk-source{background:var(--source)}.disk-output{background:var(--output)}.disk-selection{background:var(--selection)}.disk-legend{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px 22px;margin:7px 0 16px;color:var(--muted);font-variant-numeric:tabular-nums}.disk-legend span::before{display:inline-block;width:8px;height:8px;margin-right:6px;border-radius:2px;content:""}.legend-source::before{background:var(--source)}.legend-output::before{background:var(--output)}.legend-selection::before{background:var(--selection)}.unit-paths{display:grid;gap:8px;margin:0}.unit-paths div{min-width:0}.unit-paths dt{color:var(--muted);font-size:12px}.unit-paths dd{margin:2px 0 0}.unit-paths code{display:block;padding:8px 10px;border-radius:6px;background:Canvas;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,monospace}.artifact-links{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:12px}.artifact-links a{color:var(--link);font-weight:600;text-decoration:none}.artifact-links a:hover{text-decoration:underline}
@media(max-width:760px){body{padding:24px 16px 48px}.utilization-bands{grid-template-columns:repeat(2,1fr)}.split-category{grid-template-columns:1fr 1fr}.execution-unit summary{grid-template-columns:1fr}.unit-metrics{grid-template-columns:repeat(2,1fr)}.disk-legend{grid-template-columns:1fr}.unit-controls{grid-template-columns:1fr}}
</style>
"""


def _execution_proposal_script() -> str:
    return """
<script>
const unitSearch = document.getElementById('unit-search');
const splitOnly = document.getElementById('split-only');
const unitCards = Array.from(document.querySelectorAll('.execution-unit'));
const visibleCount = document.getElementById('visible-unit-count');
function filterUnits() {
  const query = unitSearch.value.trim().toLowerCase();
  let visible = 0;
  unitCards.forEach((card) => {
    const matchesText = !query || card.dataset.search.includes(query);
    const matchesSplit = !splitOnly.checked || card.dataset.split === 'true';
    card.hidden = !(matchesText && matchesSplit);
    if (!card.hidden) visible += 1;
  });
  visibleCount.textContent = `${visible.toLocaleString()} of ${unitCards.length.toLocaleString()} units`;
}
unitSearch.addEventListener('input', filterUnits);
splitOnly.addEventListener('change', filterUnits);
filterUnits();
</script>
"""


def _render_execution_proposal_html(
    execution_units: Sequence[dict[str, Any]],
    category_execution: Sequence[dict[str, Any]],
    planned_total: int,
) -> str:
    def render_metrics(items: Sequence[tuple[str, str]]) -> str:
        return (
            '<div class="execution-metrics">'
            + "".join(
                '<div class="execution-metric">'
                f'<span class="metric-label">{html.escape(label)}</span>'
                f'<span class="metric-value">{html.escape(value)}</span></div>'
                for label, value in items
            )
            + "</div>"
        )

    ordered_units = sorted(
        execution_units,
        key=lambda row: (int(row["estimated_peak_local_bytes"]), row["unit_id"]),
        reverse=True,
    )
    peak_values = [int(row["estimated_peak_local_bytes"]) for row in ordered_units]
    max_budget = max((int(row["max_unit_working_bytes"]) for row in ordered_units), default=0)
    split_categories = sorted(
        (row for row in category_execution if int(row["execution_unit_count"]) > 1),
        key=lambda row: (
            int(row["execution_unit_count"]),
            int(row["largest_estimated_peak_local_bytes"]),
            row["leaf_id"],
        ),
        reverse=True,
    )
    partial_units = sum(int(row["partial_object_count"]) > 0 for row in ordered_units)
    utilizations = [
        int(row["estimated_peak_local_bytes"]) / int(row["max_unit_working_bytes"]) for row in ordered_units
    ]
    utilization_bands = [
        ("Below 50%", sum(value < 0.5 for value in utilizations)),
        ("50–75%", sum(0.5 <= value < 0.75 for value in utilizations)),
        ("75–90%", sum(0.75 <= value < 0.9 for value in utilizations)),
        ("90% or higher", sum(value >= 0.9 for value in utilizations)),
    ]
    metrics = render_metrics(
        [
            ("Execution units", f"{len(ordered_units):,}"),
            ("Units using document selection", f"{partial_units:,}"),
            ("Planned output", f"{_human_token_count(planned_total)} tokens"),
        ]
    )
    storage_metrics = render_metrics(
        [
            ("Worker disk budget", _human_byte_count(max_budget)),
            ("Median unit", _human_byte_count(_percentile(peak_values, 0.5))),
            ("P95 unit", _human_byte_count(_percentile(peak_values, 0.95))),
            ("Largest unit", _human_byte_count(max(peak_values, default=0))),
        ]
    )
    band_html = "".join(
        f'<div class="utilization-band"><strong>{count:,}</strong><span>{label}</span></div>'
        for label, count in utilization_bands
    )
    split_html = (
        '<div class="split-list">'
        + "".join(
            '<article class="split-category">'
            f'<span class="split-name">{html.escape(str(row["mix_name"]))} / '
            f'{html.escape(str(row["category_name"]))}</span>'
            f'<span class="split-value"><span>Units</span>{int(row["execution_unit_count"]):,}</span>'
            f'<span class="split-value"><span>Output</span>'
            f'{_human_token_count(int(row["planned_uint32_values"]))} tokens</span>'
            f'<span class="split-value"><span>Largest unit</span>'
            f'{_human_byte_count(int(row["largest_estimated_peak_local_bytes"]))}</span>'
            "</article>"
            for row in split_categories
        )
        + "</div>"
        if split_categories
        else '<p class="storage-note">No category requires more than one execution unit.</p>'
    )
    unit_cards: list[str] = []
    for row in ordered_units:
        input_bytes = int(row["input_npy_bytes"]) + int(row["input_metadata_bytes"])
        output_bytes = int(row["output_npy_bytes"]) + int(row["estimated_output_metadata_bytes"])
        selection_bytes = int(row["estimated_selection_index_bytes"])
        peak_bytes = int(row["estimated_peak_local_bytes"])
        budget_bytes = int(row["max_unit_working_bytes"])
        utilization = peak_bytes / budget_bytes
        source_width = 100 * input_bytes / budget_bytes
        output_width = 100 * output_bytes / budget_bytes
        selection_width = 100 * selection_bytes / budget_bytes
        category_label = f'{row["mix_name"]} / {row["category_name"]}'
        searchable = html.escape(
            f'{row["unit_id"]} {row["mix_name"]} {row["category_name"]}'.lower(),
            quote=True,
        )
        split = int(row["unit_count_for_category"]) > 1
        unit_cards.append(
            f'<details class="execution-unit" data-search="{searchable}" '
            f'data-split="{str(split).lower()}"><summary>'
            f'<span class="unit-title">{html.escape(category_label)}'
            f'<span class="unit-position">Unit {int(row["unit_index"]):,} of '
            f'{int(row["unit_count_for_category"]):,}</span></span>'
            f'<span class="unit-disk"><strong>{_human_byte_count(peak_bytes)}</strong>'
            f"<span>{utilization:.1%} of worker disk budget</span></span></summary>"
            '<div class="unit-body"><div class="unit-metrics">'
            '<div class="unit-metric"><span>Output tokens</span>'
            f'<strong>{_human_token_count(int(row["planned_uint32_values"]))}</strong></div>'
            '<div class="unit-metric"><span>Source shard downloads</span>'
            f'<strong>{int(row["unique_object_count"]):,}</strong></div>'
            '<div class="unit-metric"><span>Source shards using document selection</span>'
            f'<strong>{int(row["partial_object_count"]):,}</strong></div>'
            '<div class="unit-metric"><span>Output shard cap</span>'
            f'<strong>{int(row["max_num_files"]):,}</strong></div>'
            "</div>"
            '<div class="disk-breakdown" aria-label="Estimated local disk composition">'
            f'<span class="disk-segment disk-source" style="width:{source_width:.8f}%"></span>'
            f'<span class="disk-segment disk-output" style="width:{output_width:.8f}%"></span>'
            f'<span class="disk-segment disk-selection" style="width:{selection_width:.8f}%"></span>'
            "</div>"
            '<div class="disk-legend">'
            f'<span class="legend-source">Source download: {_human_byte_count(input_bytes)}</span>'
            f'<span class="legend-output">Materialized output: {_human_byte_count(output_bytes)}</span>'
            f'<span class="legend-selection">Selection indexes: {_human_byte_count(selection_bytes)}</span>'
            "</div>"
            '<dl class="unit-paths"><div><dt>Unit ID</dt>'
            f'<dd><code>{html.escape(str(row["unit_id"]))}</code></dd></div>'
            "<div><dt>Destination</dt>"
            f'<dd><code>{html.escape(str(row["destination_prefix"]))}</code></dd></div></dl>'
            '<nav class="artifact-links" aria-label="Execution-unit artifacts">'
            f'<a href="{_proposal_artifact_href(row["config_path"])}">Config</a>'
            f'<a href="{_proposal_artifact_href(row["manifest_path"])}">Manifest</a>'
            f'<a href="{_proposal_artifact_href(row["launcher_path"])}">Launcher</a>'
            "</nav></div></details>"
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Dolma 3.5 materialization execution proposal</title>"
        + _execution_proposal_style()
        + "</head><body><h1>Dolma 3.5 Materialization Execution Proposal</h1>"
        '<p class="lede">Worker partitioning, local-disk requirements, and runnable artifacts for the '
        "materialization phase.</p>"
        + metrics
        + "<h2>Worker storage</h2>"
        + storage_metrics
        + f'<div class="utilization-bands">{band_html}</div>'
        + "<h2>Categories requiring multiple execution units</h2>"
        + split_html
        + '<div class="unit-heading"><h2>Execution units</h2>'
        f'<span class="visible-count" id="visible-unit-count">{len(ordered_units):,} of '
        f"{len(ordered_units):,} units</span></div>"
        '<div class="unit-controls"><input id="unit-search" type="search" '
        'placeholder="Filter by source, category, or unit ID" aria-label="Filter execution units">'
        '<label><input id="split-only" type="checkbox"> Only categories with multiple units</label></div>'
        f'<div class="unit-list">{"".join(unit_cards)}</div>' + _execution_proposal_script() + "</body></html>\n"
    )


def _report_document_with_base(document: str, relative_base: str) -> str:
    marker = "<head>"
    if marker not in document:
        raise PreparationError("Generated report is missing its HTML head")
    return document.replace(marker, f'<head><base href="{html.escape(relative_base, quote=True)}">', 1)


def _combine_plan_reports(build: Path) -> None:
    """Embed the sampling and execution reports in one tabbed, self-contained file."""

    plan_root = build / "01-plan"
    resolution_report_path = plan_root / "resolution/report.html"
    inventory_report_path = plan_root / "inventory/report.html"
    execution_report_path = plan_root / "execution/report.html"
    for report_path in (inventory_report_path, execution_report_path):
        if report_path.is_symlink() or not report_path.is_file():
            raise PreparationError(f"Cannot compose missing or unsafe report: {report_path}")

    inventory_document = _report_document_with_base(
        inventory_report_path.read_text(encoding="utf-8"),
        "inventory/",
    )
    execution_document = _report_document_with_base(
        execution_report_path.read_text(encoding="utf-8"),
        "execution/",
    )
    combined = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dolma 3.5 Resharding Plan</title>
<style>
:root{{color-scheme:light dark;--shell:#f4f8f7;--tab:#e1ece9;--selected:#14786f;--muted:#536965;--line:#cbdad7}}
@media(prefers-color-scheme:dark){{:root{{--shell:#0e1715;--tab:#172522;--selected:#5cc8bb;--muted:#a7bbb7;--line:#2a403c}}}}
*{{box-sizing:border-box}}html,body{{height:100%;margin:0}}body{{overflow:hidden;background:var(--shell);color:CanvasText;font:14px/1.4 system-ui,sans-serif}}
.report-shell{{display:grid;height:100%;grid-template-rows:auto minmax(0,1fr)}}
.report-tabs{{display:flex;gap:8px;padding:10px 18px;border-bottom:1px solid var(--line);background:var(--shell)}}
.report-tab{{padding:9px 14px;border:0;border-radius:7px;background:transparent;color:var(--muted);font:inherit;font-weight:650;cursor:pointer}}
.report-tab:hover{{background:var(--tab);color:CanvasText}}.report-tab[aria-selected="true"]{{background:var(--tab);color:var(--selected)}}
.report-panel{{min-height:0}}.report-panel[hidden]{{display:none}}.report-frame{{display:block;width:100%;height:100%;border:0;background:Canvas}}
@media(max-width:620px){{.report-tabs{{padding:8px}}.report-tab{{flex:1;padding:9px 8px}}}}
</style>
</head>
<body>
<template id="source-report-document">{inventory_document}</template>
<template id="execution-report-document">{execution_document}</template>
<main class="report-shell">
  <nav class="report-tabs" role="tablist" aria-label="Plan report views">
    <button class="report-tab" id="source-report-tab" type="button" role="tab" aria-controls="source-report-panel" aria-selected="true" data-report="source">Source inventory &amp; sampling</button>
    <button class="report-tab" id="execution-report-tab" type="button" role="tab" aria-controls="execution-report-panel" aria-selected="false" data-report="execution">Materialization execution</button>
  </nav>
  <section class="report-panel" id="source-report-panel" role="tabpanel" aria-labelledby="source-report-tab">
    <iframe class="report-frame" id="source-report-frame" title="Source inventory and sampling report"></iframe>
  </section>
  <section class="report-panel" id="execution-report-panel" role="tabpanel" aria-labelledby="execution-report-tab" hidden>
    <iframe class="report-frame" id="execution-report-frame" title="Materialization execution report"></iframe>
  </section>
</main>
<script>
const reportNames = ['source', 'execution'];
reportNames.forEach((name) => {{
  const template = document.getElementById(`${{name}}-report-document`);
  document.getElementById(`${{name}}-report-frame`).srcdoc = template.innerHTML;
}});
const reportTabs = Array.from(document.querySelectorAll('.report-tab'));
function selectReport(name, updateHash = true) {{
  reportTabs.forEach((tab) => {{
    const selected = tab.dataset.report === name;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(`${{tab.dataset.report}}-report-panel`).hidden = !selected;
  }});
  if (updateHash) history.replaceState(null, '', `#${{name}}`);
}}
reportTabs.forEach((tab, index) => {{
  tab.addEventListener('click', () => selectReport(tab.dataset.report));
  tab.addEventListener('keydown', (event) => {{
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === 'ArrowRight' ? 1 : -1;
    const next = reportTabs[(index + offset + reportTabs.length) % reportTabs.length];
    selectReport(next.dataset.report);
    next.focus();
  }});
}});
selectReport(location.hash === '#execution' ? 'execution' : 'source', false);
</script>
</body>
</html>
"""
    _write_text(plan_root / "report.html", combined)
    for stage_report_path in (
        resolution_report_path,
        inventory_report_path,
        execution_report_path,
    ):
        if stage_report_path.exists():
            if stage_report_path.is_symlink() or not stage_report_path.is_file():
                raise PreparationError(f"Refusing to remove an unsafe stage report: {stage_report_path}")
            stage_report_path.unlink()


def _render_plan_report(
    phase: Path,
    normalized_mix: Sequence[dict[str, Any]],
    normalized_paths: Sequence[dict[str, Any]],
    catalog_matches: Sequence[dict[str, Any]],
    direct_patterns: Sequence[dict[str, Any]],
) -> None:
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plots.mkdir(exist_ok=False)
    plot_data.mkdir(exist_ok=False)
    target_by_mix: dict[str, int] = defaultdict(int)
    for row in normalized_mix:
        target_by_mix[row["mix_name"]] += int(row["target_uint32_values"])
    target_total = sum(target_by_mix.values())
    target_rows = [
        {
            "mix_name": name,
            "target_uint32_values": value,
            "target_percent": f"{100 * value / target_total:.8f}" if target_total else "0",
        }
        for name, value in sorted(target_by_mix.items(), key=lambda item: item[1], reverse=True)
    ]
    _write_csv(
        plot_data / "target-mix.csv",
        target_rows,
        ["mix_name", "target_uint32_values", "target_percent"],
    )
    subcategories_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        source_family, subcategory = _split_mix_name(str(row["mix_name"]))
        row["source_family"] = source_family
        row["subcategory_name"] = subcategory
        row["display_name"] = subcategory
        subcategories_by_family[source_family].append(row)
    family_rows: list[dict[str, Any]] = []
    for source_family, subcategories in subcategories_by_family.items():
        family_target = sum(int(row["target_uint32_values"]) for row in subcategories)
        family_percent = 100 * family_target / target_total if target_total else 0.0
        ordered_subcategories = sorted(
            subcategories,
            key=lambda row: (
                int(row["target_uint32_values"]),
                str(row["subcategory_name"]),
            ),
            reverse=True,
        )
        for row in ordered_subcategories:
            sub_target = int(row["target_uint32_values"])
            sub_percent = 100 * sub_target / family_target if family_target else 0.0
            row["family_target_percent"] = f"{sub_percent:.8f}"
            row["metric_columns"] = [
                {
                    "label": "Target",
                    "value": (f"{_human_token_count(sub_target)} tokens · " f"{sub_percent:.2f}% of family"),
                }
            ]
        family_rows.append(
            {
                "mix_name": source_family,
                "target_uint32_values": family_target,
                "target_percent": f"{family_percent:.8f}",
                "subcategories": ordered_subcategories,
                "metric_columns": [
                    {
                        "label": "Target",
                        "value": (f"{_human_token_count(family_target)} tokens · " f"{family_percent:.2f}%"),
                    }
                ],
            }
        )
    family_rows.sort(
        key=lambda row: (
            int(row["target_uint32_values"]),
            str(row["mix_name"]),
        ),
        reverse=True,
    )
    _write_text(
        plots / "target-mix.svg",
        _svg_bar_chart(
            "Target allocation by source family",
            [row["mix_name"] for row in family_rows],
            [float(row["target_percent"]) for row in family_rows],
            "% of target",
            value_labels=[_mix_plot_value_label(row["target_uint32_values"], target_total) for row in family_rows],
            summary=(f"Total target: {_human_token_count(target_total)} tokens " f"({target_total:,})"),
        ),
    )
    chart_rows = _interactive_chart_rows(
        family_rows,
        "target_uint32_values",
        "target_percent",
        "plan-family-detail",
        target_total,
    )
    for family_index, family_row in enumerate(family_rows):
        family_row["subcategory_chart_rows"] = _interactive_chart_rows(
            family_row["subcategories"],
            "target_uint32_values",
            "family_target_percent",
            f"plan-subcategory-{family_index}",
            int(family_row["target_uint32_values"]),
            row_class="subcategory-row",
        )
    paths_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_paths:
        paths_by_leaf[row["leaf_id"]].append(row)
    catalog_counts = Counter(row["path_id"] for row in catalog_matches)
    catalog_sources_by_path: dict[str, list[str]] = defaultdict(list)
    for row in catalog_matches:
        catalog_sources_by_path[row["path_id"]].append(f's3://{row["bucket"]}/{row["key"]}')
    direct_sources_by_path = {
        row["path_id"]: f's3://{row["bucket"]}/{row["key_pattern"]}' for row in direct_patterns
    }
    categories_by_mix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_mix:
        categories_by_mix[row["mix_name"]].append(row)

    subcategory_detail_by_mix: dict[str, str] = {}
    for target_row in target_rows:
        mix_name = str(target_row["mix_name"])
        mix_target = int(target_row["target_uint32_values"])
        category_sections: list[str] = []
        for category in sorted(categories_by_mix[mix_name], key=lambda row: int(row["category_index"])):
            category_target = int(category["target_uint32_values"])
            category_percent = 100 * category_target / mix_target if mix_target else 0
            path_rows: list[str] = []
            for path in sorted(paths_by_leaf[category["leaf_id"]], key=lambda row: row["path_id"]):
                active = path["active"] == "true"
                source_uris = sorted(set(catalog_sources_by_path[path["path_id"]]))
                direct_source = direct_sources_by_path.get(path["path_id"])
                if direct_source is not None:
                    source_uris.append(direct_source)
                if not active:
                    matched = "inactive"
                elif direct_source is not None:
                    matched = "pending inventory"
                else:
                    matched = _count_label(catalog_counts[path["path_id"]], "matched NPY")
                source_details = _source_uri_details(source_uris, "No resolved S3 source")
                path_rows.append(
                    '<details class="path-detail"><summary>'
                    f'<span class="path-name">{html.escape(_path_subgroup(path["yaml_path"]))}</span>'
                    f'<span class="path-stat">{html.escape(matched)}'
                    '<span class="path-chevron" aria-hidden="true">›</span></span>'
                    "</summary>"
                    f"<code>{source_details}</code>"
                    "</details>"
                )
            category_sections.append(
                '<section class="category">'
                '<div class="category-head">'
                f'<span class="category-name">{html.escape(category["category_name"])}</span>'
                f'<span class="category-value">{html.escape(_human_token_count(category_target))} tokens · {category_percent:.2f}%</span>'
                "</div>"
                '<div class="category-bar" aria-hidden="true">'
                f'<span class="category-bar-fill" style="width:{category_percent:.8f}%"></span></div>'
                '<div class="path-list">' + "".join(path_rows) + "</div>" + "</section>"
            )
        subcategory_detail_by_mix[mix_name] = (
            f'<section class="subcategory-detail" id="{target_row["detail_id"]}" hidden>'
            '<div class="detail-head">'
            f'<h2>{html.escape(str(target_row["subcategory_name"]))}</h2>'
            f'<div class="detail-total">{_human_token_count(mix_target)} tokens · '
            f'{float(target_row["target_percent"]):.2f}% of target</div></div>'
            '<div class="category-grid">' + "".join(category_sections) + "</div>" + "</section>"
        )
    detail_sections: list[str] = []
    for family_row in family_rows:
        family_target = int(family_row["target_uint32_values"])
        detail_sections.append(
            f'<section class="mix-detail" id="{family_row["detail_id"]}" hidden>'
            '<div class="detail-head">'
            f'<h2>{html.escape(str(family_row["mix_name"]))}</h2>'
            f'<div class="detail-total">{_human_token_count(family_target)} tokens · '
            f'{float(family_row["target_percent"]):.2f}% of target</div></div>'
            '<div class="subcategory-list">'
            + str(family_row["subcategory_chart_rows"])
            + "</div>"
            + "".join(
                subcategory_detail_by_mix[str(subcategory["mix_name"])]
                for subcategory in family_row["subcategories"]
            )
            + "</section>"
        )
    report_html = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 target allocation and source path plan</title>'
        + _interactive_report_style()
        + "</head><body>"
        + "<h1>Dolma 3.5 Target Allocation and Source Path Plan</h1>"
        f'<div class="chart-total">Materialized output target: '
        f"{_human_token_count(target_total)} tokens ({target_total:,})</div>"
        f'<div class="mix-chart">{chart_rows}</div>'
        + "".join(detail_sections)
        + _interactive_report_script()
        + "</body></html>\n"
    )
    _write_text(phase / "report.html", report_html)


def _build_inventory_details(
    normalized_mix: Sequence[dict[str, Any]],
    normalized_paths: Sequence[dict[str, Any]],
    required_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories_by_mix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in required_rows:
        rows_by_leaf[row["leaf_id"]].append(row)
        rows_by_path[row["path_id"]].append(row)
    for row in normalized_paths:
        paths_by_leaf[row["leaf_id"]].append(row)
    for row in normalized_mix:
        categories_by_mix[row["mix_name"]].append(row)

    source_rows: list[dict[str, Any]] = []
    for mix_name, categories in categories_by_mix.items():
        source_family, subcategory_name = _split_mix_name(str(mix_name))
        category_rows: list[dict[str, Any]] = []
        source_object_uris: set[str] = set()
        for category in sorted(categories, key=lambda row: int(row["category_index"])):
            leaf_id = category["leaf_id"]
            category_objects = {
                row["npy_uri"]: int(row["estimated_uint32_values"]) for row in rows_by_leaf[leaf_id]
            }
            source_object_uris.update(category_objects)
            source_tokens = sum(category_objects.values())
            target_tokens = int(category["target_uint32_values"])
            path_definitions = sorted(paths_by_leaf[leaf_id], key=lambda row: row["path_id"])
            path_objects = [
                {row["npy_uri"]: int(row["estimated_uint32_values"]) for row in rows_by_path[path["path_id"]]}
                for path in path_definitions
            ]
            path_source_tokens = [sum(objects.values()) for objects in path_objects]
            path_targets = _apportion_by_size(target_tokens, path_source_tokens)
            lower_groups: list[dict[str, Any]] = []
            for path, objects, path_source, path_target in zip(
                path_definitions, path_objects, path_source_tokens, path_targets
            ):
                _, _, ratio = _sampling_change(path_source, path_target)
                sampling_rate, _ = _sampling_rate_label(path_source, path_target)
                lower_groups.append(
                    {
                        "path_id": path["path_id"],
                        "lower_group": _path_subgroup(path["yaml_path"]),
                        "active": path["active"] == "true",
                        "yaml_path": path["yaml_path"],
                        "source_uint32_values": path_source,
                        "implied_target_uint32_values": path_target,
                        "token_delta": path_target - path_source,
                        "sampling_ratio": ratio,
                        "sampling_rate": sampling_rate,
                        "unique_npy_count": len(objects),
                    }
                )
            _, _, ratio = _sampling_change(source_tokens, target_tokens)
            sampling_rate, _ = _sampling_rate_label(source_tokens, target_tokens)
            category_rows.append(
                {
                    "leaf_id": leaf_id,
                    "category_name": category["category_name"],
                    "active": category["active"] == "true",
                    "source_uint32_values": source_tokens,
                    "target_uint32_values": target_tokens,
                    "token_delta": target_tokens - source_tokens,
                    "sampling_ratio": ratio,
                    "sampling_rate": sampling_rate,
                    "unique_npy_count": len(category_objects),
                    "lower_groups": lower_groups,
                }
            )
        source_tokens = sum(row["source_uint32_values"] for row in category_rows)
        target_tokens = sum(row["target_uint32_values"] for row in category_rows)
        for category in category_rows:
            category["source_percent_of_parent"] = (
                100 * category["source_uint32_values"] / source_tokens if source_tokens else 0.0
            )
            category["target_percent_of_parent"] = (
                100 * category["target_uint32_values"] / target_tokens if target_tokens else 0.0
            )
            for lower_group in category["lower_groups"]:
                lower_group["source_percent_of_parent"] = (
                    100 * lower_group["source_uint32_values"] / category["source_uint32_values"]
                    if category["source_uint32_values"]
                    else 0.0
                )
                lower_group["implied_target_percent_of_parent"] = (
                    100 * lower_group["implied_target_uint32_values"] / category["target_uint32_values"]
                    if category["target_uint32_values"]
                    else 0.0
                )
        _, _, ratio = _sampling_change(source_tokens, target_tokens)
        sampling_rate, _ = _sampling_rate_label(source_tokens, target_tokens)
        source_rows.append(
            {
                "mix_name": mix_name,
                "source_family": source_family,
                "subcategory_name": subcategory_name,
                "source_uint32_values": source_tokens,
                "target_uint32_values": target_tokens,
                "token_delta": target_tokens - source_tokens,
                "sampling_ratio": ratio,
                "sampling_rate": sampling_rate,
                "unique_npy_count": len(source_object_uris),
                "categories": category_rows,
            }
        )

    source_total = sum(row["source_uint32_values"] for row in source_rows)
    target_total = sum(row["target_uint32_values"] for row in source_rows)
    for source in source_rows:
        source["source_percent_of_total"] = (
            100 * source["source_uint32_values"] / source_total if source_total else 0.0
        )
        source["target_percent_of_total"] = (
            100 * source["target_uint32_values"] / target_total if target_total else 0.0
        )
        for category in source["categories"]:
            category["source_percent_of_total"] = (
                100 * category["source_uint32_values"] / source_total if source_total else 0.0
            )
            category["target_percent_of_total"] = (
                100 * category["target_uint32_values"] / target_total if target_total else 0.0
            )
            for lower_group in category["lower_groups"]:
                lower_group["source_percent_of_total"] = (
                    100 * lower_group["source_uint32_values"] / source_total if source_total else 0.0
                )
                lower_group["implied_target_percent_of_total"] = (
                    100 * lower_group["implied_target_uint32_values"] / target_total if target_total else 0.0
                )

    source_rows.sort(
        key=lambda row: (
            row["target_uint32_values"],
            row["source_uint32_values"],
            row["mix_name"],
        ),
        reverse=True,
    )
    _, _, aggregate_ratio = _sampling_change(source_total, target_total)
    aggregate_rate, _ = _sampling_rate_label(source_total, target_total)
    return {
        "schema_version": 1,
        "source_uint32_values": source_total,
        "target_uint32_values": target_total,
        "token_delta": target_total - source_total,
        "sampling_ratio": aggregate_ratio,
        "sampling_rate": aggregate_rate,
        "source_family_count": len({str(row["source_family"]) for row in source_rows}),
        "subcategory_count": len(source_rows),
        "source_count": len(source_rows),
        "category_count": sum(len(row["categories"]) for row in source_rows),
        "lower_group_count": sum(
            len(category["lower_groups"]) for source in source_rows for category in source["categories"]
        ),
        "sources": source_rows,
    }


def _replace_inventory_json(build: Path, path: Path, value: Any) -> None:
    _validate_preparation_build(build)
    phase = build / "01-plan/inventory"
    if path.parent != phase or path.name not in {
        "inventory-summary.json",
        "inventory-details.json",
    }:
        raise PreparationError(f"Refusing to replace non-inventory artifact: {path}")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise PreparationError(f"Unsafe generated inventory artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temporary, value)
    try:
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def refresh_inventory_details(build: Path) -> dict[str, Any]:
    build = build.resolve()
    _validate_preparation_build(build)
    phase = build / "01-plan/inventory"
    summary_path = phase / "inventory-summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise PreparationError(f"Inventory summary is missing: {summary_path}")
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    details = _build_inventory_details(
        normalized_mix=_read_csv(build / "01-plan/resolution/normalized-mix.csv"),
        normalized_paths=_read_csv(build / "01-plan/resolution/normalized-paths.csv"),
        required_rows=_read_csv(phase / "required-objects.csv"),
    )
    summary_source = int(
        summary["source_uint32_values"] if "source_uint32_values" in summary else summary["original_uint32_values"]
    )
    if summary_source != details["source_uint32_values"]:
        raise PreparationError("Inventory detail source total does not match summary")
    if int(summary["target_uint32_values"]) != details["target_uint32_values"]:
        raise PreparationError("Inventory detail target total does not match summary")
    metadata = {
        "source_uint32_values": details["source_uint32_values"],
        "token_delta": details["token_delta"],
        "sampling_ratio": details["sampling_ratio"],
        "sampling_rate": details["sampling_rate"],
        "source_family_count": details["source_family_count"],
        "subcategory_count": details["subcategory_count"],
        "source_count": details["source_count"],
        "category_count": details["category_count"],
        "lower_group_count": details["lower_group_count"],
        "details_artifact": "inventory-details.json",
        "report_artifact": "../report.html",
    }
    summary.pop("sampling_change", None)
    summary.update(metadata)
    _replace_inventory_json(build, phase / "inventory-details.json", details)
    _replace_inventory_json(build, summary_path, summary)
    return summary


def _render_inventory_report(
    phase: Path,
    normalized_mix: Sequence[dict[str, Any]],
    normalized_paths: Sequence[dict[str, Any]],
    required_rows: Sequence[dict[str, Any]],
    all_objects: Sequence[dict[str, Any]],
    missing_rows: Sequence[dict[str, Any]],
    resolution_failures: Sequence[dict[str, Any]],
    invalid_sizes: Sequence[dict[str, Any]],
    sampling_rate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plots.mkdir(exist_ok=False)
    plot_data.mkdir(exist_ok=False)
    found_npy = {(row["bucket"], row["key"]) for row in required_rows}
    found_metadata = {(row["bucket"], _pair_metadata_key(row["key"])) for row in required_rows}
    missing_npy = sum(row["object_type"] == "npy" for row in missing_rows)
    missing_metadata = sum(row["object_type"] == "metadata" for row in missing_rows)
    coverage_rows = [
        {"state": "found NPYs", "count": len(found_npy)},
        {"state": "missing NPYs", "count": missing_npy},
        {"state": "found metadata", "count": len(found_metadata)},
        {"state": "missing metadata", "count": missing_metadata},
        {"state": "path resolution failures", "count": len(resolution_failures)},
        {"state": "invalid NPY sizes", "count": len(invalid_sizes)},
    ]
    original_by_mix: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    for row in required_rows:
        identity = (row["leaf_id"], row["npy_uri"])
        if identity in seen:
            continue
        seen.add(identity)
        original_by_mix[row["mix_name"]] += int(row["estimated_uint32_values"])
    target_by_mix: dict[str, int] = defaultdict(int)
    categories_by_mix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_by_leaf: dict[str, int] = {}
    for row in normalized_mix:
        target = int(row["target_uint32_values"])
        target_by_mix[row["mix_name"]] += target
        target_by_leaf[row["leaf_id"]] = target
        categories_by_mix[row["mix_name"]].append(row)
    original_total = sum(original_by_mix.values())
    target_total = sum(target_by_mix.values())
    mix_names = sorted(
        set(original_by_mix) | set(target_by_mix),
        key=lambda name: (target_by_mix[name], original_by_mix[name], name),
        reverse=True,
    )
    comparison_rows: list[dict[str, Any]] = []
    for name in mix_names:
        original = original_by_mix[name]
        target = target_by_mix[name]
        source_percent = 100 * original / original_total if original_total else 0.0
        target_percent = 100 * target / target_total if target_total else 0.0
        _, change_class, ratio = _sampling_change(original, target)
        sampling_rate, _ = _sampling_rate_label(original, target)
        comparison_rows.append(
            {
                "mix_name": name,
                "available_uint32_values": original,
                "available_percent": f"{source_percent:.8f}",
                "target_uint32_values": target,
                "target_percent": f"{target_percent:.8f}",
                "sampling_ratio": "" if ratio is None else f"{ratio:.12g}",
                "sampling_rate": sampling_rate,
                "sampling_class": change_class,
                "metric_columns": [
                    {
                        "label": "Source",
                        "value": (f"{_human_token_count(original)} tokens · " f"{source_percent:.2f}%"),
                    },
                    {
                        "label": "Target",
                        "value": (f"{_human_token_count(target)} tokens · " f"{target_percent:.2f}%"),
                    },
                    {"label": "Sampling", "value": sampling_rate},
                ],
            }
        )

    subcategories_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        source_family, subcategory = _split_mix_name(str(row["mix_name"]))
        row["source_family"] = source_family
        row["subcategory_name"] = subcategory
        row["display_name"] = subcategory
        subcategories_by_family[source_family].append(row)

    family_rows: list[dict[str, Any]] = []
    for source_family, subcategories in subcategories_by_family.items():
        family_source = sum(int(row["available_uint32_values"]) for row in subcategories)
        family_target = sum(int(row["target_uint32_values"]) for row in subcategories)
        source_percent = 100 * family_source / original_total if original_total else 0.0
        target_percent = 100 * family_target / target_total if target_total else 0.0
        sampling_rate, sampling_class = _sampling_rate_label(family_source, family_target)
        ordered_subcategories = sorted(
            subcategories,
            key=lambda row: (
                int(row["target_uint32_values"]),
                int(row["available_uint32_values"]),
                str(row["subcategory_name"]),
            ),
            reverse=True,
        )
        for row in ordered_subcategories:
            sub_source = int(row["available_uint32_values"])
            sub_target = int(row["target_uint32_values"])
            sub_source_percent = 100 * sub_source / family_source if family_source else 0.0
            sub_target_percent = 100 * sub_target / family_target if family_target else 0.0
            row["family_target_percent"] = f"{sub_target_percent:.8f}"
            row["metric_columns"] = [
                {
                    "label": "Source",
                    "value": (
                        f"{_human_token_count(sub_source)} tokens · " f"{sub_source_percent:.2f}% of source"
                    ),
                },
                {
                    "label": "Target",
                    "value": (
                        f"{_human_token_count(sub_target)} tokens · " f"{sub_target_percent:.2f}% of target"
                    ),
                },
                {"label": "Sampling", "value": row["sampling_rate"]},
            ]
        family_rows.append(
            {
                "mix_name": source_family,
                "available_uint32_values": family_source,
                "available_percent": f"{source_percent:.8f}",
                "target_uint32_values": family_target,
                "target_percent": f"{target_percent:.8f}",
                "sampling_rate": sampling_rate,
                "sampling_class": sampling_class,
                "subcategories": ordered_subcategories,
                "metric_columns": [
                    {
                        "label": "Source",
                        "value": (f"{_human_token_count(family_source)} tokens · " f"{source_percent:.2f}%"),
                    },
                    {
                        "label": "Target",
                        "value": (f"{_human_token_count(family_target)} tokens · " f"{target_percent:.2f}%"),
                    },
                    {"label": "Sampling", "value": sampling_rate},
                ],
            }
        )
    family_rows.sort(
        key=lambda row: (
            int(row["target_uint32_values"]),
            int(row["available_uint32_values"]),
            str(row["mix_name"]),
        ),
        reverse=True,
    )
    available_rows = sorted(
        family_rows,
        key=lambda row: int(row["available_uint32_values"]),
        reverse=True,
    )[:40]
    _write_csv(plot_data / "coverage.csv", coverage_rows, ["state", "count"])
    _write_csv(
        plot_data / "available-by-mix.csv",
        comparison_rows,
        [
            "mix_name",
            "available_uint32_values",
            "available_percent",
            "target_uint32_values",
            "target_percent",
            "sampling_ratio",
            "sampling_rate",
        ],
    )
    _write_text(
        plots / "coverage.svg",
        _svg_bar_chart(
            "Source inventory coverage",
            [row["state"] for row in coverage_rows],
            [row["count"] for row in coverage_rows],
            "items",
        ),
    )
    _write_text(
        plots / "available-by-mix.svg",
        _svg_bar_chart(
            "Largest available mix entries by share of available tokens",
            [row["mix_name"] for row in available_rows],
            [float(row["available_percent"]) for row in available_rows],
            "% of available tokens",
            value_labels=[
                _mix_plot_value_label(row["available_uint32_values"], original_total) for row in available_rows
            ],
            summary=(f"Source aggregate: {_human_token_count(original_total)} tokens " f"({original_total:,})"),
        ),
    )
    chart_rows = _interactive_chart_rows(
        family_rows,
        "target_uint32_values",
        "target_percent",
        "inventory-family-detail",
        target_total,
    )
    for family_index, family_row in enumerate(family_rows):
        family_row["subcategory_chart_rows"] = _interactive_chart_rows(
            family_row["subcategories"],
            "target_uint32_values",
            "family_target_percent",
            f"inventory-subcategory-{family_index}",
            int(family_row["target_uint32_values"]),
            row_class="subcategory-row",
        )
    rows_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in required_rows:
        rows_by_leaf[row["leaf_id"]].append(row)
        rows_by_path[row["path_id"]].append(row)
    paths_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_paths:
        paths_by_leaf[row["leaf_id"]].append(row)

    category_comparisons: list[dict[str, Any]] = []
    path_comparisons: list[dict[str, Any]] = []
    subcategory_detail_by_mix: dict[str, str] = {}
    for mix_row in comparison_rows:
        mix_name = str(mix_row["mix_name"])
        mix_original = int(mix_row["available_uint32_values"])
        mix_target = int(mix_row["target_uint32_values"])
        category_sections: list[str] = []
        for category in sorted(categories_by_mix[mix_name], key=lambda row: int(row["category_index"])):
            leaf_id = category["leaf_id"]
            leaf_rows = rows_by_leaf[leaf_id]
            category_objects = {row["npy_uri"]: int(row["estimated_uint32_values"]) for row in leaf_rows}
            category_original = sum(category_objects.values())
            category_target = target_by_leaf[leaf_id]
            category_source_percent = 100 * category_original / mix_original if mix_original else 0.0
            category_target_percent = 100 * category_target / mix_target if mix_target else 0.0
            _, category_class, category_ratio = _sampling_change(category_original, category_target)
            category_sampling_rate, _ = _sampling_rate_label(category_original, category_target)
            category_comparisons.append(
                {
                    "leaf_id": leaf_id,
                    "mix_name": mix_name,
                    "category_name": category["category_name"],
                    "original_uint32_values": category_original,
                    "source_percent_of_parent": category_source_percent,
                    "target_uint32_values": category_target,
                    "target_percent_of_parent": category_target_percent,
                    "sampling_ratio": "" if category_ratio is None else f"{category_ratio:.12g}",
                    "sampling_rate": category_sampling_rate,
                }
            )
            path_definitions = sorted(paths_by_leaf[leaf_id], key=lambda row: row["path_id"])
            path_objects: list[dict[str, int]] = []
            for path in path_definitions:
                path_objects.append(
                    {row["npy_uri"]: int(row["estimated_uint32_values"]) for row in rows_by_path[path["path_id"]]}
                )
            path_originals = [sum(objects.values()) for objects in path_objects]
            path_targets = _apportion_by_size(category_target, path_originals)
            path_rows: list[str] = []
            for path, objects, path_original, path_target in zip(
                path_definitions, path_objects, path_originals, path_targets
            ):
                _, path_class, path_ratio = _sampling_change(path_original, path_target)
                path_sampling_rate, _ = _sampling_rate_label(path_original, path_target)
                path_source_percent = 100 * path_original / category_original if category_original else 0.0
                path_target_percent = 100 * path_target / category_target if category_target else 0.0
                path_comparisons.append(
                    {
                        "path_id": path["path_id"],
                        "leaf_id": leaf_id,
                        "mix_name": mix_name,
                        "category_name": category["category_name"],
                        "lower_group": _path_subgroup(path["yaml_path"]),
                        "yaml_path": path["yaml_path"],
                        "original_uint32_values": path_original,
                        "source_percent_of_parent": path_source_percent,
                        "implied_target_uint32_values": path_target,
                        "implied_target_percent_of_parent": path_target_percent,
                        "sampling_ratio": "" if path_ratio is None else f"{path_ratio:.12g}",
                        "sampling_rate": path_sampling_rate,
                        "unique_npy_count": len(objects),
                    }
                )
                source_details = _source_uri_details(objects, "No source NPYs resolved")
                path_rows.append(
                    '<details class="path-detail"><summary>'
                    f'<span class="path-name">{html.escape(_path_subgroup(path["yaml_path"]))}</span>'
                    f'<span class="path-stat">{_count_label(len(objects), "file")}'
                    '<span class="path-chevron" aria-hidden="true">›</span></span>'
                    '<span class="path-sampling">'
                    '<span class="path-metric"><span class="mix-metric-label">Source</span>'
                    f'<span class="mix-metric-value">{_human_token_count(path_original)} tokens · '
                    f"{path_source_percent:.2f}% of category</span></span>"
                    '<span class="path-metric"><span class="mix-metric-label">Target</span>'
                    f'<span class="mix-metric-value">{_human_token_count(path_target)} tokens · '
                    f"{path_target_percent:.2f}% of category</span></span>"
                    '<span class="path-metric"><span class="mix-metric-label">Sampling</span>'
                    f'<span class="mix-metric-value sampling {path_class}">'
                    f"{html.escape(path_sampling_rate)}</span></span>"
                    + _comparison_bars(path_original, path_target)
                    + "</span></summary>"
                    '<code><span class="path-detail-metrics">'
                    f'<span class="path-detail-metric">Source: {path_original:,} tokens</span>'
                    f'<span class="path-detail-metric">Implied target: {path_target:,} tokens</span>'
                    f"</span>{source_details}</code>"
                    "</details>"
                )
            category_sections.append(
                '<section class="category">'
                '<div class="category-head">'
                f'<span class="category-name">{html.escape(category["category_name"])}</span>'
                "</div>"
                '<div class="category-metrics">'
                '<span class="category-metric"><span class="mix-metric-label">Source</span>'
                f'<span class="mix-metric-value">{_human_token_count(category_original)} tokens · '
                f"{category_source_percent:.2f}% of entry</span></span>"
                '<span class="category-metric"><span class="mix-metric-label">Target</span>'
                f'<span class="mix-metric-value">{_human_token_count(category_target)} tokens · '
                f"{category_target_percent:.2f}% of entry</span></span>"
                '<span class="category-metric"><span class="mix-metric-label">Sampling</span>'
                f'<span class="mix-metric-value sampling {category_class}">'
                f"{html.escape(category_sampling_rate)}</span></span></div>"
                + _comparison_bars(category_original, category_target)
                + '<div class="path-list">'
                + "".join(path_rows)
                + "</div></section>"
            )
        subcategory_detail_by_mix[mix_name] = (
            f'<section class="subcategory-detail" id="{mix_row["detail_id"]}" hidden>'
            '<div class="detail-head">'
            f'<h2>{html.escape(str(mix_row["subcategory_name"]))}</h2>'
            f'<div class="detail-total">source {_human_token_count(mix_original)} → '
            f"target {_human_token_count(mix_target)}<br>"
            f'<span class="sampling {mix_row["sampling_class"]}">'
            f'{html.escape(str(mix_row["sampling_rate"]))}</span></div></div>'
            '<div class="category-grid">' + "".join(category_sections) + "</div></section>"
        )

    detail_sections: list[str] = []
    for family_row in family_rows:
        family_source = int(family_row["available_uint32_values"])
        family_target = int(family_row["target_uint32_values"])
        detail_sections.append(
            f'<section class="mix-detail" id="{family_row["detail_id"]}" hidden>'
            '<div class="detail-head">'
            f'<h2>{html.escape(str(family_row["mix_name"]))}</h2>'
            f'<div class="detail-total">source {_human_token_count(family_source)} → '
            f"target {_human_token_count(family_target)}<br>"
            f'<span class="sampling {family_row["sampling_class"]}">'
            f'{html.escape(str(family_row["sampling_rate"]))}</span></div></div>'
            '<div class="subcategory-list">'
            + str(family_row["subcategory_chart_rows"])
            + "</div>"
            + "".join(
                subcategory_detail_by_mix[str(subcategory["mix_name"])]
                for subcategory in family_row["subcategories"]
            )
            + "</section>"
        )

    _write_csv(
        plot_data / "sampling-by-category.csv",
        category_comparisons,
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "original_uint32_values",
            "source_percent_of_parent",
            "target_uint32_values",
            "target_percent_of_parent",
            "sampling_ratio",
            "sampling_rate",
        ],
    )
    _write_csv(
        plot_data / "sampling-by-lower-group.csv",
        path_comparisons,
        [
            "path_id",
            "leaf_id",
            "mix_name",
            "category_name",
            "lower_group",
            "yaml_path",
            "original_uint32_values",
            "source_percent_of_parent",
            "implied_target_uint32_values",
            "implied_target_percent_of_parent",
            "sampling_ratio",
            "sampling_rate",
            "unique_npy_count",
        ],
    )
    aggregate_rate, _ = _sampling_rate_label(original_total, target_total)
    sampling_rate_failures = [row for row in sampling_rate_rows if row["status"] == "above_expected_maximum"]
    sampling_audit_html = ""
    if sampling_rate_failures:
        configured_maximum = sampling_rate_failures[0]["maximum_expected_upsample_rate"]
        sampling_audit_html = (
            '<section class="audit-warning"><h2>Source-size consistency check failed</h2>'
            f"<p>{len(sampling_rate_failures):,} categories exceed the configured "
            f"{float(configured_maximum):.2f}× expected maximum. Proposal generation is "
            "blocked until the resolved source objects agree with the mix's source-size basis. "
            "Exact values are in <code>sampling-rate-audit.csv</code>.</p><ul>"
            + "".join(
                "<li>"
                f"{html.escape(str(row['mix_name']))} / "
                f"{html.escape(str(row['category_name']))}: "
                f"{float(row['sample_rate']):.2f}× "
                f"({_human_token_count(int(row['original_uint32_values']))} original → "
                f"{_human_token_count(int(row['target_uint32_values']))} target)"
                "</li>"
                for row in sampling_rate_failures
            )
            + "</ul></section>"
        )
    report_html = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 sampling plan</title>'
        + _interactive_report_style()
        + "</head><body><h1>Dolma 3.5 Source Inventory and Sampling Plan</h1>"
        + _summary_metrics(
            [
                ("Source tokens", _human_token_count(original_total)),
                ("Target tokens", _human_token_count(target_total)),
                ("Overall sampling", aggregate_rate),
            ]
        )
        + sampling_audit_html
        + f'<div class="mix-chart">{chart_rows}</div>'
        + "".join(detail_sections)
        + _interactive_report_script()
        + "</body></html>\n"
    )
    _write_text(phase / "report.html", report_html)
    details = _build_inventory_details(
        normalized_mix=normalized_mix,
        normalized_paths=normalized_paths,
        required_rows=required_rows,
    )
    _write_json(phase / "inventory-details.json", details)
    return {
        "source_family_count": details["source_family_count"],
        "subcategory_count": details["subcategory_count"],
        "source_count": details["source_count"],
        "category_count": details["category_count"],
        "lower_group_count": details["lower_group_count"],
        "details_artifact": "inventory-details.json",
        "report_artifact": "../report.html",
    }


def _render_report(
    build: Path,
    phase: Path,
    allocations: Sequence[dict[str, Any]],
    object_uses: Sequence[dict[str, Any]],
    inventory: Sequence[dict[str, str]],
    execution_units: Sequence[dict[str, Any]],
    category_execution: Sequence[dict[str, Any]],
) -> dict[str, int]:
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plot_data.mkdir(exist_ok=False)
    normalized_mix = _read_csv(build / "01-plan/resolution/normalized-mix.csv")
    largest_units = sorted(
        execution_units,
        key=lambda row: int(row["estimated_peak_local_bytes"]),
        reverse=True,
    )[:40]
    _write_csv(
        plot_data / "largest-execution-units.csv",
        largest_units,
        [
            "unit_id",
            "leaf_id",
            "mix_name",
            "category_name",
            "planned_uint32_values",
            "input_npy_bytes",
            "input_metadata_bytes",
            "output_npy_bytes",
            "estimated_output_metadata_bytes",
            "estimated_selection_index_bytes",
            "estimated_peak_local_bytes",
            "max_unit_working_bytes",
            "working_budget_utilization",
        ],
    )
    _write_text(
        plots / "largest-execution-units.svg",
        _svg_bar_chart(
            "Largest estimated execution-unit working sets",
            [row["unit_id"] for row in largest_units],
            [int(row["estimated_peak_local_bytes"]) / 1e9 for row in largest_units],
            "GB",
        ),
    )
    by_mix: dict[str, dict[str, Any]] = defaultdict(lambda: {"target": 0, "planned": 0})
    for row in allocations:
        entry = by_mix[row["mix_name"]]
        entry["target"] += int(row["target_uint32_values"])
        entry["planned"] += int(row["planned_uint32_values"])
    top_targets = sorted(by_mix.items(), key=lambda item: item[1]["target"], reverse=True)[:40]
    target_rows = [
        {
            "mix_name": name,
            "target_uint32_values": values["target"],
            "planned_uint32_values": values["planned"],
        }
        for name, values in top_targets
    ]
    target_total = sum(values["target"] for values in by_mix.values())
    for row in target_rows:
        row["target_percent"] = (
            f"{100 * int(row['target_uint32_values']) / target_total:.8f}" if target_total else "0"
        )
    _write_csv(
        plot_data / "target-by-mix.csv",
        target_rows,
        [
            "mix_name",
            "target_uint32_values",
            "planned_uint32_values",
            "target_percent",
        ],
    )
    _write_text(
        plots / "target-mix.svg",
        _svg_bar_chart(
            "Largest target mix entries by share of target",
            [name for name, _ in top_targets],
            [float(row["target_percent"]) for row in target_rows],
            "% of target",
            value_labels=[
                _mix_plot_value_label(int(row["target_uint32_values"]), target_total) for row in target_rows
            ],
            summary=(f"Total target: {_human_token_count(target_total)} tokens " f"({target_total:,})"),
        ),
    )
    _write_csv(
        plot_data / "proposal-target-residuals.csv",
        allocations,
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "target_uint32_values",
            "planned_uint32_values",
            "target_residual_uint32_values",
            "target_residual_bps_of_total",
            "target_residual_fraction",
        ],
    )
    normalized_paths = _read_csv(build / "01-plan/resolution/normalized-paths.csv")
    allocation_by_leaf = {row["leaf_id"]: row for row in allocations}
    inventory_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inventory_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        inventory_by_leaf[row["leaf_id"]].append(row)
        inventory_by_path[row["path_id"]].append(row)
    uses_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uses_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in object_uses:
        uses_by_leaf[row["leaf_id"]].append(row)
        uses_by_path[row["path_id"]].append(row)
    categories_by_mix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_mix:
        categories_by_mix[row["mix_name"]].append(row)
    for row in normalized_paths:
        paths_by_leaf[row["leaf_id"]].append(row)

    source_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"original": 0, "target": 0, "planned": 0})
    original_by_leaf: dict[str, int] = {}
    for category in normalized_mix:
        leaf_id = category["leaf_id"]
        original_objects = {
            row["npy_uri"]: int(row["estimated_uint32_values"]) for row in inventory_by_leaf[leaf_id]
        }
        original = sum(original_objects.values())
        planned = sum(int(row["planned_uint32_values"]) for row in uses_by_leaf[leaf_id])
        original_by_leaf[leaf_id] = original
        entry = source_totals[category["mix_name"]]
        entry["original"] += original
        entry["target"] += int(category["target_uint32_values"])
        entry["planned"] += planned

    original_total = sum(entry["original"] for entry in source_totals.values())
    proposed_total = sum(entry["planned"] for entry in source_totals.values())
    comparison_rows: list[dict[str, Any]] = []
    for mix_name, totals in sorted(
        source_totals.items(),
        key=lambda item: (item[1]["planned"], item[1]["target"], item[0]),
        reverse=True,
    ):
        effective = totals["planned"] / totals["original"] if totals["original"] else 0.0
        source_percent = 100 * totals["original"] / original_total if original_total else 0.0
        planned_percent = 100 * totals["planned"] / proposed_total if proposed_total else 0.0
        target_percent = 100 * totals["target"] / target_total if target_total else 0.0
        sampling_rate, sampling_class = _sampling_rate_label(totals["original"], totals["planned"])
        comparison_rows.append(
            {
                "mix_name": mix_name,
                "planned_uint32_values": totals["planned"],
                "planned_percent": f"{planned_percent:.8f}",
                "metric_columns": [
                    {
                        "label": "Source",
                        "value": (f"{_human_token_count(totals['original'])} tokens · " f"{source_percent:.2f}%"),
                    },
                    {
                        "label": "Proposed",
                        "value": (f"{_human_token_count(totals['planned'])} tokens · " f"{planned_percent:.2f}%"),
                    },
                    {
                        "label": "Target",
                        "value": (f"{_human_token_count(totals['target'])} tokens · " f"{target_percent:.2f}%"),
                    },
                    {"label": "Sampling", "value": sampling_rate},
                ],
                "original": totals["original"],
                "target": totals["target"],
                "planned": totals["planned"],
                "sampling_class": sampling_class,
                "effective": effective,
                "sampling_rate": sampling_rate,
            }
        )

    subcategories_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        source_family, subcategory = _split_mix_name(str(row["mix_name"]))
        row["source_family"] = source_family
        row["subcategory_name"] = subcategory
        row["display_name"] = subcategory
        subcategories_by_family[source_family].append(row)

    family_rows: list[dict[str, Any]] = []
    for source_family, subcategories in subcategories_by_family.items():
        family_source = sum(int(row["original"]) for row in subcategories)
        family_proposed = sum(int(row["planned"]) for row in subcategories)
        family_target = sum(int(row["target"]) for row in subcategories)
        source_percent = 100 * family_source / original_total if original_total else 0.0
        proposed_percent = 100 * family_proposed / proposed_total if proposed_total else 0.0
        target_percent = 100 * family_target / target_total if target_total else 0.0
        sampling_rate, sampling_class = _sampling_rate_label(family_source, family_proposed)
        ordered_subcategories = sorted(
            subcategories,
            key=lambda row: (
                int(row["planned"]),
                int(row["target"]),
                str(row["subcategory_name"]),
            ),
            reverse=True,
        )
        for row in ordered_subcategories:
            sub_source = int(row["original"])
            sub_proposed = int(row["planned"])
            sub_target = int(row["target"])
            sub_source_percent = 100 * sub_source / family_source if family_source else 0.0
            sub_proposed_percent = 100 * sub_proposed / family_proposed if family_proposed else 0.0
            sub_target_percent = 100 * sub_target / family_target if family_target else 0.0
            row["family_planned_percent"] = f"{sub_proposed_percent:.8f}"
            row["metric_columns"] = [
                {
                    "label": "Source",
                    "value": (
                        f"{_human_token_count(sub_source)} tokens · " f"{sub_source_percent:.2f}% of source"
                    ),
                },
                {
                    "label": "Proposed",
                    "value": (
                        f"{_human_token_count(sub_proposed)} tokens · " f"{sub_proposed_percent:.2f}% of proposed"
                    ),
                },
                {
                    "label": "Target",
                    "value": (
                        f"{_human_token_count(sub_target)} tokens · " f"{sub_target_percent:.2f}% of target"
                    ),
                },
                {"label": "Sampling", "value": row["sampling_rate"]},
            ]
        family_rows.append(
            {
                "mix_name": source_family,
                "planned_uint32_values": family_proposed,
                "planned_percent": f"{proposed_percent:.8f}",
                "original": family_source,
                "target": family_target,
                "planned": family_proposed,
                "sampling_rate": sampling_rate,
                "sampling_class": sampling_class,
                "subcategories": ordered_subcategories,
                "metric_columns": [
                    {
                        "label": "Source",
                        "value": (f"{_human_token_count(family_source)} tokens · " f"{source_percent:.2f}%"),
                    },
                    {
                        "label": "Proposed",
                        "value": (f"{_human_token_count(family_proposed)} tokens · " f"{proposed_percent:.2f}%"),
                    },
                    {
                        "label": "Target",
                        "value": (f"{_human_token_count(family_target)} tokens · " f"{target_percent:.2f}%"),
                    },
                    {"label": "Sampling", "value": sampling_rate},
                ],
            }
        )
    family_rows.sort(
        key=lambda row: (
            int(row["planned"]),
            int(row["target"]),
            str(row["mix_name"]),
        ),
        reverse=True,
    )
    chart_rows = _interactive_chart_rows(
        family_rows,
        "planned_uint32_values",
        "planned_percent",
        "proposal-family-detail",
        proposed_total,
    )
    for family_index, family_row in enumerate(family_rows):
        family_row["subcategory_chart_rows"] = _interactive_chart_rows(
            family_row["subcategories"],
            "planned_uint32_values",
            "family_planned_percent",
            f"proposal-subcategory-{family_index}",
            int(family_row["planned"]),
            row_class="subcategory-row",
        )

    category_sampling_rows: list[dict[str, Any]] = []
    path_sampling_rows: list[dict[str, Any]] = []
    subcategory_detail_by_mix: dict[str, str] = {}
    for mix_row in comparison_rows:
        mix_name = str(mix_row["mix_name"])
        category_sections: list[str] = []
        for category in sorted(categories_by_mix[mix_name], key=lambda row: int(row["category_index"])):
            leaf_id = category["leaf_id"]
            original = original_by_leaf[leaf_id]
            target = int(category["target_uint32_values"])
            planned = sum(int(row["planned_uint32_values"]) for row in uses_by_leaf[leaf_id])
            allocation = allocation_by_leaf.get(leaf_id)
            if allocation is None:
                minimum_repetition = 0
                maximum_repetition = 0
                total_object_uses = 0
                unique_object_count = len({row["npy_uri"] for row in inventory_by_leaf[leaf_id]})
                selected_object_count = 0
                dropped_object_count = unique_object_count
                repeated_object_count = 0
                partial_object_count = 0
            else:
                minimum_repetition = int(allocation["minimum_repetition"])
                maximum_repetition = int(allocation["maximum_repetition"])
                total_object_uses = int(allocation["total_object_uses"])
                unique_object_count = int(allocation["unique_object_count"])
                selected_object_count = int(allocation["selected_object_count"])
                dropped_object_count = int(allocation["dropped_object_count"])
                repeated_object_count = int(allocation["repeated_object_count"])
                partial_object_count = int(allocation["partial_object_count"])
            effective = planned / original if original else 0.0
            sampling_label, sampling_class = _sampling_rate_label(original, planned)
            category_sampling_rows.append(
                {
                    "leaf_id": leaf_id,
                    "mix_name": mix_name,
                    "category_name": category["category_name"],
                    "original_uint32_values": original,
                    "target_uint32_values": target,
                    "proposed_uint32_values": planned,
                    "token_change_from_original": planned - original,
                    "effective_repetitions": f"{effective:.12g}",
                    "minimum_repetition": minimum_repetition,
                    "maximum_repetition": maximum_repetition,
                    "total_object_uses": total_object_uses,
                    "unique_object_count": unique_object_count,
                    "selected_object_count": selected_object_count,
                    "dropped_object_count": dropped_object_count,
                    "repeated_object_count": repeated_object_count,
                    "partial_object_count": partial_object_count,
                }
            )
            path_rows: list[str] = []
            for path in sorted(paths_by_leaf[leaf_id], key=lambda row: row["path_id"]):
                path_original_objects = {
                    row["npy_uri"]: int(row["estimated_uint32_values"])
                    for row in inventory_by_path[path["path_id"]]
                }
                path_original = sum(path_original_objects.values())
                path_uses = uses_by_path[path["path_id"]]
                path_planned = sum(int(row["planned_uint32_values"]) for row in path_uses)
                path_repetitions = [int(row["repeat_count"]) for row in path_uses]
                path_partials = [int(row.get("partial_target_uint32_values", 0)) for row in path_uses]
                if not path_repetitions:
                    path_repetitions = [0 for _ in path_original_objects]
                    path_partials = [0 for _ in path_original_objects]
                path_minimum = min(path_repetitions, default=0)
                path_maximum = max(path_repetitions, default=0)
                path_partial_count = sum(value > 0 for value in path_partials)
                path_total_uses = sum(path_repetitions) + path_partial_count
                path_selected = sum(
                    repeat > 0 or partial > 0 for repeat, partial in zip(path_repetitions, path_partials)
                )
                path_dropped = sum(
                    repeat == 0 and partial == 0 for repeat, partial in zip(path_repetitions, path_partials)
                )
                path_repeated = sum(
                    int(row["planned_uint32_values"]) > int(row["estimated_uint32_values"]) for row in path_uses
                )
                path_effective = path_planned / path_original if path_original else 0.0
                path_sampling_rate, path_sampling_class = _sampling_rate_label(path_original, path_planned)
                path_sampling_rows.append(
                    {
                        "path_id": path["path_id"],
                        "leaf_id": leaf_id,
                        "mix_name": mix_name,
                        "category_name": category["category_name"],
                        "lower_group": _path_subgroup(path["yaml_path"]),
                        "yaml_path": path["yaml_path"],
                        "original_uint32_values": path_original,
                        "proposed_uint32_values": path_planned,
                        "token_change_from_original": path_planned - path_original,
                        "effective_repetitions": f"{path_effective:.12g}",
                        "minimum_repetition": path_minimum,
                        "maximum_repetition": path_maximum,
                        "total_object_uses": path_total_uses,
                        "unique_object_count": len(path_original_objects),
                        "selected_object_count": path_selected,
                        "dropped_object_count": path_dropped,
                        "repeated_object_count": path_repeated,
                        "partial_object_count": path_partial_count,
                    }
                )
                repeat_range = (
                    f"{path_minimum}×" if path_minimum == path_maximum else f"{path_minimum}–{path_maximum}×"
                )
                source_details = _source_uri_details(
                    (
                        row["npy_uri"]
                        for row in path_uses
                        if int(row["repeat_count"]) > 0 or int(row.get("partial_target_uint32_values", 0)) > 0
                    ),
                    "No source NPYs selected for materialization",
                )
                path_rows.append(
                    '<details class="path-detail"><summary>'
                    f'<span class="path-name">{html.escape(_path_subgroup(path["yaml_path"]))}</span>'
                    f'<span class="path-stat">repeat {repeat_range}'
                    '<span class="path-chevron" aria-hidden="true">›</span></span>'
                    '<span class="path-sampling">'
                    '<span class="path-metric"><span class="mix-metric-label">Source</span>'
                    f'<span class="mix-metric-value">{_human_token_count(path_original)} tokens</span></span>'
                    '<span class="path-metric"><span class="mix-metric-label">Proposed</span>'
                    f'<span class="mix-metric-value">{_human_token_count(path_planned)} tokens</span></span>'
                    '<span class="path-metric"><span class="mix-metric-label">Sampling</span>'
                    f'<span class="mix-metric-value sampling {path_sampling_class}">'
                    f"{html.escape(path_sampling_rate)}</span></span>"
                    f'<span class="path-use-summary">Partial selections: {path_partial_count:,} · '
                    f"NPYs repeated: {path_repeated:,} · dropped: {path_dropped:,} · "
                    f"{path_total_uses:,} total object uses</span>"
                    + _comparison_bars(path_original, path_planned)
                    + "</span></summary>"
                    '<code><span class="path-detail-metrics">'
                    f'<span class="path-detail-metric">Source: {path_original:,} tokens</span>'
                    f'<span class="path-detail-metric">Proposed: {path_planned:,} tokens</span>'
                    f'<span class="path-detail-metric">Repeats: {repeat_range}</span>'
                    f"</span>{source_details}</code></details>"
                )
            category_sections.append(
                '<section class="category">'
                '<div class="category-head">'
                f'<span class="category-name">{html.escape(category["category_name"])}</span>'
                "</div>"
                '<div class="category-metrics proposal-metrics">'
                '<span class="category-metric"><span class="mix-metric-label">Source</span>'
                f'<span class="mix-metric-value">{_human_token_count(original)} tokens</span></span>'
                '<span class="category-metric"><span class="mix-metric-label">Proposed</span>'
                f'<span class="mix-metric-value">{_human_token_count(planned)} tokens</span></span>'
                '<span class="category-metric"><span class="mix-metric-label">Target</span>'
                f'<span class="mix-metric-value">{_human_token_count(target)} tokens</span></span>'
                '<span class="category-metric"><span class="mix-metric-label">Sampling</span>'
                f'<span class="mix-metric-value sampling {sampling_class}">'
                f"{html.escape(sampling_label)}</span></span></div>"
                + _comparison_bars(original, planned)
                + '<div class="path-list">'
                + "".join(path_rows)
                + "</div></section>"
            )
        subcategory_detail_by_mix[mix_name] = (
            f'<section class="subcategory-detail" id="{mix_row["detail_id"]}" hidden>'
            '<div class="detail-head">'
            f'<h2>{html.escape(str(mix_row["subcategory_name"]))}</h2>'
            f'<div class="detail-total">{_human_token_count(int(mix_row["original"]))} source → '
            f'{_human_token_count(int(mix_row["planned"]))} proposed<br>'
            f'target {_human_token_count(int(mix_row["target"]))} · '
            f'<span class="sampling {mix_row["sampling_class"]}">'
            f'{html.escape(str(mix_row["sampling_rate"]))}</span></div></div>'
            '<div class="category-grid">' + "".join(category_sections) + "</div></section>"
        )

    detail_sections: list[str] = []
    for family_row in family_rows:
        detail_sections.append(
            f'<section class="mix-detail" id="{family_row["detail_id"]}" hidden>'
            '<div class="detail-head">'
            f'<h2>{html.escape(str(family_row["mix_name"]))}</h2>'
            f'<div class="detail-total">{_human_token_count(int(family_row["original"]))} source → '
            f'{_human_token_count(int(family_row["planned"]))} proposed<br>'
            f'target {_human_token_count(int(family_row["target"]))} · '
            f'<span class="sampling {family_row["sampling_class"]}">'
            f'{html.escape(str(family_row["sampling_rate"]))}</span></div></div>'
            '<div class="subcategory-list">'
            + str(family_row["subcategory_chart_rows"])
            + "</div>"
            + "".join(
                subcategory_detail_by_mix[str(subcategory["mix_name"])]
                for subcategory in family_row["subcategories"]
            )
            + "</section>"
        )

    _write_csv(
        plot_data / "proposed-sampling-by-category.csv",
        category_sampling_rows,
        list(category_sampling_rows[0]) if category_sampling_rows else [],
    )
    _write_csv(
        plot_data / "proposed-sampling-by-lower-group.csv",
        path_sampling_rows,
        list(path_sampling_rows[0]) if path_sampling_rows else [],
    )
    report = _render_execution_proposal_html(
        execution_units=execution_units,
        category_execution=category_execution,
        planned_total=proposed_total,
    )
    _write_text(phase / "report.html", report)
    return {
        "source_uint32_values": original_total,
        "proposed_uint32_values": proposed_total,
        "target_uint32_values": target_total,
    }


def validate_build(args: argparse.Namespace) -> None:
    build = args.build.resolve()
    manifest = _load_build(build)
    checks: list[tuple[str, bool, str]] = []
    plan_failures = _read_csv(build / "01-plan/resolution/resolution-failures.csv")
    checks.append(("plan_failures", not plan_failures, str(len(plan_failures))))
    inventory_summary_path = build / "01-plan/inventory/inventory-summary.json"
    checks.append(
        (
            "inventory_exists",
            inventory_summary_path.is_file(),
            str(inventory_summary_path),
        )
    )
    if inventory_summary_path.is_file():
        with inventory_summary_path.open() as f:
            summary = json.load(f)
        for name in (
            "missing_objects",
            "path_resolution_failures",
            "invalid_npy_sizes",
            "head_errors",
            "sampling_rate_limit_failures",
        ):
            value = summary.get(
                name,
                summary.get("direct_resolution_failures", 0) if name == "path_resolution_failures" else 0,
            )
            checks.append((name, int(value) == 0, str(value)))
    validation_path = build / "01-plan/execution/validation-summary.json"
    checks.append(("execution_exists", validation_path.is_file(), str(validation_path)))
    if validation_path.is_file():
        with validation_path.open() as f:
            proposal = json.load(f)
        checks.append(("execution_passed", bool(proposal["passed"]), str(proposal)))
        try:
            layout = _validate_execution_layout(build)
            checks.append(("execution_layout_current", True, str(layout)))
        except PreparationError as exc:
            checks.append(("execution_layout_current", False, str(exc)))
    combined_report_path = build / "01-plan/report.html"
    checks.append(("combined_plan_report_exists", combined_report_path.is_file(), str(combined_report_path)))
    checks.append(
        (
            "stage_reports_consolidated",
            not (build / "01-plan/resolution/report.html").exists()
            and not (build / "01-plan/inventory/report.html").exists()
            and not (build / "01-plan/execution/report.html").exists(),
            "individual stage reports must not remain after composition",
        )
    )
    preflight_path = build / "02-preflight/preflight-summary.json"
    if preflight_path.is_file():
        with preflight_path.open() as f:
            preflight = json.load(f)
        checks.append(("preflight_passed", bool(preflight["passed"]), str(preflight)))
    output_path = build / "03-output-validation/output-summary.json"
    if output_path.is_file():
        with output_path.open() as f:
            output = json.load(f)
        checks.append(("output_validation_passed", bool(output["passed"]), str(output)))
    failed = [check for check in checks if not check[1]]
    print(
        json.dumps(
            {"build_id": manifest["build_id"], "checks": checks, "passed": not failed},
            indent=2,
        )
    )
    if failed:
        raise PreparationError(f"Build has {len(failed)} failed validation check(s)")
