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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import unquote, urlparse

import boto3
import yaml


UINT32_BYTES = 4
DEFAULT_TARGET = 14_000_000_000_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_PATH = REPOSITORY_ROOT / "runs/dolma3p5-resharding/14t"
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
}


class PreparationError(RuntimeError):
    """A user-actionable preparation or validation failure."""


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


def _slug(value: str, max_length: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.").lower()
    if not value:
        value = "unnamed"
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{value[: max_length - 9]}-{suffix}"


def _exclusive_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PreparationError(
            f"Refusing to replace existing output directory: {path}"
        ) from exc


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(value)
    except FileExistsError as exc:
        raise PreparationError(
            f"Refusing to replace existing artifact: {path}"
        ) from exc


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("x", newline="", encoding="utf-8")
    except FileExistsError as exc:
        raise PreparationError(
            f"Refusing to replace existing artifact: {path}"
        ) from exc
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
    return settings


def _load_catalog(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise PreparationError(
                    f"Catalog row {line_number} has fewer than two columns"
                )
            bucket, key = row[0].strip(), row[1].strip()
            if (
                line_number == 1
                and bucket.lower() == "bucket"
                and key.lower() in {"key", "path"}
            ):
                continue
            if not bucket:
                raise PreparationError(
                    f"Invalid catalog object on row {line_number}: {row[:2]}"
                )
            if any(ord(char) < 32 for char in bucket + key):
                raise PreparationError(
                    f"Control character in catalog object on row {line_number}"
                )
            if not key.endswith(".npy"):
                # The reference file is nominally an NPY catalog but currently
                # contains at least one metadata row. Metadata never defines
                # membership, so ignore it here and verify the derived partner
                # against S3 during inventory.
                continue
            rows.append(
                {"bucket": bucket, "key": key, "catalog_line": str(line_number)}
            )
    if not rows:
        raise PreparationError(f"Catalog contains no NPY objects: {path}")
    return rows


def _catalog_pattern(yaml_path: str) -> str:
    relative = yaml_path.removeprefix("dolma3p5_pool/")
    return (
        relative if relative.startswith("preprocessed/") else f"preprocessed/{relative}"
    )


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
    _exclusive_directory(output)
    phase = output / "01-plan"
    _exclusive_directory(phase)

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
                    else "direct_s3"
                    if yaml_path.startswith("preprocessed/")
                    else "unsupported"
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
                        "resolution_route": resolution_route,
                        "yaml_path": yaml_path,
                    }
                )
                if not active:
                    continue
                if resolution_route == "catalog":
                    pattern = _catalog_pattern(yaml_path)
                    matched = [
                        row for row in catalog if _matches_key(row["key"], pattern)
                    ]
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
                                "yaml_path": yaml_path,
                                "bucket": match["bucket"],
                                "key": match["key"],
                                "catalog_line": match["catalog_line"],
                            }
                        )
                elif resolution_route == "direct_s3":
                    bucket, pattern = _direct_s3_pattern(
                        yaml_path, str(settings["direct_s3_bucket"])
                    )
                    direct_patterns.append(
                        {
                            "path_id": path_id,
                            "leaf_id": leaf_id,
                            "mix_index": mix_index,
                            "mix_name": mix_name,
                            "category_index": category_index,
                            "category_name": category_name,
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
            estimated = sum(
                1
                for row in catalog_by_bucket[bucket]
                if row["key"].startswith(subgroup_prefix)
            )
            overfetch = estimated / max(1, len(required))
            if (
                estimated > int(settings["max_listing_catalog_objects"])
                or overfetch > float(settings["max_listing_overfetch_ratio"])
            ) and len(subgroup) > 1:
                final_groups = [[row] for row in subgroup]
            else:
                final_groups = [subgroup]
            for final_group in final_groups:
                final_prefix = _common_directory_prefix(
                    [row["key"] for row in final_group]
                )
                key = (bucket, final_prefix)
                entry = listing_groups.setdefault(
                    key,
                    {
                        "bucket": bucket,
                        "listing_prefix": final_prefix,
                        "resolution_routes": set(),
                        "leaf_ids": set(),
                    },
                )
                entry["resolution_routes"].add("catalog")
                entry["leaf_ids"].update(row["leaf_id"] for row in final_group)

    for row in direct_patterns:
        key = (row["bucket"], row["listing_prefix"])
        entry = listing_groups.setdefault(
            key,
            {
                "bucket": row["bucket"],
                "listing_prefix": row["listing_prefix"],
                "resolution_routes": set(),
                "leaf_ids": set(),
            },
        )
        entry["resolution_routes"].add("direct_s3")
        entry["leaf_ids"].add(row["leaf_id"])

    listing_plan: list[dict[str, Any]] = []
    for listing_id, ((bucket, prefix), entry) in enumerate(
        sorted(listing_groups.items())
    ):
        estimated = sum(
            1 for row in catalog_by_bucket[bucket] if row["key"].startswith(prefix)
        )
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
                "resolution_routes": ";".join(sorted(entry["resolution_routes"])),
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
    _write_json(output / "build.json", manifest)
    _write_csv(phase / "normalized-mix.csv", normalized_mix, list(normalized_mix[0]))
    _write_csv(
        phase / "normalized-paths.csv", normalized_paths, list(normalized_paths[0])
    )
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
        "ls --etag --storage-class "
        + shlex.quote("s3://" + row["bucket"] + "/" + row["listing_prefix"] + "*")
        for row in listing_plan
    ]
    _write_text(
        phase / "bulk-listing-commands.txt",
        "\n".join(commands) + ("\n" if commands else ""),
    )
    inventory_command = "python scripts/dolma3p5_resharding/inventory.py"
    if output != DEFAULT_BUILD_PATH.resolve():
        inventory_command += f" --build {shlex.quote(str(output))}"
    _write_text(
        phase / "NEXT-STEPS.txt",
        "This phase made no AWS calls.\n\n"
        "Run the read-only inventory collector from the repository root:\n"
        f"  {inventory_command}\n",
    )
    _render_plan_report(
        phase,
        normalized_mix=normalized_mix,
        catalog_matches=catalog_matches,
        direct_patterns=direct_patterns,
        failures=failures,
        corrections=corrections,
    )

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


def collect_inventory(args: argparse.Namespace) -> None:
    build = args.build.resolve()
    manifest = _load_build(build)
    phase = build / "02-inventory"
    _exclusive_directory(phase)
    listing_plan = _read_csv(build / "01-plan/listing-plan.csv")

    session = (
        boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    )
    client = session.client("s3", region_name=args.region)
    max_workers = args.max_workers or int(manifest["settings"]["inventory_max_workers"])
    listed: dict[tuple[str, str], S3Object] = {}
    errors: list[dict[str, str]] = []
    collector = "s5cmd" if shutil.which("s5cmd") else "boto3"
    if collector == "s5cmd":
        raw_output = phase / "raw-listings.jsonl"
        commands_path = build / "01-plan/bulk-listing-commands.txt"
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
        if args.region:
            environment["AWS_REGION"] = args.region
            environment["AWS_DEFAULT_REGION"] = args.region
        with raw_output.open("x") as stdout:
            result = subprocess.run(
                command,
                text=True,
                stdout=stdout,
                stderr=subprocess.PIPE,
                env=environment,
            )
        _write_json(
            phase / "collector.json",
            {
                "collector": collector,
                "command": command,
                "raw_output": raw_output.name,
                "read_only": True,
            },
        )
        if result.returncode != 0:
            _write_text(phase / "collector-error.txt", result.stderr)
            raise PreparationError(
                f"Read-only S3 listing failed; inspect {phase / 'collector-error.txt'}"
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
    else:
        _write_json(
            phase / "collector.json",
            {
                "collector": collector,
                "reason": "s5cmd executable was not found",
                "max_workers": max_workers,
                "read_only": True,
            },
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _list_prefix, client, row["bucket"], row["listing_prefix"]
                ): row
                for row in listing_plan
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    for obj in future.result():
                        listed[(obj.bucket, obj.key)] = obj
                except Exception as exc:
                    errors.append(
                        {
                            "operation": "list",
                            "bucket": row["bucket"],
                            "key": row["listing_prefix"],
                            "error": repr(exc),
                        }
                    )
    if errors:
        _write_csv(
            phase / "inventory-errors.csv",
            errors,
            ["operation", "bucket", "key", "error"],
        )
        raise PreparationError(
            f"S3 listing failed; inspect {phase / 'inventory-errors.csv'}"
        )
    _finalize_inventory(build, phase, listed, client=client, max_workers=max_workers)
    print(f"Created read-only S3 inventory: {phase}")


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
                errors.append(
                    {"line": str(line_number), "error": f"invalid JSON: {exc}"}
                )
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
            elif (
                not (isinstance(uri, str) and uri.startswith("s3://"))
                and bucket_value
                and key_value
            ):
                uri = f"s3://{bucket_value}/{key_value}"
            if (
                not isinstance(uri, str)
                or not uri.startswith("s3://")
                or any(c in uri for c in "*?[")
            ):
                continue
            size = _first_value(
                nodes, {"size", "size_bytes", "content_length", "contentlength"}
            )
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
                last_modified=str(
                    _first_value(nodes, {"last_modified", "lastmodified", "modtime"})
                    or ""
                ),
                storage_class=str(
                    _first_value(nodes, {"storage_class", "storageclass"}) or ""
                ),
                source="s5cmd",
            )
    return objects, errors


def _finalize_inventory(
    build: Path,
    phase: Path,
    listed: dict[tuple[str, str], S3Object],
    client: Any,
    max_workers: int,
) -> None:
    catalog_matches = _read_csv(build / "01-plan/catalog-matches.csv")
    direct_patterns = _read_csv(build / "01-plan/direct-s3-patterns.csv")
    membership: list[dict[str, Any]] = []
    resolution_failures: list[dict[str, str]] = []

    for row in catalog_matches:
        membership.append({**row, "resolution_route": "catalog"})
    for pattern in direct_patterns:
        matches = [
            obj
            for (bucket, key), obj in listed.items()
            if bucket == pattern["bucket"]
            and key.endswith(".npy")
            and _matches_key(key, pattern["key_pattern"])
        ]
        if not matches:
            resolution_failures.append(
                {
                    "path_id": pattern["path_id"],
                    "yaml_path": pattern["yaml_path"],
                    "reason": "direct S3 pattern returned no NPY objects",
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
                    "yaml_path": pattern["yaml_path"],
                    "bucket": obj.bucket,
                    "key": obj.key,
                    "catalog_line": "",
                    "resolution_route": "direct_s3",
                }
            )

    required_keys: set[tuple[str, str]] = set()
    for row in membership:
        required_keys.add((row["bucket"], row["key"]))
        required_keys.add((row["bucket"], _pair_metadata_key(row["key"])))

    missing = sorted(required_keys - set(listed))
    head_errors: list[dict[str, str]] = []
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_head_object, client, bucket, key): (bucket, key)
                for bucket, key in missing
            }
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
        missing = sorted(required_keys - set(listed))

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
        "yaml_path",
        "resolution_route",
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
    _write_csv(
        phase / "missing-objects.csv", missing_rows, ["bucket", "key", "object_type"]
    )
    _write_csv(
        phase / "invalid-npy-sizes.csv",
        invalid_sizes,
        ["bucket", "key", "size_bytes", "reason"],
    )
    _write_csv(
        phase / "direct-resolution-failures.csv",
        resolution_failures,
        ["path_id", "yaml_path", "reason"],
    )
    _write_csv(
        phase / "head-errors.csv", head_errors, ["operation", "bucket", "key", "error"]
    )
    _write_json(
        phase / "inventory-summary.json",
        {
            "created_at": _utc_now(),
            "listed_objects": len(listed),
            "required_membership_rows": len(required_rows),
            "unique_required_objects_including_metadata": len(required_keys),
            "missing_objects": len(missing),
            "direct_resolution_failures": len(resolution_failures),
            "invalid_npy_sizes": len(invalid_sizes),
            "head_errors": len(head_errors),
        },
    )
    _render_inventory_report(
        phase,
        required_rows=required_rows,
        all_objects=all_objects,
        missing_rows=missing_rows,
        resolution_failures=resolution_failures,
        invalid_sizes=invalid_sizes,
    )
    if resolution_failures or missing or invalid_sizes or head_errors:
        raise PreparationError(
            f"Inventory validation failed; inspect artifacts in {phase}"
        )


def _allocate_object_repetitions(
    target: int, sizes: Sequence[int]
) -> tuple[list[int], int]:
    if target <= 0 or not sizes or any(size <= 0 for size in sizes):
        raise ValueError(
            "Allocation requires a positive target and positive object sizes"
        )
    available = sum(sizes)
    base = target // available
    repetitions = [base for _ in sizes]
    residual = target - base * available
    selected: set[int] = set()
    remaining = residual

    for index in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i)):
        candidate = remaining - sizes[index]
        if abs(candidate) < abs(remaining):
            selected.add(index)
            remaining = candidate

    improved = True
    while improved:
        improved = False
        current_error = abs(remaining)
        unselected = [i for i in range(len(sizes)) if i not in selected]
        for remove in sorted(selected):
            for add in unselected:
                candidate = remaining + sizes[remove] - sizes[add]
                if abs(candidate) < current_error:
                    selected.remove(remove)
                    selected.add(add)
                    remaining = candidate
                    improved = True
                    break
            if improved:
                break

    if base == 0 and not selected:
        closest = min(
            range(len(sizes)), key=lambda i: (abs(sizes[i] - target), sizes[i], i)
        )
        selected.add(closest)
    for index in selected:
        repetitions[index] += 1
    planned = sum(size * repeat for size, repeat in zip(sizes, repetitions))
    return repetitions, planned


def _execution_unit_sizes(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    input_npy_bytes = sum(int(row["npy_size_bytes"]) for row in rows)
    input_metadata_bytes = sum(int(row["metadata_size_bytes"]) for row in rows)
    output_npy_bytes = sum(
        int(row["npy_size_bytes"]) * int(row["repeat_count"]) for row in rows
    )
    estimated_output_metadata_bytes = sum(
        int(row["metadata_size_bytes"]) * int(row["repeat_count"]) for row in rows
    )
    return {
        "input_npy_bytes": input_npy_bytes,
        "input_metadata_bytes": input_metadata_bytes,
        "output_npy_bytes": output_npy_bytes,
        "estimated_output_metadata_bytes": estimated_output_metadata_bytes,
        "estimated_peak_local_bytes": input_npy_bytes
        + input_metadata_bytes
        + output_npy_bytes
        + estimated_output_metadata_bytes,
    }


def _partition_object_uses(
    rows: Sequence[dict[str, Any]], max_unit_working_bytes: int
) -> list[list[dict[str, Any]]]:
    """Partition one category into deterministic worker-sized execution units."""

    if max_unit_working_bytes <= 0:
        raise ValueError("max_unit_working_bytes must be positive")
    positive = [dict(row) for row in rows if int(row["repeat_count"]) > 0]
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
        input_bytes = int(row["npy_size_bytes"]) + int(row["metadata_size_bytes"])
        output_bytes_per_repeat = input_bytes
        maximum_repeats_alone = (
            max_unit_working_bytes - input_bytes
        ) // output_bytes_per_repeat
        if maximum_repeats_alone < 1:
            raise PreparationError(
                "One source object cannot fit in an execution unit with one output copy: "
                f"{row['npy_uri']} requires at least {input_bytes + output_bytes_per_repeat} bytes, "
                f"limit is {max_unit_working_bytes}"
            )

        whole_row = dict(row)
        whole_row["repeat_count"] = remaining
        if (
            _execution_unit_sizes([whole_row])["estimated_peak_local_bytes"]
            <= max_unit_working_bytes
        ):
            candidate = [*current, whole_row]
            if (
                current
                and _execution_unit_sizes(candidate)["estimated_peak_local_bytes"]
                > max_unit_working_bytes
            ):
                flush()
            current.append(whole_row)
            continue

        flush()
        while remaining:
            repeat_count = min(remaining, maximum_repeats_alone)
            chunk = dict(row)
            chunk["repeat_count"] = repeat_count
            current.append(chunk)
            remaining -= repeat_count
            if remaining:
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

unit_root=$(mktemp -d "${{TMPDIR:-/tmp}}/dolma3p5-{unit_id}.XXXXXX")
cleanup() {{ rm -rf -- "$unit_root"; }}
trap cleanup EXIT
mkdir -p "$unit_root/config" "$unit_root/manifests"

python - <<'PY'
from dolma.tokenizer.reshard import RESHARDING_MANIFEST_SCHEMA_VERSION

if RESHARDING_MANIFEST_SCHEMA_VERSION != 1:
    raise RuntimeError(
        "Worker Dolma runtime does not match the reviewed manifest-resharding schema"
    )
PY

python - "$unit_root/config/{config_name}" "$unit_root/manifests/{manifest_name}" <<'PY'
import base64
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode("{config_payload}", validate=True))
pathlib.Path(sys.argv[2]).write_bytes(base64.b64decode("{manifest_payload}", validate=True))
PY

python -m dolma.tokenizer.reshard "$unit_root/config/{config_name}"
"""


def _validate_destination_root(destination_root: str) -> str:
    parsed = urlparse(destination_root)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise PreparationError("destination-root must be an s3:// URI")
    prefix = parsed.path.strip("/")
    components = prefix.split("/")
    if len(components) < 2:
        raise PreparationError(
            "destination-root must contain at least two path components below the bucket"
        )
    if any(component in {"", ".", ".."} for component in components):
        raise PreparationError(
            "destination-root cannot contain empty, '.' or '..' components"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", prefix):
        raise PreparationError(
            "destination-root must use only letters, digits, '.', '_', '-', and '/'"
        )
    return f"s3://{parsed.netloc}/{prefix}"


def propose_configs(args: argparse.Namespace) -> None:
    build = args.build.resolve()
    manifest = _load_build(build)
    inventory_phase = build / "02-inventory"
    if not (inventory_phase / "inventory-summary.json").is_file():
        raise PreparationError(
            "Inventory is missing; run scripts/dolma3p5_resharding/inventory.py first"
        )
    with (inventory_phase / "inventory-summary.json").open() as f:
        inventory_summary = json.load(f)
    blocking = sum(
        int(inventory_summary[name])
        for name in (
            "missing_objects",
            "direct_resolution_failures",
            "invalid_npy_sizes",
            "head_errors",
        )
    )
    if blocking:
        raise PreparationError("Inventory contains blocking validation failures")

    phase = build / "03-proposal"
    _exclusive_directory(phase)
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

    normalized_mix = _read_csv(build / "01-plan/normalized-mix.csv")
    inventory = _read_csv(inventory_phase / "required-objects.csv")
    by_leaf: dict[str, dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in inventory:
        object_id = (row["bucket"], row["key"])
        by_leaf[row["leaf_id"]][object_id] = row
        memberships[object_id].add(row["leaf_id"])
    overlaps = [
        {"bucket": bucket, "key": key, "leaf_ids": ";".join(sorted(leaves))}
        for (bucket, key), leaves in memberships.items()
        if len(leaves) > 1
    ]
    _write_csv(
        phase / "cross-leaf-overlaps.csv", overlaps, ["bucket", "key", "leaf_ids"]
    )
    if overlaps:
        raise PreparationError(
            f"Found {len(overlaps)} exact NPY object(s) assigned to multiple active categories; inspect cross-leaf-overlaps.csv"
        )

    allocation_rows: list[dict[str, Any]] = []
    object_use_rows: list[dict[str, Any]] = []
    config_index: list[dict[str, Any]] = []
    category_execution_rows: list[dict[str, Any]] = []
    local_unit_commands: list[str] = []

    settings = manifest["settings"]
    total_planned = 0
    manifest_fields = [
        "npy_uri",
        "metadata_uri",
        "repeat_count",
        "npy_size_bytes",
        "estimated_uint32_values",
        "npy_etag",
        "metadata_size_bytes",
        "metadata_etag",
        "leaf_id",
        "mix_name",
        "category_name",
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
            raise PreparationError(
                f"Active category has no inventoried objects: {leaf['leaf_id']}"
            )
        sizes = [int(row["estimated_uint32_values"]) for row in objects]
        target = int(leaf["target_uint32_values"])
        repetitions, planned = _allocate_object_repetitions(target, sizes)
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
                "ideal_sample_rate": f"{target / available:.12g}",
                "effective_sample_rate": f"{planned / available:.12g}",
                "absolute_error": planned - target,
                "absolute_error_bps_of_total": f"{10_000 * (planned - target) / int(settings['target_uint32_values']):.12g}",
                "relative_error": f"{(planned - target) / target:.12g}",
                "unique_object_count": len(objects),
                "maximum_repetition": max(repetitions),
            }
        )
        leaf_object_uses: list[dict[str, Any]] = []
        for obj, repeat_count in zip(objects, repetitions):
            object_use = {
                "leaf_id": leaf["leaf_id"],
                "mix_index": mix_index,
                "mix_name": mix_name,
                "category_index": category_index,
                "category_name": category_name,
                "npy_uri": obj["npy_uri"],
                "metadata_uri": obj["metadata_uri"],
                "npy_size_bytes": obj["npy_size_bytes"],
                "estimated_uint32_values": obj["estimated_uint32_values"],
                "npy_etag": obj["npy_etag"],
                "metadata_size_bytes": obj["metadata_size_bytes"],
                "metadata_etag": obj["metadata_etag"],
                "repeat_count": repeat_count,
                "planned_uint32_values": int(obj["estimated_uint32_values"])
                * repeat_count,
            }
            object_use_rows.append(object_use)
            if repeat_count > 0:
                leaf_object_uses.append(object_use)

        units = _partition_object_uses(leaf_object_uses, max_unit_working_bytes)
        category_slug = (
            f"{mix_index:03d}-{_slug(mix_name, 60)}--"
            f"{category_index:02d}-{_slug(category_name, 40)}"
        )
        unit_planned_values = [
            _execution_unit_sizes(unit)["output_npy_bytes"] // UINT32_BYTES
            for unit in units
        ]
        unit_targets: list[int] = []
        remaining_target = target
        for unit_index, unit_planned in enumerate(unit_planned_values):
            unit_target = (
                remaining_target
                if unit_index == len(units) - 1
                else round(target * unit_planned / planned)
            )
            unit_targets.append(unit_target)
            remaining_target -= unit_target

        unit_peak_bytes: list[int] = []
        unit_input_bytes: list[int] = []
        for unit_index, (unit_rows, unit_target) in enumerate(zip(units, unit_targets)):
            unit_number = unit_index + 1
            unit_id = f"{category_slug}--unit-{unit_number:04d}-of-{len(units):04d}"
            unit_sizes = _execution_unit_sizes(unit_rows)
            unit_peak_bytes.append(unit_sizes["estimated_peak_local_bytes"])
            unit_input_bytes.append(
                unit_sizes["input_npy_bytes"] + unit_sizes["input_metadata_bytes"]
            )
            unit_planned = unit_sizes["output_npy_bytes"] // UINT32_BYTES
            unit_max_repeat = max(int(row["repeat_count"]) for row in unit_rows)
            manifest_path = manifests_dir / f"{unit_id}.csv"
            _write_csv(manifest_path, unit_rows, manifest_fields)
            if unit_planned < 10_000_000_000:
                shard_floor = 2
            elif unit_planned < 100_000_000_000:
                shard_floor = 4
            else:
                shard_floor = 8
            max_num_files = max(shard_floor, unit_max_repeat)
            destination = (
                f"{destination_root}/{manifest['build_id']}/categories/"
                f"{category_slug}/unit-{unit_number:04d}"
            )
            config = {
                "destination_prefix": destination,
                "source_manifests": [
                    {"manifest": f"../manifests/{manifest_path.name}"}
                ],
                "local_tempdir": str(local_temp_root / manifest["build_id"] / unit_id),
                "max_num_files": max_num_files,
                "max_workers": min(
                    int(settings["max_workers_per_reshard"]), max_num_files
                ),
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
                "unit_index": unit_number,
                "unit_count_for_category": len(units),
                "config_path": str(config_path.relative_to(build)),
                "manifest_path": str(manifest_path.relative_to(build)),
                "launcher_path": str(launcher_path.relative_to(build)),
                "destination_prefix": destination,
                "target_uint32_values": unit_target,
                "planned_uint32_values": unit_planned,
                "absolute_error": unit_planned - unit_target,
                "category_target_uint32_values": target,
                "category_planned_uint32_values": planned,
                "input_npy_bytes": unit_sizes["input_npy_bytes"],
                "input_metadata_bytes": unit_sizes["input_metadata_bytes"],
                "output_npy_bytes": unit_sizes["output_npy_bytes"],
                "estimated_output_metadata_bytes": unit_sizes[
                    "estimated_output_metadata_bytes"
                ],
                "estimated_peak_local_bytes": unit_sizes["estimated_peak_local_bytes"],
                "max_unit_working_bytes": max_unit_working_bytes,
                "working_budget_utilization": f"{unit_sizes['estimated_peak_local_bytes'] / max_unit_working_bytes:.12g}",
                "unique_object_count": len(unit_rows),
                "maximum_repetition": unit_max_repeat,
                "max_num_files": max_num_files,
            }
            config_index.append(unit_row)
            local_unit_commands.append(
                f"python -m dolma.tokenizer.reshard {shlex.quote(str(config_path))}"
            )

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
        "ideal_sample_rate",
        "effective_sample_rate",
        "absolute_error",
        "absolute_error_bps_of_total",
        "relative_error",
        "unique_object_count",
        "maximum_repetition",
    ]
    _write_csv(phase / "category-allocation.csv", allocation_rows, allocation_fields)
    _write_csv(
        phase / "planned-object-uses.csv", object_use_rows, list(object_use_rows[0])
    )
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
    _write_text(
        phase / "DISTRIBUTED-LAUNCH.txt",
        "# INERT EXAMPLE. Review the execution-unit plan and preflight before launching.\n"
        "# launcher-scripts contains only self-contained executable unit scripts.\n"
        f"pmr map --name YOUR_CLUSTER@YOUR_PROJECT --script {shlex.quote(str(launcher_dir))}\n",
    )
    _write_json(
        phase / "dataset-layout.json",
        {
            "schema_version": 1,
            "build_id": manifest["build_id"],
            "dataset_root": f"{destination_root}/{manifest['build_id']}/categories",
            "category_count": len(category_execution_rows),
            "execution_unit_count": len(config_index),
            "nominal_target_uint32_values": int(settings["target_uint32_values"]),
            "planned_uint32_values": total_planned,
            "max_unit_working_bytes": max_unit_working_bytes,
            "destination_prefixes_file": "dataset-prefixes.txt",
        },
    )
    _write_json(
        phase / "runtime-requirements.json",
        {
            "required_python_module": "dolma.tokenizer.reshard",
            "required_resharding_manifest_schema_version": 1,
            "launcher_runtime_check": True,
        },
    )
    _write_text(
        phase / "dataset-prefixes.txt",
        "\n".join(row["destination_prefix"] for row in config_index) + "\n",
    )

    _validate_proposal(build, phase, config_index, allocation_rows, object_use_rows)
    _render_report(
        build,
        phase,
        allocation_rows,
        object_use_rows,
        inventory,
        config_index,
        category_execution_rows,
    )
    _write_json(
        phase / "proposal-summary.json",
        {
            "created_at": _utc_now(),
            "build_id": manifest["build_id"],
            "destination_root": destination_root,
            "category_count": len(category_execution_rows),
            "execution_unit_count": len(config_index),
            "config_count": len(config_index),
            "max_unit_working_bytes": max_unit_working_bytes,
            "largest_estimated_peak_local_bytes": max(
                int(row["estimated_peak_local_bytes"]) for row in config_index
            ),
            "target_uint32_values": int(settings["target_uint32_values"]),
            "planned_uint32_values": total_planned,
            "absolute_error": total_planned - int(settings["target_uint32_values"]),
            "materialization_executed": False,
        },
    )
    print(f"Created proposal without materializing data: {phase}")


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
    for row in config_index:
        config_path = build / row["config_path"]
        with config_path.open() as f:
            config = yaml.safe_load(f)
        if config.get("allow_existing_destination") is not False:
            failures.append(
                {"check": "no_existing_destination", "detail": str(config_path)}
            )
        manifest_path = config_path.parent / config["source_manifests"][0]["manifest"]
        if not manifest_path.resolve().is_file():
            failures.append({"check": "manifest_exists", "detail": str(manifest_path)})
        launcher_path = build / row["launcher_path"]
        if not launcher_path.is_file() or not os.access(launcher_path, os.X_OK):
            failures.append(
                {"check": "launcher_is_executable", "detail": str(launcher_path)}
            )
        if int(row["estimated_peak_local_bytes"]) > int(row["max_unit_working_bytes"]):
            failures.append(
                {
                    "check": "unit_within_working_budget",
                    "detail": row["unit_id"],
                }
            )
    active_leaf_ids = {
        row["leaf_id"]
        for row in _read_csv(build / "01-plan/normalized-mix.csv")
        if row["active"] == "true"
    }
    allocated_leaf_ids = {row["leaf_id"] for row in allocations}
    for leaf_id in sorted(active_leaf_ids - allocated_leaf_ids):
        failures.append({"check": "active_leaf_allocated", "detail": leaf_id})
    units_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in config_index:
        units_by_leaf[row["leaf_id"]].append(row)
    allocation_by_leaf = {row["leaf_id"]: row for row in allocations}
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
            failures.append(
                {"check": "unit_planned_sum", "detail": f"{leaf_id}: {planned}"}
            )
        if target != int(allocation["target_uint32_values"]):
            failures.append(
                {"check": "unit_target_sum", "detail": f"{leaf_id}: {target}"}
            )
    planned_uses = {(row["leaf_id"], row["npy_uri"]) for row in object_uses}
    required_uses = {
        (row["leaf_id"], row["npy_uri"])
        for row in _read_csv(build / "02-inventory/required-objects.csv")
    }
    for leaf_id, uri in sorted(required_uses - planned_uses):
        failures.append(
            {"check": "required_object_considered", "detail": f"{leaf_id}: {uri}"}
        )
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
        raise PreparationError(
            f"Proposal validation failed; inspect {phase / 'validation-failures.csv'}"
        )


def preflight_build(args: argparse.Namespace) -> None:
    """Re-inventory approved inputs and verify that all destinations are empty."""

    build = args.build.resolve()
    manifest = _load_build(build)
    proposal_summary = build / "03-proposal/proposal-summary.json"
    if not proposal_summary.is_file():
        raise PreparationError("Proposal is missing; run propose first")
    phase = build / "04-preflight"
    _exclusive_directory(phase)
    listing_plan = _read_csv(build / "01-plan/listing-plan.csv")
    approved_inventory = _read_csv(build / "02-inventory/normalized-s3-inventory.csv")
    approved = {
        (row["bucket"], row["key"]): row
        for row in approved_inventory
        if row["required"] == "true"
    }

    session = (
        boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    )
    client = session.client("s3", region_name=args.region)
    max_workers = args.max_workers or int(manifest["settings"]["inventory_max_workers"])
    current: dict[tuple[str, str], S3Object] = {}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_list_prefix, client, row["bucket"], row["listing_prefix"]): row
            for row in listing_plan
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
            futures = {
                pool.submit(_head_object, client, bucket, key): (bucket, key)
                for bucket, key in missing
            }
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
        elif expected["last_modified"] and _normalize_timestamp(
            actual.last_modified
        ) != _normalize_timestamp(expected["last_modified"]):
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

    config_index = _read_csv(build / "03-proposal/config-index.csv")
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
            "destination_prefix",
            "status",
            "first_existing_key",
        ],
    )
    _write_csv(
        phase / "preflight-errors.csv", errors, ["operation", "bucket", "key", "error"]
    )
    drifted = sum(row["status"] != "unchanged" for row in drift_rows)
    occupied = sum(row["status"] != "empty" for row in destination_rows)
    passed = (
        not errors
        and not drifted
        and not occupied
        and len(destination_rows) == len(config_index)
    )
    summary = {
        "created_at": _utc_now(),
        "approved_input_objects": len(approved),
        "drifted_input_objects": drifted,
        "destinations_checked": len(destination_rows),
        "occupied_destinations": occupied,
        "errors": len(errors),
        "passed": passed,
        "materialization_executed": False,
    }
    _write_json(phase / "preflight-summary.json", summary)
    status_rows = [
        {"state": "unchanged inputs", "count": len(drift_rows) - drifted},
        {"state": "drifted inputs", "count": drifted},
        {"state": "empty destinations", "count": len(destination_rows) - occupied},
        {"state": "occupied destinations", "count": occupied},
        {"state": "request errors", "count": len(errors)},
    ]
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plots.mkdir(exist_ok=False)
    plot_data.mkdir(exist_ok=False)
    _write_csv(plot_data / "preflight-counts.csv", status_rows, ["state", "count"])
    _write_text(
        plots / "preflight-counts.svg",
        _svg_bar_chart(
            "Preflight input and destination status",
            [row["state"] for row in status_rows],
            [row["count"] for row in status_rows],
            "items",
        ),
    )
    _write_text(
        phase / "report.html",
        '<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 preflight</title></head><body>'
        "<h1>Pre-materialization preflight</h1>"
        f"<p>Passed: {str(passed).lower()}. This command made read-only S3 requests and did not materialize data.</p>"
        '<img src="plots/preflight-counts.svg" alt="Preflight status counts">'
        "</body></html>\n",
    )
    if not passed:
        raise PreparationError(f"Preflight failed; inspect artifacts in {phase}")
    print(f"Preflight passed without materializing data: {phase}")


def verify_output(args: argparse.Namespace) -> None:
    """Verify materialized outputs using only S3 object listings and sizes."""

    build = args.build.resolve()
    manifest = _load_build(build)
    config_index = _read_csv(build / "03-proposal/config-index.csv")
    if not config_index:
        raise PreparationError("Proposal config index is empty or missing")
    phase = build / "05-output-validation"
    _exclusive_directory(phase)
    session = (
        boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    )
    client = session.client("s3", region_name=args.region)
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
            unexpected = [
                obj for obj in objects if not obj.key.endswith((".npy", ".csv.gz"))
            ]
            missing_metadata = [
                _pair_metadata_key(obj.key)
                for obj in npys
                if _pair_metadata_key(obj.key) not in metadata
            ]
            orphan_metadata = [
                key
                for key in metadata
                if key[: -len(".csv.gz")] + ".npy" not in object_map
            ]
            invalid_npys = [
                obj
                for obj in npys
                if obj.size_bytes <= 0 or obj.size_bytes % UINT32_BYTES
            ]
            actual = sum(obj.size_bytes // UINT32_BYTES for obj in npys)
            predicted = int(row["planned_uint32_values"])
            status = (
                "passed"
                if npys
                and not missing_metadata
                and not orphan_metadata
                and not invalid_npys
                and not unexpected
                and actual == predicted
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
                        "object_type": "npy"
                        if obj.key.endswith(".npy")
                        else "metadata"
                        if obj.key.endswith(".csv.gz")
                        else "unexpected",
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
    _write_csv(
        phase / "output-errors.csv", errors, ["operation", "bucket", "key", "error"]
    )
    failed = sum(row["status"] != "passed" for row in validation_rows)
    passed = not errors and not failed and len(validation_rows) == len(config_index)
    _write_json(
        phase / "output-summary.json",
        {
            "created_at": _utc_now(),
            "destinations_expected": len(config_index),
            "destinations_checked": len(validation_rows),
            "failed_destinations": failed,
            "errors": len(errors),
            "target_uint32_values": sum(
                int(row["target_uint32_values"]) for row in validation_rows
            ),
            "predicted_uint32_values": sum(
                int(row["predicted_uint32_values"]) for row in validation_rows
            ),
            "actual_uint32_values": sum(
                int(row["actual_uint32_values"]) for row in validation_rows
            ),
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
            "Predicted versus actual materialized sizes",
            points,
            "Predicted (B uint32)",
            "Actual (B uint32)",
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
        "<p>Actual quantities are output NPY bytes divided by four. No arrays or metadata rows were read.</p>"
        '<img src="plots/predicted-vs-actual.svg" alt="Predicted versus actual sizes">'
        '<img src="plots/output-status.svg" alt="Output validation status">'
        "</body></html>\n",
    )
    if not passed:
        raise PreparationError(
            f"Output validation failed; inspect artifacts in {phase}"
        )
    print(f"Output validation passed using size-only checks: {phase}")


def _svg_bar_chart(
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    unit: str,
    width: int = 1100,
) -> str:
    row_height = 24
    margin_left = 360
    margin_right = 140
    height = 70 + row_height * len(labels)
    plot_width = width - margin_left - margin_right
    maximum = max(values, default=1.0) or 1.0
    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 50 + index * row_height
        bar_width = max(0.0, plot_width * value / maximum)
        rows.append(
            f'<text x="{margin_left - 8}" y="{y + 14}" text-anchor="end">{html.escape(label[:52])}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="16" />'
            f'<text x="{margin_left + bar_width + 6:.2f}" y="{y + 14}">{value:.4g} {html.escape(unit)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title)}"><title>{html.escape(title)}</title><style>'
        "text{font:12px sans-serif;fill:#222}rect{fill:#356cb6}</style>"
        + "".join(rows)
        + "</svg>\n"
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


def _render_plan_report(
    phase: Path,
    normalized_mix: Sequence[dict[str, Any]],
    catalog_matches: Sequence[dict[str, Any]],
    direct_patterns: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    corrections: Sequence[dict[str, Any]],
) -> None:
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plots.mkdir(exist_ok=False)
    plot_data.mkdir(exist_ok=False)
    target_by_mix: dict[str, int] = defaultdict(int)
    for row in normalized_mix:
        target_by_mix[row["mix_name"]] += int(row["target_uint32_values"])
    target_rows = [
        {"mix_name": name, "target_uint32_values": value}
        for name, value in sorted(
            target_by_mix.items(), key=lambda item: item[1], reverse=True
        )[:40]
    ]
    resolution_rows = [
        {"state": "catalog-resolved NPYs", "count": len(catalog_matches)},
        {"state": "direct S3 patterns", "count": len(direct_patterns)},
        {"state": "explicit corrections", "count": len(corrections)},
        {"state": "blocking failures", "count": len(failures)},
    ]
    _write_csv(
        plot_data / "target-mix.csv",
        target_rows,
        ["mix_name", "target_uint32_values"],
    )
    _write_csv(plot_data / "resolution-counts.csv", resolution_rows, ["state", "count"])
    _write_text(
        plots / "target-mix.svg",
        _svg_bar_chart(
            "Largest target mix entries",
            [row["mix_name"] for row in target_rows],
            [row["target_uint32_values"] / 1e9 for row in target_rows],
            "B uint32",
        ),
    )
    _write_text(
        plots / "resolution-counts.svg",
        _svg_bar_chart(
            "Path-resolution planning counts",
            [row["state"] for row in resolution_rows],
            [row["count"] for row in resolution_rows],
            "items",
        ),
    )
    _write_text(
        phase / "report.html",
        '<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 path plan</title></head><body>'
        "<h1>Path-resolution checkpoint</h1>"
        "<p>The YAML defines membership. Catalog counts include only active YAML matches; direct S3 patterns remain pending live inventory.</p>"
        '<img src="plots/target-mix.svg" alt="Largest target mix entries">'
        '<img src="plots/resolution-counts.svg" alt="Path resolution counts">'
        "</body></html>\n",
    )


def _render_inventory_report(
    phase: Path,
    required_rows: Sequence[dict[str, Any]],
    all_objects: Sequence[dict[str, Any]],
    missing_rows: Sequence[dict[str, Any]],
    resolution_failures: Sequence[dict[str, Any]],
    invalid_sizes: Sequence[dict[str, Any]],
) -> None:
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plots.mkdir(exist_ok=False)
    plot_data.mkdir(exist_ok=False)
    found_npy = {(row["bucket"], row["key"]) for row in required_rows}
    found_metadata = {
        (row["bucket"], _pair_metadata_key(row["key"])) for row in required_rows
    }
    missing_npy = sum(row["object_type"] == "npy" for row in missing_rows)
    missing_metadata = sum(row["object_type"] == "metadata" for row in missing_rows)
    coverage_rows = [
        {"state": "found NPYs", "count": len(found_npy)},
        {"state": "missing NPYs", "count": missing_npy},
        {"state": "found metadata", "count": len(found_metadata)},
        {"state": "missing metadata", "count": missing_metadata},
        {"state": "direct pattern failures", "count": len(resolution_failures)},
        {"state": "invalid NPY sizes", "count": len(invalid_sizes)},
    ]
    by_mix: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    for row in required_rows:
        identity = (row["leaf_id"], row["npy_uri"])
        if identity in seen:
            continue
        seen.add(identity)
        by_mix[row["mix_name"]] += int(row["estimated_uint32_values"])
    available_rows = [
        {"mix_name": name, "available_uint32_values": value}
        for name, value in sorted(
            by_mix.items(), key=lambda item: item[1], reverse=True
        )[:40]
    ]
    unique_sizes = {
        row["key"]: int(row["size_bytes"])
        for row in all_objects
        if row["required"] == "true" and row["key"].endswith(".npy")
    }
    sizes = list(unique_sizes.values())
    bins = [0] * 20
    maximum = max(sizes, default=0)
    if maximum:
        for size in sizes:
            bins[min(19, int(20 * size / maximum))] += 1
    size_rows = [
        {
            "bin_start_bytes": round(index * maximum / 20),
            "bin_end_bytes": round((index + 1) * maximum / 20),
            "object_count": count,
        }
        for index, count in enumerate(bins)
    ]
    _write_csv(plot_data / "coverage.csv", coverage_rows, ["state", "count"])
    _write_csv(
        plot_data / "available-by-mix.csv",
        available_rows,
        ["mix_name", "available_uint32_values"],
    )
    _write_csv(
        plot_data / "object-size-bins.csv",
        size_rows,
        ["bin_start_bytes", "bin_end_bytes", "object_count"],
    )
    _write_text(
        plots / "coverage.svg",
        _svg_bar_chart(
            "S3 inventory coverage",
            [row["state"] for row in coverage_rows],
            [row["count"] for row in coverage_rows],
            "items",
        ),
    )
    _write_text(
        plots / "available-by-mix.svg",
        _svg_bar_chart(
            "Largest available mix entries",
            [row["mix_name"] for row in available_rows],
            [row["available_uint32_values"] / 1e9 for row in available_rows],
            "B uint32",
        ),
    )
    _write_text(
        plots / "object-size-histogram.svg",
        _svg_bar_chart(
            "Required NPY object-size histogram",
            [
                f"{row['bin_start_bytes'] / 1e9:.1f}–{row['bin_end_bytes'] / 1e9:.1f} GB"
                for row in size_rows
            ],
            [row["object_count"] for row in size_rows],
            "objects",
        ),
    )
    _write_text(
        phase / "report.html",
        '<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 inventory</title></head><body>'
        "<h1>S3 inventory checkpoint</h1>"
        "<p>All available quantities are NPY bytes divided by four. Metadata sizes are excluded.</p>"
        '<img src="plots/coverage.svg" alt="Inventory coverage">'
        '<img src="plots/available-by-mix.svg" alt="Available values by mix">'
        '<img src="plots/object-size-histogram.svg" alt="NPY object size histogram">'
        "</body></html>\n",
    )


def _render_report(
    build: Path,
    phase: Path,
    allocations: Sequence[dict[str, Any]],
    object_uses: Sequence[dict[str, Any]],
    inventory: Sequence[dict[str, str]],
    execution_units: Sequence[dict[str, Any]],
    category_execution: Sequence[dict[str, Any]],
) -> None:
    plots = phase / "plots"
    plot_data = phase / "plot-data"
    plot_data.mkdir(exist_ok=False)
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
    most_split_categories = sorted(
        category_execution,
        key=lambda row: int(row["execution_unit_count"]),
        reverse=True,
    )[:40]
    _write_csv(
        plot_data / "most-split-categories.csv",
        most_split_categories,
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "target_uint32_values",
            "planned_uint32_values",
            "execution_unit_count",
            "largest_estimated_peak_local_bytes",
            "max_unit_working_bytes",
        ],
    )
    _write_text(
        plots / "execution-units-per-category.svg",
        _svg_bar_chart(
            "Most partitioned categories",
            [
                f"{row['mix_name']} / {row['category_name']}"
                for row in most_split_categories
            ],
            [int(row["execution_unit_count"]) for row in most_split_categories],
            "units",
        ),
    )
    by_mix: dict[str, dict[str, Any]] = defaultdict(lambda: {"target": 0, "planned": 0})
    for row in allocations:
        entry = by_mix[row["mix_name"]]
        entry["target"] += int(row["target_uint32_values"])
        entry["planned"] += int(row["planned_uint32_values"])
    top_targets = sorted(
        by_mix.items(), key=lambda item: item[1]["target"], reverse=True
    )[:40]
    target_rows = [
        {
            "mix_name": name,
            "target_uint32_values": values["target"],
            "planned_uint32_values": values["planned"],
        }
        for name, values in top_targets
    ]
    _write_csv(
        plot_data / "target-by-mix.csv",
        target_rows,
        ["mix_name", "target_uint32_values", "planned_uint32_values"],
    )
    _write_text(
        plots / "target-mix.svg",
        _svg_bar_chart(
            "Largest target mix entries",
            [name for name, _ in top_targets],
            [values["target"] / 1e9 for _, values in top_targets],
            "B uint32",
        ),
    )
    points = [
        (
            int(row["target_uint32_values"]) / 1e9,
            int(row["planned_uint32_values"]) / 1e9,
            row["leaf_id"],
        )
        for row in allocations
    ]
    _write_csv(
        plot_data / "target-vs-proposed.csv",
        [
            {
                "leaf_id": row["leaf_id"],
                "mix_name": row["mix_name"],
                "category_name": row["category_name"],
                "target_uint32_values": row["target_uint32_values"],
                "planned_uint32_values": row["planned_uint32_values"],
            }
            for row in allocations
        ],
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "target_uint32_values",
            "planned_uint32_values",
        ],
    )
    _write_text(
        plots / "target-vs-proposed.svg",
        _svg_scatter(
            "Target versus proposed", points, "Target (B uint32)", "Proposed (B uint32)"
        ),
    )
    residuals = sorted(
        allocations, key=lambda row: abs(int(row["absolute_error"])), reverse=True
    )[:40]
    _write_csv(
        plot_data / "largest-absolute-residuals.csv",
        residuals,
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "target_uint32_values",
            "planned_uint32_values",
            "absolute_error",
            "absolute_error_bps_of_total",
        ],
    )
    _write_text(
        plots / "largest-absolute-residuals.svg",
        _svg_bar_chart(
            "Largest absolute allocation residuals",
            [f"{row['mix_name']} / {row['category_name']}" for row in residuals],
            [abs(int(row["absolute_error"])) / 1e6 for row in residuals],
            "M uint32",
        ),
    )
    pressure = sorted(
        allocations, key=lambda row: float(row["ideal_sample_rate"]), reverse=True
    )[:40]
    _write_csv(
        plot_data / "upsampling-pressure.csv",
        pressure,
        [
            "leaf_id",
            "mix_name",
            "category_name",
            "target_uint32_values",
            "available_uint32_values",
            "ideal_sample_rate",
            "effective_sample_rate",
        ],
    )
    _write_text(
        plots / "upsampling-pressure.svg",
        _svg_bar_chart(
            "Largest requested/available ratios",
            [f"{row['mix_name']} / {row['category_name']}" for row in pressure],
            [float(row["ideal_sample_rate"]) for row in pressure],
            "x",
        ),
    )
    sizes = list(
        {row["npy_uri"]: int(row["npy_size_bytes"]) for row in inventory}.values()
    )
    if sizes:
        bins = [0] * 20
        maximum = max(sizes)
        for size in sizes:
            bins[min(19, int(20 * size / max(1, maximum)))] += 1
        size_rows = [
            {
                "bin_start_bytes": round(index * maximum / 20),
                "bin_end_bytes": round((index + 1) * maximum / 20),
                "object_count": count,
            }
            for index, count in enumerate(bins)
        ]
        _write_csv(
            plot_data / "object-size-bins.csv",
            size_rows,
            ["bin_start_bytes", "bin_end_bytes", "object_count"],
        )
        _write_text(
            plots / "object-size-histogram.svg",
            _svg_bar_chart(
                "NPY object-size histogram",
                [
                    f"{i * maximum / 20 / 1e9:.1f}–{(i + 1) * maximum / 20 / 1e9:.1f} GB"
                    for i in range(20)
                ],
                bins,
                "objects",
            ),
        )

    report_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['mix_name'])}</td><td>{html.escape(row['category_name'])}</td>"
        f"<td>{int(row['target_uint32_values']):,}</td><td>{int(row['available_uint32_values']):,}</td>"
        f"<td>{int(row['planned_uint32_values']):,}</td><td>{int(row['absolute_error']):,}</td>"
        f"<td>{float(row['ideal_sample_rate']):.4f}×</td></tr>"
        for row in sorted(
            allocations, key=lambda item: abs(int(item["absolute_error"])), reverse=True
        )
    )
    report = f"""<!doctype html><html><head><meta charset="utf-8"><title>Dolma 3.5 14T proposal</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ccc;padding:6px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}</style></head><body>
<h1>Dolma 3.5 14T pre-materialization proposal</h1>
<p>All quantities are estimated from NPY byte sizes divided by four. No arrays were opened and no content-level counts were performed.</p>
<p>The dataset is materialized as {len(execution_units):,} independent execution units across {len(category_execution):,} active categories. No worker is assigned the full dataset.</p>
<h2>Largest execution units</h2><img src="plots/largest-execution-units.svg" alt="Largest estimated execution-unit working sets">
<h2>Units per category</h2><img src="plots/execution-units-per-category.svg" alt="Most partitioned categories">
<h2>Target mix</h2><img src="plots/target-mix.svg" alt="Largest target mix entries">
<h2>Target versus proposed</h2><img src="plots/target-vs-proposed.svg" alt="Target versus proposed scatter plot">
<h2>Largest residuals</h2><img src="plots/largest-absolute-residuals.svg" alt="Largest absolute residuals">
<h2>Upsampling pressure</h2><img src="plots/upsampling-pressure.svg" alt="Largest upsampling ratios">
<h2>Object sizes</h2><img src="plots/object-size-histogram.svg" alt="NPY object size histogram">
<h2>Category details</h2><table><thead><tr><th>Mix</th><th>Category</th><th>Target</th><th>Available</th><th>Proposed</th><th>Error</th><th>Requested/available</th></tr></thead><tbody>{report_rows}</tbody></table>
</body></html>\n"""
    _write_text(phase / "report.html", report)


def validate_build(args: argparse.Namespace) -> None:
    build = args.build.resolve()
    manifest = _load_build(build)
    checks: list[tuple[str, bool, str]] = []
    plan_failures = _read_csv(build / "01-plan/resolution-failures.csv")
    checks.append(("plan_failures", not plan_failures, str(len(plan_failures))))
    inventory_summary_path = build / "02-inventory/inventory-summary.json"
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
            "direct_resolution_failures",
            "invalid_npy_sizes",
            "head_errors",
        ):
            checks.append((name, int(summary[name]) == 0, str(summary[name])))
    validation_path = build / "03-proposal/validation-summary.json"
    checks.append(("proposal_exists", validation_path.is_file(), str(validation_path)))
    if validation_path.is_file():
        with validation_path.open() as f:
            proposal = json.load(f)
        checks.append(("proposal_passed", bool(proposal["passed"]), str(proposal)))
    preflight_path = build / "04-preflight/preflight-summary.json"
    if preflight_path.is_file():
        with preflight_path.open() as f:
            preflight = json.load(f)
        checks.append(("preflight_passed", bool(preflight["passed"]), str(preflight)))
    output_path = build / "05-output-validation/output-summary.json"
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
