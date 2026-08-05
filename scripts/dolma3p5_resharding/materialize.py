"""Select and dispatch reviewed Dolma 3.5 materialization units with poormanray."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

scripts_root = Path(__file__).resolve().parents[1]
if str(scripts_root) not in sys.path:
    sys.path.insert(0, str(scripts_root))

from resharding.dispatch import (
    PoormanrayDispatchError,
    build_poormanray_map_command,
    require_spindown_coverage,
)

try:
    from .workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        PreparationError,
        _human_byte_count,
        _human_token_count,
        _filter_execution_units,
        _unit_selection_digest,
        _validate_execution_layout,
        _validate_preparation_build,
        normalize_region,
    )
except ImportError:
    from workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        PreparationError,
        _human_byte_count,
        _human_token_count,
        _filter_execution_units,
        _unit_selection_digest,
        _validate_execution_layout,
        _validate_preparation_build,
        normalize_region,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=DEFAULT_BUILD_PATH,
        help="local preparation build directory",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="select every execution unit",
    )
    selection.add_argument(
        "--category",
        help="exact mix name, leaf ID, or MIX_NAME::CATEGORY_NAME selector",
    )
    selection.add_argument(
        "--unit",
        help="select one exact execution-unit ID",
    )
    selection.add_argument(
        "--list-categories",
        nargs="?",
        const="",
        metavar="FILTER",
        help="list category selectors, optionally filtered by a case-insensitive substring",
    )
    parser.add_argument(
        "--cluster",
        default="dolma3p5-14t",
        help="poormanray cluster name",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("PMR_REGION") or DEFAULT_REGION,
        help="poormanray region; PMR_REGION overrides the us-east-1 default",
    )
    parser.add_argument(
        "--project",
        help="optional poormanray project tag",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="stage and print the exact dispatch without invoking poormanray",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="invoke poormanray after requiring a passing preflight; otherwise the command is a dry run",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"Required execution artifact is missing or unsafe: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_execution_units(build: Path) -> list[dict[str, str]]:
    _validate_preparation_build(build)
    _validate_execution_layout(build)
    rows = _read_csv(build / "01-plan/execution/config-index.csv")
    if not rows:
        raise PreparationError("Execution plan contains no units")
    required = {
        "unit_id",
        "leaf_id",
        "mix_name",
        "category_name",
        "launcher_path",
        "destination_prefix",
        "planned_uint32_values",
        "estimated_peak_local_bytes",
    }
    missing = required - set(rows[0])
    if missing:
        raise PreparationError("Execution index is missing columns: " + ", ".join(sorted(missing)))
    return rows


def _category_selector(row: dict[str, str]) -> str:
    return f'{row["mix_name"]}::{row["category_name"]}'


def _category_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["leaf_id"], row["mix_name"], row["category_name"])].append(row)
    output: list[dict[str, Any]] = []
    for (leaf_id, mix_name, category_name), units in grouped.items():
        output.append(
            {
                "leaf_id": leaf_id,
                "mix_name": mix_name,
                "category_name": category_name,
                "selector": f"{mix_name}::{category_name}",
                "execution_units": len(units),
                "planned_uint32_values": sum(int(row["planned_uint32_values"]) for row in units),
                "largest_estimated_peak_local_bytes": max(
                    int(row["estimated_peak_local_bytes"]) for row in units
                ),
            }
        )
    return sorted(output, key=lambda row: (int(row["leaf_id"].split(":", 1)[0]), row["leaf_id"]))


def _print_categories(rows: Sequence[dict[str, str]], filter_text: str) -> None:
    needle = filter_text.casefold()
    matches = [
        row
        for row in _category_rows(rows)
        if not needle
        or needle
        in " ".join(
            (row["leaf_id"], row["mix_name"], row["category_name"], row["selector"])
        ).casefold()
    ]
    if not matches:
        raise PreparationError(f"No categories match: {filter_text}")
    print("leaf_id\texecution_units\tplanned_tokens\tlargest_unit\tselector")
    for row in matches:
        print(
            f'{row["leaf_id"]}\t{row["execution_units"]:,}\t'
            f'{_human_token_count(int(row["planned_uint32_values"]))}\t'
            f'{_human_byte_count(int(row["largest_estimated_peak_local_bytes"]))}\t'
            f'{row["selector"]}'
        )


def _select_units(args: argparse.Namespace, rows: Sequence[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    if args.all:
        return "all", list(rows)
    if args.unit:
        selected = _filter_execution_units(rows, unit=args.unit)
        return f"unit-{args.unit}", selected

    selector = str(args.category)
    selected = _filter_execution_units(rows, category=selector)
    return f"category-{selector}", selected


def _safe_slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part)[:80] or "selection"


def _resolve_launcher(build: Path, row: dict[str, str]) -> Path:
    launcher = (build / row["launcher_path"]).resolve()
    launcher_root = (build / "01-plan/execution/launcher-scripts").resolve()
    if launcher.parent != launcher_root or launcher.is_symlink() or not launcher.is_file():
        raise PreparationError(f"Unsafe or missing launcher for {row['unit_id']}: {launcher}")
    return launcher


def _stage_selection(build: Path, label: str, rows: Sequence[dict[str, str]]) -> Path:
    launchers = []
    for row in rows:
        launcher = _resolve_launcher(build, row)
        launchers.append((row, launcher, hashlib.sha256(launcher.read_bytes()).digest()))
    digest = hashlib.sha256()
    for row, _, launcher_digest in sorted(launchers, key=lambda item: item[0]["unit_id"]):
        digest.update(row["unit_id"].encode())
        digest.update(launcher_digest)
    selection_name = f"{_safe_slug(label)}-{digest.hexdigest()[:12]}"
    dispatch_root = build / "01-plan/execution/dispatch"
    if dispatch_root.exists() and (dispatch_root.is_symlink() or not dispatch_root.is_dir()):
        raise PreparationError(f"Unsafe dispatch artifact path: {dispatch_root}")
    dispatch_root.mkdir(exist_ok=True)
    selection_dir = dispatch_root / selection_name
    expected_names = {launcher.name for _, launcher, _ in launchers}
    if selection_dir.exists():
        if selection_dir.is_symlink() or not selection_dir.is_dir():
            raise PreparationError(f"Unsafe dispatch selection path: {selection_dir}")
        children = list(selection_dir.iterdir())
        actual_names = {path.name for path in children}
        unsafe = any(path.is_symlink() or not path.is_file() for path in children)
        content_changed = any(
            hashlib.sha256((selection_dir / launcher.name).read_bytes()).digest()
            != launcher_digest
            for _, launcher, launcher_digest in launchers
            if (selection_dir / launcher.name).is_file()
        )
        if actual_names != expected_names or unsafe or content_changed:
            raise PreparationError(f"Existing dispatch selection does not match the plan: {selection_dir}")
        return selection_dir

    selection_dir.mkdir(exist_ok=False)
    for _, launcher, _ in launchers:
        destination = selection_dir / launcher.name
        shutil.copy2(launcher, destination)
    return selection_dir


def _require_preflight(build: Path, selected: Sequence[dict[str, str]]) -> str:
    summary_path = build / "02-preflight/preflight-summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise PreparationError(
            "A passing preflight is required before --execute. Run "
            "python scripts/dolma3p5_resharding/preflight.py immediately before dispatch."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid preflight summary: {summary_path}") from exc
    blocking_fields = ("drifted_input_objects", "occupied_destinations", "errors")
    if not summary.get("passed") or any(int(summary.get(name, 0)) for name in blocking_fields):
        raise PreparationError(f"Preflight did not pass: {summary_path}")
    selected_digest = _unit_selection_digest(selected)
    preflight_scope = summary.get("selection_scope")
    if preflight_scope != "all" and summary.get("selected_unit_ids_sha256") != selected_digest:
        raise PreparationError(
            "Preflight selection does not match this dispatch. Rerun preflight.py with the same "
            "--category or --unit selector."
        )

    destination_rows = _read_csv(build / "02-preflight/destination-status.csv")
    statuses = {row["unit_id"]: row["status"] for row in destination_rows}
    invalid = [row["unit_id"] for row in selected if statuses.get(row["unit_id"]) != "empty"]
    if invalid:
        raise PreparationError(
            f"Preflight does not show an empty destination for {len(invalid):,} selected unit(s)"
        )
    return str(summary.get("created_at", "unknown"))


def _require_every_cluster_node_has_work(
    args: argparse.Namespace,
    selected: Sequence[dict[str, str]],
) -> None:
    """Prevent poormanray from leaving unassigned cluster nodes running."""

    try:
        require_spindown_coverage(
            cluster=args.cluster,
            region=args.region,
            script_count=len(selected),
        )
    except PoormanrayDispatchError as exc:
        raise PreparationError(str(exc)) from exc


def _print_dispatch(
    label: str,
    rows: Sequence[dict[str, str]],
    script_dir: Path,
    command: Sequence[str],
    execute: bool,
) -> None:
    category_count = len({row["leaf_id"] for row in rows})
    planned_tokens = sum(int(row["planned_uint32_values"]) for row in rows)
    largest_unit = max(int(row["estimated_peak_local_bytes"]) for row in rows)
    print("EXECUTE" if execute else "DRY RUN — poormanray will not be invoked")
    print(f"Selection: {label}")
    print(f"Categories: {category_count:,}")
    print(f"Execution units: {len(rows):,}")
    print(f"Planned output: {_human_token_count(planned_tokens)} tokens")
    print(f"Largest local working set: {_human_byte_count(largest_unit)}")
    print(f"Staged launchers: {script_dir}")
    print("\nCommand:")
    print(shlex.join(command))
    print("\nUnits:")
    for row in sorted(rows, key=lambda item: item["unit_id"]):
        print(
            f'{row["unit_id"]}\t{_human_token_count(int(row["planned_uint32_values"]))} tokens\t'
            f'{_human_byte_count(int(row["estimated_peak_local_bytes"]))}\t'
            f'{row["destination_prefix"]}'
        )


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()
        args.region = normalize_region(args.region)
        build = args.build.resolve()
        rows = _load_execution_units(build)
        if args.list_categories is not None:
            _print_categories(rows, args.list_categories)
            return

        label, selected = _select_units(args, rows)
        script_dir = _stage_selection(build, label, selected)
        command = build_poormanray_map_command(
            cluster=args.cluster,
            project=args.project,
            region=args.region,
            script_dir=script_dir,
            spindown=True,
        )
        _print_dispatch(label, selected, script_dir, command, args.execute)
        if not args.execute:
            return
        preflight_created_at = _require_preflight(build, selected)
        if shutil.which("pmr") is None:
            raise PreparationError("pmr is required for --execute and was not found on PATH")
        _require_every_cluster_node_has_work(args, selected)
        print(f"\nPreflight passed: {preflight_created_at}")
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise PreparationError(f"poormanray dispatch failed with exit code {result.returncode}")
        print(f"Dispatched {len(selected):,} execution unit(s); monitor worker status before verification")
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
