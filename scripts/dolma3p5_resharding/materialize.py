# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "poormanray",
#   "PyYAML",
# ]
# ///

"""Provision workers and materialize reviewed Dolma 3.5 execution units."""

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

scripts_root = Path(__file__).resolve().parents[1]
if str(scripts_root) not in sys.path:
    sys.path.insert(0, str(scripts_root))

from resharding.dispatch import (
    build_poormanray_create_command,
    build_poormanray_instance_command,
    build_poormanray_map_command,
    build_poormanray_run_command,
    build_poormanray_setup_dolma_command,
    build_poormanray_transfer_command,
)

try:
    from .workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        PreparationError,
        _filter_execution_units,
        _human_byte_count,
        _human_token_count,
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
        _filter_execution_units,
        _human_byte_count,
        _human_token_count,
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
    parser.add_argument("--profile", help="AWS profile used for the worker lifecycle")
    parser.add_argument(
        "--project",
        default="oe-other",
        help="Ai2 project tag for poormanray dispatch",
    )
    parser.add_argument(
        "-j",
        "--parallelism",
        type=lambda value: _positive_integer(value, "parallelism"),
        default=128,
        help="maximum number of workers to provision and run concurrently",
    )
    parser.add_argument(
        "--instance-type",
        default="i4i.2xlarge",
        help="worker EC2 instance type",
    )
    parser.add_argument(
        "--root-storage-type",
        default="gp3",
        help="worker root-volume type",
    )
    parser.add_argument(
        "--root-storage-size",
        type=lambda value: _positive_integer(value, "root-storage-size"),
        default=200,
        metavar="GIB",
        help="worker root-volume size in GiB",
    )
    parser.add_argument(
        "--storage-layout",
        choices=("single", "raid0"),
        default="single",
        help="local NVMe layout prepared on every worker",
    )
    parser.add_argument(
        "--ssh-key-path",
        type=Path,
        help="SSH private key passed to poormanray; its normal default is used when omitted",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="stage and print the complete worker lifecycle without invoking poormanray",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="provision, prepare, and dispatch poormanray workers after a passing preflight",
    )
    return parser


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive")
    return parsed


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKER_STORAGE_SCRIPT = Path(__file__).with_name("setup_worker_storage.sh")
RESHARD_MODULE = REPOSITORY_ROOT / "python/dolma/tokenizer/reshard.py"
DOCUMENT_SELECTION_MODULE = REPOSITORY_ROOT / "python/dolma/tokenizer/document_selection.py"
REMOTE_STORAGE_SCRIPT = "/tmp/dolma3p5-setup-worker-storage.sh"
REMOTE_RESHARD_MODULE = "/tmp/dolma3p5-runtime/reshard.py"
REMOTE_DOCUMENT_SELECTION_MODULE = "/tmp/dolma3p5-runtime/document_selection.py"


@dataclass(frozen=True)
class ClusterInstance:
    instance_id: str
    state: str
    instance_type: str
    project: str | None


def _describe_cluster_instances(
    cluster: str, region: str, profile: str | None = None
) -> list[ClusterInstance]:
    """Return every unterminated AWS instance bearing the poormanray cluster tag."""

    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("ec2", region_name=region)
    descriptions: dict[str, dict[str, Any]] = {}
    states = ["pending", "running", "stopping", "stopped"]
    for tag_name in ("project", "Project"):
        paginator = client.get_paginator("describe_instances")
        for page in paginator.paginate(
            Filters=[
                {"Name": "instance-state-name", "Values": states},
                {"Name": f"tag:{tag_name}", "Values": [cluster]},
            ]
        ):
            for reservation in page.get("Reservations", []):
                for description in reservation.get("Instances", []):
                    instance_id = description.get("InstanceId")
                    if isinstance(instance_id, str):
                        descriptions[instance_id] = description

    instances = []
    for instance_id, description in descriptions.items():
        tags = {
            str(tag.get("Key")): str(tag.get("Value"))
            for tag in description.get("Tags", [])
            if tag.get("Key") is not None and tag.get("Value") is not None
        }
        instances.append(
            ClusterInstance(
                instance_id=instance_id,
                state=str(description.get("State", {}).get("Name", "unknown")),
                instance_type=str(description.get("InstanceType", "unknown")),
                project=tags.get("ai2-project"),
            )
        )
    return sorted(instances, key=lambda instance: instance.instance_id)


def _run_lifecycle_command(stage: str, command: Sequence[str]) -> None:
    print(f"\n[{stage}]")
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise PreparationError(f"{stage} failed with exit code {result.returncode}")


def _instance_options(args: argparse.Namespace, instance_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "cluster": args.cluster,
        "project": args.project,
        "region": args.region,
        "instance_ids": instance_ids,
        "parallelism": (
            min(args.parallelism, len(instance_ids)) if instance_ids else args.parallelism
        ),
        "ssh_key_path": args.ssh_key_path,
    }


def _create_command(args: argparse.Namespace, number: int) -> list[str]:
    return build_poormanray_create_command(
        cluster=args.cluster,
        project=args.project,
        region=args.region,
        number=number,
        instance_type=args.instance_type,
        storage_type=args.root_storage_type,
        storage_size_gib=args.root_storage_size,
        parallelism=min(args.parallelism, number),
        ssh_key_path=args.ssh_key_path,
    )


def _wait_command(args: argparse.Namespace, instance_ids: Sequence[str]) -> list[str]:
    return build_poormanray_instance_command(
        "wait",
        cluster=args.cluster,
        project=args.project,
        region=args.region,
        instance_ids=instance_ids,
        ssh_key_path=args.ssh_key_path,
    )


def _resume_command(args: argparse.Namespace, instance_ids: Sequence[str]) -> list[str]:
    options = _instance_options(args, instance_ids)
    options.pop("ssh_key_path")
    return build_poormanray_instance_command("resume", **options)


def _pause_command(args: argparse.Namespace, instance_ids: Sequence[str]) -> list[str]:
    options = _instance_options(args, instance_ids)
    options.pop("ssh_key_path")
    return build_poormanray_instance_command("pause", **options)


def _worker_minimum_available_bytes(rows: Sequence[dict[str, str]]) -> int:
    largest_working_set = max(int(row["estimated_peak_local_bytes"]) for row in rows)
    return (largest_working_set * 11 + 9) // 10


def _storage_transfer_command(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> list[str]:
    return build_poormanray_transfer_command(
        transfers=((WORKER_STORAGE_SCRIPT, REMOTE_STORAGE_SCRIPT),),
        **_instance_options(args, instance_ids),
    )


def _storage_setup_command(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> list[str]:
    minimum_bytes = _worker_minimum_available_bytes(rows)
    remote_command = (
        f"DOLMA_MIN_AVAILABLE_BYTES={minimum_bytes} "
        f"bash {shlex.quote(REMOTE_STORAGE_SCRIPT)} --apply --layout {args.storage_layout}"
    )
    return build_poormanray_run_command(
        remote_command=remote_command,
        **_instance_options(args, instance_ids),
    )


def _runtime_setup_command(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> list[str]:
    return build_poormanray_setup_dolma_command(**_instance_options(args, instance_ids))


def _runtime_transfer_command(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> list[str]:
    return build_poormanray_transfer_command(
        transfers=(
            (RESHARD_MODULE, REMOTE_RESHARD_MODULE),
            (DOCUMENT_SELECTION_MODULE, REMOTE_DOCUMENT_SELECTION_MODULE),
        ),
        **_instance_options(args, instance_ids),
    )


def _runtime_validation_command(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> list[str]:
    remote_command = """set -euo pipefail
export PYTHONSAFEPATH=1
cd /tmp
python_bin="$HOME/.venv/bin/python"
module_dir=$(
  "$python_bin" -P -c 'import pathlib, dolma.tokenizer; print(pathlib.Path(dolma.tokenizer.__file__).parent)'
)
install -m 0644 /tmp/dolma3p5-runtime/reshard.py "$module_dir/reshard.py"
install -m 0644 /tmp/dolma3p5-runtime/document_selection.py "$module_dir/document_selection.py"
"$python_bin" -P -c 'from dolma.tokenizer.reshard import RESHARDING_MANIFEST_SCHEMA_VERSION; assert RESHARDING_MANIFEST_SCHEMA_VERSION == 2'
s5cmd version
findmnt /mnt/dolma
test -w /mnt/dolma/dolma3p5-resharding"""
    return build_poormanray_run_command(
        remote_command=remote_command,
        **_instance_options(args, instance_ids),
    )


def _map_command(
    args: argparse.Namespace,
    script_dir: Path,
    instance_ids: Sequence[str] = (),
) -> list[str]:
    return build_poormanray_map_command(
        cluster=args.cluster,
        project=args.project,
        region=args.region,
        script_dir=script_dir,
        spindown=True,
        instance_ids=instance_ids,
        ssh_key_path=args.ssh_key_path,
    )


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


def _safe_path_launcher_payload(launcher: Path) -> bytes:
    """Force safe-path mode for existing and newly generated worker launchers."""

    text = launcher.read_text(encoding="utf-8")
    safe_path_export = "export PYTHONSAFEPATH=1"
    safe_working_directory = "cd /tmp"
    strict_mode = "set -euo pipefail\n"
    if strict_mode not in text:
        raise PreparationError(f"Worker launcher is missing strict shell mode: {launcher}")
    missing_directives = [
        directive
        for directive in (safe_path_export, safe_working_directory)
        if directive not in text
    ]
    if missing_directives:
        inserted = "\n".join(missing_directives)
        text = text.replace(strict_mode, f"{strict_mode}\n{inserted}\n", 1)
    return text.encode("utf-8")


def _stage_selection(build: Path, label: str, rows: Sequence[dict[str, str]]) -> Path:
    launchers = []
    for row in rows:
        launcher = _resolve_launcher(build, row)
        payload = _safe_path_launcher_payload(launcher)
        launchers.append((row, launcher, payload, hashlib.sha256(payload).digest()))
    digest = hashlib.sha256()
    for row, _, _, launcher_digest in sorted(launchers, key=lambda item: item[0]["unit_id"]):
        digest.update(row["unit_id"].encode())
        digest.update(launcher_digest)
    selection_name = f"{_safe_slug(label)}-{digest.hexdigest()[:12]}"
    dispatch_root = build / "01-plan/execution/dispatch"
    if dispatch_root.exists() and (dispatch_root.is_symlink() or not dispatch_root.is_dir()):
        raise PreparationError(f"Unsafe dispatch artifact path: {dispatch_root}")
    dispatch_root.mkdir(exist_ok=True)
    selection_dir = dispatch_root / selection_name
    expected_names = {launcher.name for _, launcher, _, _ in launchers}
    if selection_dir.exists():
        if selection_dir.is_symlink() or not selection_dir.is_dir():
            raise PreparationError(f"Unsafe dispatch selection path: {selection_dir}")
        children = list(selection_dir.iterdir())
        actual_names = {path.name for path in children}
        unsafe = any(path.is_symlink() or not path.is_file() for path in children)
        content_changed = any(
            hashlib.sha256((selection_dir / launcher.name).read_bytes()).digest()
            != launcher_digest
            for _, launcher, _, launcher_digest in launchers
            if (selection_dir / launcher.name).is_file()
        )
        if actual_names != expected_names or unsafe or content_changed:
            raise PreparationError(f"Existing dispatch selection does not match the plan: {selection_dir}")
        return selection_dir

    selection_dir.mkdir(exist_ok=False)
    for _, launcher, payload, _ in launchers:
        destination = selection_dir / launcher.name
        destination.write_bytes(payload)
        destination.chmod(launcher.stat().st_mode & 0o777)
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


def _pause_workers_after_failure(args: argparse.Namespace, instance_ids: Sequence[str]) -> None:
    if not instance_ids:
        return
    command = _pause_command(args, instance_ids)
    print("\n[cleanup: pause workers after failure]", file=sys.stderr)
    print(shlex.join(command), file=sys.stderr, flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        print(
            f"WARNING: worker cleanup failed with exit code {result.returncode}; "
            f"pause cluster {args.cluster!r} immediately",
            file=sys.stderr,
        )


def _prepare_workers(args: argparse.Namespace, worker_count: int) -> list[str]:
    """Resume stopped compatible workers and create any remaining workers."""

    try:
        before = _describe_cluster_instances(args.cluster, args.region, args.profile)
    except Exception as exc:
        raise PreparationError(
            f"Could not inspect poormanray cluster {args.cluster!r} in {args.region}: {exc}"
        ) from exc

    busy = [instance for instance in before if instance.state != "stopped"]
    if busy:
        states = ", ".join(
            f"{instance.instance_id}={instance.state}" for instance in busy
        )
        raise PreparationError(
            f"Cluster {args.cluster!r} already has active or transitioning workers ({states}); "
            "refusing to mix this materialization with another lifecycle"
        )

    reusable = [
        instance
        for instance in before
        if instance.state == "stopped"
        and instance.instance_type == args.instance_type
        and instance.project == args.project
    ][:worker_count]
    selected_ids = [instance.instance_id for instance in reusable]
    before_ids = {instance.instance_id for instance in before}

    try:
        if selected_ids:
            _run_lifecycle_command("resume workers", _resume_command(args, selected_ids))

        missing = worker_count - len(selected_ids)
        if missing:
            _run_lifecycle_command("create workers", _create_command(args, missing))
            after = _describe_cluster_instances(args.cluster, args.region, args.profile)
            created = [
                instance
                for instance in after
                if instance.instance_id not in before_ids
                and instance.state in {"pending", "running"}
                and instance.instance_type == args.instance_type
                and instance.project == args.project
            ]
            if len(created) != missing:
                selected_ids.extend(instance.instance_id for instance in created)
                raise PreparationError(
                    f"Expected poormanray to create {missing:,} worker(s), but found "
                    f"{len(created):,} new matching worker(s)"
                )
            selected_ids.extend(instance.instance_id for instance in created)

        if len(selected_ids) != worker_count:
            raise PreparationError(
                f"Worker lifecycle selected {len(selected_ids):,} workers; expected {worker_count:,}"
            )
        _run_lifecycle_command("wait for workers", _wait_command(args, selected_ids))
        return sorted(selected_ids)
    except BaseException:
        if len(selected_ids) < worker_count:
            try:
                after_failure = _describe_cluster_instances(
                    args.cluster, args.region, args.profile
                )
                selected_ids.extend(
                    instance.instance_id
                    for instance in after_failure
                    if instance.instance_id not in before_ids
                    and instance.state in {"pending", "running"}
                    and instance.project == args.project
                )
            except (BotoCoreError, ClientError) as cleanup_exc:
                print(
                    f"WARNING: could not discover partially created workers for cleanup: {cleanup_exc}",
                    file=sys.stderr,
                )
        _pause_workers_after_failure(args, sorted(set(selected_ids)))
        raise


def _dry_run_lifecycle_commands(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    script_dir: Path,
    worker_count: int,
) -> list[tuple[str, list[str]]]:
    """Show the create path; execute may resume compatible stopped workers instead."""

    return [
        ("create missing workers", _create_command(args, worker_count)),
        ("wait for selected workers", _wait_command(args, ())),
        ("upload storage setup", _storage_transfer_command(args, ())),
        ("prepare local NVMe", _storage_setup_command(args, (), rows)),
        ("install Dolma and s5cmd", _runtime_setup_command(args, ())),
        ("upload reviewed resharder", _runtime_transfer_command(args, ())),
        ("install and validate reviewed resharder", _runtime_validation_command(args, ())),
        ("dispatch and stop workers when done", _map_command(args, script_dir)),
    ]


def _print_dispatch(
    label: str,
    rows: Sequence[dict[str, str]],
    script_dir: Path,
    lifecycle_commands: Sequence[tuple[str, Sequence[str]]],
    worker_count: int,
    execute: bool,
) -> None:
    category_count = len({row["leaf_id"] for row in rows})
    planned_tokens = sum(int(row["planned_uint32_values"]) for row in rows)
    largest_unit = max(int(row["estimated_peak_local_bytes"]) for row in rows)
    print("EXECUTE" if execute else "DRY RUN — poormanray will not be invoked")
    print(f"Selection: {label}")
    print(f"Categories: {category_count:,}")
    print(f"Execution units: {len(rows):,}")
    print(f"Workers: {worker_count:,}")
    print(f"Planned output: {_human_token_count(planned_tokens)} tokens")
    print(f"Largest local working set: {_human_byte_count(largest_unit)}")
    print(f"Staged launchers: {script_dir}")
    print("\nWorker lifecycle:")
    for stage, command in lifecycle_commands:
        print(f"\n{stage}:")
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
        worker_count = min(len(selected), args.parallelism)
        lifecycle_commands = _dry_run_lifecycle_commands(
            args,
            selected,
            script_dir,
            worker_count,
        )
        _print_dispatch(
            label,
            selected,
            script_dir,
            lifecycle_commands,
            worker_count,
            args.execute,
        )
        if not args.execute:
            return
        preflight_created_at = _require_preflight(build, selected)
        if shutil.which("pmr") is None:
            raise PreparationError(
                "pmr is unavailable; run materialize.py with uv so its inline dependencies are installed"
            )
        for required_path in (
            WORKER_STORAGE_SCRIPT,
            RESHARD_MODULE,
            DOCUMENT_SELECTION_MODULE,
        ):
            if required_path.is_symlink() or not required_path.is_file():
                raise PreparationError(f"Required reviewed worker file is missing or unsafe: {required_path}")
        if args.profile:
            os.environ["AWS_PROFILE"] = args.profile
        print(f"\nPreflight passed: {preflight_created_at}")
        worker_ids = _prepare_workers(args, worker_count)
        try:
            _run_lifecycle_command(
                "upload storage setup", _storage_transfer_command(args, worker_ids)
            )
            _run_lifecycle_command(
                "prepare local NVMe", _storage_setup_command(args, worker_ids, selected)
            )
            _run_lifecycle_command(
                "install Dolma and s5cmd", _runtime_setup_command(args, worker_ids)
            )
            _run_lifecycle_command(
                "upload reviewed resharder", _runtime_transfer_command(args, worker_ids)
            )
            _run_lifecycle_command(
                "install and validate reviewed resharder",
                _runtime_validation_command(args, worker_ids),
            )
            _run_lifecycle_command(
                "dispatch materialization", _map_command(args, script_dir, worker_ids)
            )
        except BaseException:
            _pause_workers_after_failure(args, worker_ids)
            raise
        print(
            f"Dispatched {len(selected):,} execution unit(s) across {worker_count:,} worker(s); "
            "each worker will stop after its assigned units finish"
        )
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
