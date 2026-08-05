# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "poormanray",
#   "PyYAML",
#   "rich",
# ]
# ///

"""Provision workers and materialize Dolma 3.5 execution units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console
from rich.table import Table
from rich.text import Text

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
        UINT32_BYTES,
        PreparationError,
        _filter_execution_units,
        _human_byte_count,
        _human_token_count,
        _list_prefix,
        _pair_metadata_key,
        _unit_selection_digest,
        _validate_execution_layout,
        _validate_preparation_build,
        normalize_region,
    )
except ImportError:
    from workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        UINT32_BYTES,
        PreparationError,
        _filter_execution_units,
        _human_byte_count,
        _human_token_count,
        _list_prefix,
        _pair_metadata_key,
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
        help="worker project tag",
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
    parser.add_argument(
        "--completion-poll-seconds",
        type=lambda value: _positive_integer(value, "completion-poll-seconds"),
        default=30,
        metavar="SECONDS",
        help="interval for checking whether materialization workers have stopped",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="stream poormanray output and periodically show worker resharding logs",
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
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WAIT_STATUS = re.compile(
    r"Waiting for instances\.\.\.\s*(?P<ready>\d+/\d+ ready)"
    r"(?:\s*\((?P<elapsed>[^)]+)\))?",
    re.IGNORECASE,
)
AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
SECRET_VALUE = re.compile(
    r"(?i)\b(aws_secret_access_key|aws_session_token|secret_access_key)"
    r"(\s*[=:]\s*)\S+"
)
PROCESS_TAIL_LINES = 12
WORKER_LOG_STYLES = (
    "bold bright_cyan",
    "bold bright_magenta",
    "bold bright_green",
    "bold bright_yellow",
    "bold bright_blue",
    "bold bright_red",
)


@dataclass(frozen=True)
class ClusterInstance:
    instance_id: str
    state: str
    instance_type: str
    project: str | None
    name: str = ""


@dataclass(frozen=True)
class MaterializedUnitCheck:
    unit_id: str
    actual_uint32_values: int
    npy_count: int
    metadata_count: int
    problems: tuple[str, ...]


def _worker_log_line(tag: str, style: str, message: str, *, bold: bool = False) -> Text:
    line = Text()
    line.append(f"[{tag}]", style=style)
    line.append(" ")
    line.append(message, style="bold" if bold else None)
    return line


def _describe_cluster_instances(
    cluster: str, region: str, profile: str | None = None
) -> list[ClusterInstance]:
    """Return instances using the cluster tag plus legacy poormanray discovery."""

    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("ec2", region_name=region)
    descriptions: dict[str, dict[str, Any]] = {}
    states = ["pending", "running", "stopping", "stopped"]
    for tag_name in ("cluster", "project", "Project"):
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
                project=tags.get("ai2-project") or tags.get("project"),
                name=tags.get("Name", ""),
            )
        )
    return sorted(instances, key=lambda instance: instance.instance_id)


def _retag_cluster_instances(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    *,
    names: dict[str, str] | None = None,
) -> None:
    """Apply accounting and cluster tags, replacing poormanray's project misuse."""

    if not instance_ids:
        return
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("ec2", region_name=args.region)
    required_tags = {
        "project": args.project,
        "ai2-project": args.project,
        "cluster": args.cluster,
    }
    client.create_tags(
        Resources=list(instance_ids),
        Tags=[{"Key": key, "Value": value} for key, value in required_tags.items()],
    )
    for instance_id, name in sorted((names or {}).items()):
        client.create_tags(
            Resources=[instance_id],
            Tags=[{"Key": "Name", "Value": name}],
        )

    deadline = time.monotonic() + 60
    while True:
        response = client.describe_instances(InstanceIds=list(instance_ids))
        observed: dict[str, dict[str, str]] = {}
        for reservation in response.get("Reservations", []):
            for description in reservation.get("Instances", []):
                instance_id = str(description.get("InstanceId", ""))
                observed[instance_id] = {
                    str(tag.get("Key")): str(tag.get("Value"))
                    for tag in description.get("Tags", [])
                    if tag.get("Key") is not None and tag.get("Value") is not None
                }
        if all(
            all(observed.get(instance_id, {}).get(key) == value for key, value in required_tags.items())
            and (
                names is None
                or instance_id not in names
                or observed[instance_id].get("Name") == names[instance_id]
            )
            for instance_id in instance_ids
        ):
            return
        if time.monotonic() >= deadline:
            raise PreparationError(
                "Timed out waiting for required worker tags on "
                + ", ".join(sorted(instance_ids))
            )
        time.sleep(2)


def _clean_process_line(raw_line: str) -> str:
    line = ANSI_ESCAPE.sub("", raw_line).replace("\r", "").strip()
    line = AWS_ACCESS_KEY.sub("<redacted-aws-key>", line)
    return SECRET_VALUE.sub(r"\1\2<redacted>", line)


def _status_detail(stage: str, line: str) -> str | None:
    if not line:
        return None
    if stage == "wait for workers":
        match = WAIT_STATUS.search(line)
        if match:
            elapsed = match.group("elapsed")
            return f"{match.group('ready')} · {elapsed}" if elapsed else match.group("ready")
        if line.startswith(("·", "•")):
            return None
    if line.startswith("[INFO]"):
        if stage == "submit materialization":
            scripts = re.search(r"Found ([\d,]+) scripts? to distribute", line)
            if scripts:
                return f"{scripts.group(1)} units"
            workers = re.search(r"Job \S+ started on ([\d,]+) instances?", line)
            if workers:
                return f"accepted by {workers.group(1)} workers"
        return None
    if line.startswith(("Instance ", "stdout:", "stderr:")):
        return None
    return line if len(line) <= 120 else f"{line[:117]}..."


def _stage_status(stage: str, detail: str | None = None) -> Text:
    status = Text(stage, style="bold")
    if detail:
        status.append("  ", style="dim")
        status.append(detail, style="dim")
    return status


def _elapsed_time(started_at: float) -> str:
    elapsed = max(0, round(time.monotonic() - started_at))
    minutes, seconds = divmod(elapsed, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _run_compact_process(
    stage: str,
    command: Sequence[str],
    *,
    console: Console | None = None,
    verbose: bool = False,
) -> int:
    """Run a command with one in-place status and a bounded failure log."""

    output = console or Console(stderr=True, highlight=False)
    started_at = time.monotonic()
    tail: deque[str] = deque(maxlen=PROCESS_TAIL_LINES)
    process: subprocess.Popen[str] | None = None
    live_status = (
        output.status(_stage_status(stage), spinner="dots")
        if output.is_terminal and not verbose
        else None
    )
    if verbose:
        output.print(Text.assemble(("→", "cyan"), " ", (stage, "bold")))
    elif live_status is None:
        output.print(Text.assemble(("…", "cyan"), " ", (stage, "bold")))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if live_status is not None:
            live_status.start()
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = _clean_process_line(raw_line)
            if not line:
                continue
            if not tail or tail[-1] != line:
                tail.append(line)
            if verbose:
                output.print(Text(f"  {line}", style="dim"))
            detail = _status_detail(stage, line)
            if live_status is not None and detail:
                live_status.update(_stage_status(stage, detail))
        return_code = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if live_status is not None:
            live_status.stop()

    duration = _elapsed_time(started_at)
    if return_code == 0:
        output.print(Text.assemble(("✓", "bold green"), " ", stage, (f"  {duration}", "dim")))
        return return_code

    output.print(Text.assemble(("✗", "bold red"), " ", stage, (f"  {duration}", "dim")))
    if tail:
        output.print(Text("last output:", style="bold red"))
        for line in tail:
            output.print(Text(f"  {line[:240]}", style="dim"))
    return return_code


def _run_lifecycle_command(
    stage: str,
    command: Sequence[str],
    *,
    verbose: bool = False,
) -> None:
    try:
        return_code = _run_compact_process(stage, command, verbose=verbose)
    except OSError as exc:
        raise PreparationError(f"could not start {stage}: {exc}") from exc
    if return_code:
        raise PreparationError(f"{stage} failed with exit code {return_code}")


def _instance_options(args: argparse.Namespace, instance_ids: Sequence[str]) -> dict[str, Any]:
    return {
        # Poormanray selects existing AWS instances through the `project` tag
        # supplied as --name. Our workers use the accounting project there and
        # are isolated by explicit instance IDs plus the separate `cluster` tag.
        "cluster": args.project,
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
        cluster=args.project,
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
    remote_command = (
        "set -euo pipefail; export PYTHONSAFEPATH=1; cd /tmp; "
        'python_bin="$HOME/.venv/bin/python"; '
        "module_dir=$(\"$python_bin\" -P -c 'import pathlib, dolma.tokenizer; "
        "print(pathlib.Path(dolma.tokenizer.__file__).parent)'); "
        'install -m 0644 /tmp/dolma3p5-runtime/reshard.py "$module_dir/reshard.py"; '
        "install -m 0644 /tmp/dolma3p5-runtime/document_selection.py "
        '"$module_dir/document_selection.py"; '
        "\"$python_bin\" -P -c 'from dolma.tokenizer.reshard import "
        "RESHARDING_MANIFEST_SCHEMA_VERSION; assert RESHARDING_MANIFEST_SCHEMA_VERSION == 2'; "
        "s5cmd version; findmnt /mnt/dolma; test -w /mnt/dolma/dolma3p5-resharding"
    )
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
        cluster=args.project,
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


def _safe_path_launcher_payload(
    launcher: Path,
    status_run_id: str | None = None,
) -> bytes:
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
    if status_run_id is not None:
        if not re.fullmatch(r"[a-z0-9-]+", status_run_id):
            raise PreparationError(f"Unsafe materialization run ID: {status_run_id!r}")
        status_directive = (
            'export DOLMA_STATUS_ROOT="$HOME/dolma3p5-resharding-status/'
            f'{status_run_id}"'
        )
        text = text.replace(strict_mode, f"{strict_mode}\n{status_directive}\n", 1)
    return text.encode("utf-8")


def _stage_selection(
    build: Path,
    label: str,
    rows: Sequence[dict[str, str]],
    status_run_id: str | None = None,
) -> Path:
    launchers = []
    for row in rows:
        launcher = _resolve_launcher(build, row)
        payload = _safe_path_launcher_payload(launcher, status_run_id)
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
    try:
        return_code = _run_compact_process(
            "pause workers after failure",
            command,
            verbose=getattr(args, "verbose", False),
        )
    except OSError as exc:
        print(
            f"WARNING: worker cleanup could not start: {exc}; "
            f"run {shlex.join(command)} immediately",
            file=sys.stderr,
        )
        return
    if return_code:
        print(
            f"WARNING: worker cleanup failed with exit code {return_code}; "
            f"run {shlex.join(command)} immediately",
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
    name_pattern = re.compile(rf"^{re.escape(args.cluster)}-(\d+)$")
    existing_name_indices = [
        int(match.group(1))
        for instance in before
        if (match := name_pattern.fullmatch(instance.name)) is not None
    ]
    next_name_index = max(existing_name_indices, default=-1) + 1

    try:
        if selected_ids:
            _retag_cluster_instances(args, selected_ids)
            _run_lifecycle_command(
                "resume workers",
                _resume_command(args, selected_ids),
                verbose=getattr(args, "verbose", False),
            )

        missing = worker_count - len(selected_ids)
        if missing:
            _run_lifecycle_command(
                "create workers",
                _create_command(args, missing),
                verbose=getattr(args, "verbose", False),
            )
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
            created_ids = sorted(instance.instance_id for instance in created)
            created_names = {
                instance_id: f"{args.cluster}-{next_name_index + offset:04d}"
                for offset, instance_id in enumerate(created_ids)
            }
            _retag_cluster_instances(args, created_ids, names=created_names)
            selected_ids.extend(created_ids)

        if len(selected_ids) != worker_count:
            raise PreparationError(
                f"Worker lifecycle selected {len(selected_ids):,} workers; expected {worker_count:,}"
            )
        _run_lifecycle_command(
            "wait for workers",
            _wait_command(args, selected_ids),
            verbose=getattr(args, "verbose", False),
        )
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
        try:
            _retag_cluster_instances(args, sorted(set(selected_ids)))
        except (BotoCoreError, ClientError, PreparationError) as tag_exc:
            print(
                f"WARNING: could not apply worker tags before cleanup: {tag_exc}",
                file=sys.stderr,
            )
        _pause_workers_after_failure(args, sorted(set(selected_ids)))
        raise


def _worker_log_command(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    status_run_id: str,
) -> list[str]:
    status_root = f"$HOME/dolma3p5-resharding-status/{status_run_id}"
    remote_script = f"""status_root=\"{status_root}\"
shopt -s nullglob
status_files=(\"$status_root\"/*.status)
log_files=(\"$status_root\"/*.log)
if (( ${{#status_files[@]}} == 0 )); then
  echo 'no unit status yet'
else
  for path in \"${{status_files[@]}}\"; do
    printf '@@DOLMA_STATUS@@\t%s\t' \"$(basename \"$path\" .status)\"
    tr -d '\n' < \"$path\"
    printf '\n'
  done
fi
for path in \"${{log_files[@]}}\"; do
  printf '@@DOLMA_LOG_BEGIN@@\t%s\n' \"$(basename \"$path\")\"
  cat \"$path\"
  printf '\n@@DOLMA_LOG_END@@\t%s\n' \"$(basename \"$path\")\"
done"""
    return build_poormanray_run_command(
        cluster=args.project,
        project=args.project,
        region=args.region,
        remote_command=f"bash -lc {shlex.quote(remote_script)}",
        instance_ids=instance_ids,
        parallelism=min(args.parallelism, len(instance_ids)),
        ssh_key_path=args.ssh_key_path,
    )


def _worker_log_snapshots(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    status_run_id: str,
) -> dict[str, dict[str, dict[str, tuple[str, ...] | str]]]:
    """Read every current-run status and log from every active worker."""

    try:
        result = subprocess.run(
            _worker_log_command(args, instance_ids, status_run_id),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}

    instance_payloads: dict[str, list[str]] = {}
    current_instance: str | None = None
    reading_stdout = False
    for raw_line in result.stdout.splitlines():
        line = _clean_process_line(raw_line)
        instance_match = re.fullmatch(r"Instance ([^:]+):", line)
        if instance_match:
            current_instance = instance_match.group(1)
            instance_payloads.setdefault(current_instance, [])
            reading_stdout = False
            continue
        if line.startswith("stdout:"):
            reading_stdout = True
            remainder = line.removeprefix("stdout:").strip()
            if remainder and current_instance is not None:
                instance_payloads[current_instance].append(remainder)
            continue
        if line == "stderr:":
            reading_stdout = False
            continue
        if reading_stdout and line and current_instance is not None:
            instance_payloads[current_instance].append(line)

    snapshots: dict[str, dict[str, dict[str, tuple[str, ...] | str]]] = {}
    for instance_id, payload in instance_payloads.items():
        statuses: dict[str, str] = {}
        logs: dict[str, tuple[str, ...]] = {}
        current_log: str | None = None
        current_lines: list[str] = []
        for line in payload:
            if line.startswith("@@DOLMA_STATUS@@\t"):
                _, unit_id, status = line.split("\t", 2)
                statuses[unit_id] = status
                continue
            if line.startswith("@@DOLMA_LOG_BEGIN@@\t"):
                current_log = line.split("\t", 1)[1]
                current_lines = []
                continue
            if line.startswith("@@DOLMA_LOG_END@@\t"):
                if current_log is not None:
                    logs[current_log] = tuple(current_lines)
                current_log = None
                current_lines = []
                continue
            if current_log is not None:
                current_lines.append(line)
        snapshots[instance_id] = {"statuses": statuses, "logs": logs}
    return snapshots


def _wait_for_workers_to_stop(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    unit_count: int,
    status_run_id: str,
    *,
    describe: Callable[[str, str, str | None], list[ClusterInstance]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    console: Console | None = None,
) -> None:
    """Wait for PMR spindown so detached dispatch is not mistaken for completion."""

    describe_instances = describe or _describe_cluster_instances
    output = console or Console(stderr=True, highlight=False)
    expected_ids = set(instance_ids)
    started_at = time.monotonic()
    verbose = getattr(args, "verbose", False)
    previous_statuses: dict[tuple[str, str], str] = {}
    emitted_log_lines: dict[tuple[str, str], int] = {}
    worker_number_width = max(2, len(str(len(expected_ids))))
    worker_tags = {
        instance_id: (
            f"worker {index:0{worker_number_width}d} · {instance_id}",
            WORKER_LOG_STYLES[(index - 1) % len(WORKER_LOG_STYLES)],
        )
        for index, instance_id in enumerate(sorted(expected_ids), start=1)
    }
    live_status = output.status(_stage_status("materializing"), spinner="dots") if output.is_terminal else None
    if live_status is None:
        output.print(Text.assemble(("…", "cyan"), " ", ("materializing", "bold")))
    else:
        live_status.start()

    try:
        while True:
            cluster = describe_instances(args.cluster, args.region, args.profile)
            selected = {
                instance.instance_id: instance
                for instance in cluster
                if instance.instance_id in expected_ids
            }
            missing = expected_ids - set(selected)
            if missing:
                raise PreparationError(
                    "Could not find selected materialization worker(s): "
                    + ", ".join(sorted(missing))
                )
            state_counts = Counter(instance.state for instance in selected.values())
            stopped = state_counts["stopped"]
            detail = " · ".join(
                (
                    f"{unit_count:,} units",
                    f"{state_counts['running']:,} running",
                    f"{state_counts['stopping']:,} stopping",
                    f"{stopped:,} stopped",
                    _elapsed_time(started_at),
                )
            )
            if live_status is not None:
                live_status.update(_stage_status("materializing", detail))
            if verbose:
                output.print(Text(f"materializing  {detail}", style="dim"))
                running_ids = sorted(
                    instance.instance_id
                    for instance in selected.values()
                    if instance.state == "running"
                )
                if running_ids:
                    snapshots = _worker_log_snapshots(args, running_ids, status_run_id)
                    for instance_id in sorted(snapshots):
                        worker_tag, worker_style = worker_tags[instance_id]
                        snapshot = snapshots[instance_id]
                        statuses = snapshot["statuses"]
                        assert isinstance(statuses, dict)
                        for unit_id, status in sorted(statuses.items()):
                            assert isinstance(status, str)
                            status_key = (instance_id, unit_id)
                            if previous_statuses.get(status_key) != status:
                                previous_statuses[status_key] = status
                                output.print(
                                    _worker_log_line(
                                        worker_tag,
                                        worker_style,
                                        f"{unit_id} · {status}",
                                        bold=True,
                                    )
                                )
                        logs = snapshot["logs"]
                        assert isinstance(logs, dict)
                        for log_name, log_lines in sorted(logs.items()):
                            assert isinstance(log_lines, tuple)
                            log_key = (instance_id, log_name)
                            emitted = emitted_log_lines.get(log_key, 0)
                            if len(log_lines) < emitted:
                                emitted = 0
                            new_lines = log_lines[emitted:]
                            if new_lines:
                                output.print(
                                    _worker_log_line(
                                        worker_tag,
                                        worker_style,
                                        log_name,
                                        bold=True,
                                    )
                                )
                                for line in new_lines:
                                    output.print(
                                        _worker_log_line(
                                            worker_tag,
                                            worker_style,
                                            line,
                                        )
                                    )
                            emitted_log_lines[log_key] = len(log_lines)
            if stopped == len(expected_ids):
                break
            sleep(args.completion_poll_seconds)
    finally:
        if live_status is not None:
            live_status.stop()

    output.print(
        Text.assemble(
            ("✓", "bold green"),
            " materialization workers stopped",
            (f"  {_elapsed_time(started_at)}", "dim"),
        )
    )


def _check_materialized_unit(client: Any, row: dict[str, str]) -> MaterializedUnitCheck:
    parsed = urlparse(row["destination_prefix"])
    if parsed.scheme != "s3" or not parsed.netloc:
        return MaterializedUnitCheck(
            row["unit_id"],
            0,
            0,
            0,
            (f"unsupported destination: {row['destination_prefix']}",),
        )
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    objects = _list_prefix(client, parsed.netloc, prefix)
    object_map = {obj.key: obj for obj in objects}
    npys = [obj for obj in objects if obj.key.endswith(".npy")]
    metadata = {obj.key for obj in objects if obj.key.endswith(".csv.gz")}
    problems: list[str] = []
    if not npys:
        problems.append("no NPY output")
    invalid_npys = [obj for obj in npys if obj.size_bytes <= 0 or obj.size_bytes % UINT32_BYTES]
    if invalid_npys:
        problems.append(f"{len(invalid_npys):,} invalid NPY sizes")
    missing_metadata = [
        _pair_metadata_key(obj.key)
        for obj in npys
        if _pair_metadata_key(obj.key) not in metadata
    ]
    if missing_metadata:
        problems.append(f"{len(missing_metadata):,} NPYs missing metadata")
    orphan_metadata = [
        key for key in metadata if key[: -len(".csv.gz")] + ".npy" not in object_map
    ]
    if orphan_metadata:
        problems.append(f"{len(orphan_metadata):,} orphan metadata files")
    unexpected = [
        obj for obj in objects if not obj.key.endswith((".npy", ".csv.gz"))
    ]
    if unexpected:
        problems.append(f"{len(unexpected):,} unexpected output objects")

    actual = sum(obj.size_bytes // UINT32_BYTES for obj in npys)
    predicted = int(row["planned_uint32_values"])
    allowed = int(row["allowed_materialized_target_residual_uint32_values"])
    residual = actual - predicted
    if abs(residual) > allowed:
        problems.append(
            f"token estimate differs by {residual:+,}; allowed residual is {allowed:,}"
        )
    return MaterializedUnitCheck(
        unit_id=row["unit_id"],
        actual_uint32_values=actual,
        npy_count=len(npys),
        metadata_count=len(metadata),
        problems=tuple(problems),
    )


def _verify_materialized_units(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    *,
    client: Any | None = None,
    console: Console | None = None,
) -> list[MaterializedUnitCheck]:
    """Prove selected units completed using destination objects and their sizes."""

    if client is None:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        client = session.client("s3", region_name=args.region)
    output = console or Console(stderr=True, highlight=False)
    started_at = time.monotonic()
    live_status = output.status(_stage_status("verify materialized outputs"), spinner="dots") if output.is_terminal else None
    if live_status is None:
        output.print(Text.assemble(("…", "cyan"), " ", ("verify materialized outputs", "bold")))
    else:
        live_status.start()

    checks: list[MaterializedUnitCheck] = []
    request_errors: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=min(args.parallelism, len(rows))) as pool:
            futures = {pool.submit(_check_materialized_unit, client, row): row for row in rows}
            for completed, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    checks.append(future.result())
                except Exception as exc:
                    request_errors.append(f"{row['unit_id']}: {exc}")
                if live_status is not None:
                    live_status.update(
                        _stage_status(
                            "verify materialized outputs",
                            f"{completed}/{len(rows)} units checked",
                        )
                    )
    finally:
        if live_status is not None:
            live_status.stop()

    failed = [check for check in checks if check.problems]
    if request_errors or failed or len(checks) != len(rows):
        details = [*request_errors]
        details.extend(
            f"{check.unit_id}: {', '.join(check.problems)}" for check in failed
        )
        summary = "; ".join(details[:5])
        if len(details) > 5:
            summary += f"; and {len(details) - 5:,} more"
        raise PreparationError(
            f"Materialization output verification failed for "
            f"{len(request_errors) + len(failed):,} unit(s): {summary}"
        )

    total = sum(check.actual_uint32_values for check in checks)
    output.print(
        Text.assemble(
            ("✓", "bold green"),
            f" materialization verified  {len(checks):,}/{len(rows):,} units · ",
            (_human_token_count(total), "bold"),
            " tokens",
            (f"  {_elapsed_time(started_at)}", "dim"),
        )
    )
    return sorted(checks, key=lambda check: check.unit_id)


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
        ("upload resharding runtime", _runtime_transfer_command(args, ())),
        ("install and validate resharding runtime", _runtime_validation_command(args, ())),
        ("dispatch and stop workers when done", _map_command(args, script_dir)),
    ]


def _print_dispatch(
    label: str,
    rows: Sequence[dict[str, str]],
    script_dir: Path,
    lifecycle_commands: Sequence[tuple[str, Sequence[str]]],
    worker_count: int,
    execute: bool,
    *,
    cluster: str | None = None,
    project: str | None = None,
    region: str | None = None,
) -> None:
    category_count = len({row["leaf_id"] for row in rows})
    planned_tokens = sum(int(row["planned_uint32_values"]) for row in rows)
    largest_unit = max(int(row["estimated_peak_local_bytes"]) for row in rows)
    display_label = label.removeprefix("category-") if label.startswith("category-") else label

    def count(value: int, noun: str) -> str:
        return f"{value:,} {noun if value == 1 else noun + 's'}"

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", no_wrap=True)
    summary.add_column()
    summary.add_row("Selection", display_label)
    summary.add_row(
        "Work",
        " · ".join(
            (
                count(category_count, "category"),
                count(len(rows), "unit"),
                count(worker_count, "worker"),
            )
        ),
    )
    summary.add_row("Output", f"{_human_token_count(planned_tokens)} tokens")
    summary.add_row("Largest working set", _human_byte_count(largest_unit))
    if cluster:
        summary.add_row("Cluster", cluster)
    if project:
        summary.add_row("Project", project)
    if region:
        summary.add_row("Region", region)
    output = Console(highlight=False)
    output.print(Text("Execution" if execute else "Dry run", style="bold"))
    output.print(summary)
    if execute:
        return
    print("\ncommands:")
    for _, command in lifecycle_commands:
        print(shlex.join(command))
    print("\nunits:")
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
        status_run_id = f"{int(time.time())}-{uuid.uuid4().hex[:12]}" if args.execute else None
        script_dir = _stage_selection(build, label, selected, status_run_id)
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
            cluster=args.cluster,
            project=args.project,
            region=args.region,
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
                raise PreparationError(f"Required worker runtime file is missing or unsafe: {required_path}")
        if args.profile:
            os.environ["AWS_PROFILE"] = args.profile
        print(f"preflight=passed created_at={preflight_created_at}")
        worker_ids = _prepare_workers(args, worker_count)
        try:
            _run_lifecycle_command(
                "upload storage setup",
                _storage_transfer_command(args, worker_ids),
                verbose=args.verbose,
            )
            _run_lifecycle_command(
                "prepare local NVMe",
                _storage_setup_command(args, worker_ids, selected),
                verbose=args.verbose,
            )
            _run_lifecycle_command(
                "install Dolma and s5cmd",
                _runtime_setup_command(args, worker_ids),
                verbose=args.verbose,
            )
            _run_lifecycle_command(
                "upload resharding runtime",
                _runtime_transfer_command(args, worker_ids),
                verbose=args.verbose,
            )
            _run_lifecycle_command(
                "install and validate resharding runtime",
                _runtime_validation_command(args, worker_ids),
                verbose=args.verbose,
            )
            _run_lifecycle_command(
                "submit materialization",
                _map_command(args, script_dir, worker_ids),
                verbose=args.verbose,
            )
        except BaseException:
            _pause_workers_after_failure(args, worker_ids)
            raise
        assert status_run_id is not None
        _wait_for_workers_to_stop(
            args,
            worker_ids,
            len(selected),
            status_run_id,
        )
        _verify_materialized_units(args, selected)
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
