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
import copy
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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, NoReturn, TypeVar
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
        _count_label,
        _filter_execution_units,
        _human_byte_count,
        _human_token_count,
        _list_prefix,
        _pair_metadata_key,
        _unit_selection_digest,
        _validate_execution_layout,
        _validate_preparation_build,
        normalize_region,
        preflight_build,
    )
except ImportError:
    from workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        UINT32_BYTES,
        PreparationError,
        _count_label,
        _filter_execution_units,
        _human_byte_count,
        _human_token_count,
        _list_prefix,
        _pair_metadata_key,
        _unit_selection_digest,
        _validate_execution_layout,
        _validate_preparation_build,
        normalize_region,
        preflight_build,
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
        "--exclude-category",
        action="append",
        default=[],
        metavar="SELECTOR",
        help=(
            "exclude an exact mix name, leaf ID, or MIX_NAME::CATEGORY_NAME from --all; "
            "repeat for multiple completed categories"
        ),
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
        help="maximum number of materialization workers kept active concurrently",
    )
    parser.add_argument(
        "--provision-batch-size",
        type=lambda value: _positive_integer(value, "provision-batch-size"),
        default=DEFAULT_PROVISION_BATCH_SIZE,
        help="maximum VM lifecycle requests submitted concurrently in one batch",
    )
    parser.add_argument(
        "--provision-batch-delay-seconds",
        type=lambda value: _nonnegative_float(value, "provision-batch-delay-seconds"),
        default=DEFAULT_PROVISION_BATCH_DELAY_SECONDS,
        metavar="SECONDS",
        help="delay between VM lifecycle batches so provider API quotas can refill",
    )
    parser.add_argument(
        "--bootstrap-parallelism",
        type=lambda value: _positive_integer(value, "bootstrap-parallelism"),
        default=32,
        help="maximum ready workers bootstrapped and dispatched concurrently",
    )
    parser.add_argument(
        "--readiness-poll-seconds",
        type=lambda value: _positive_integer(value, "readiness-poll-seconds"),
        default=10,
        metavar="SECONDS",
        help="interval for detecting newly ready workers",
    )
    parser.add_argument(
        "--instance-type",
        help="override the worker type selected by the execution plan",
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
        choices=("auto", "single", "raid0"),
        default="auto",
        help="local NVMe layout; auto uses the execution plan",
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
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run the selected read-only preflight immediately before --execute",
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


def _nonnegative_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{name} must not be negative")
    return parsed


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKER_STORAGE_SCRIPT = Path(__file__).with_name("setup_worker_storage.sh")
RESHARD_MODULE = REPOSITORY_ROOT / "python/dolma/tokenizer/reshard.py"
DOCUMENT_SELECTION_MODULE = (
    REPOSITORY_ROOT / "python/dolma/tokenizer/document_selection.py"
)
REMOTE_STORAGE_SCRIPT = "/tmp/dolma3p5-setup-worker-storage.sh"
REMOTE_RESHARD_MODULE = "/tmp/dolma3p5-runtime/reshard.py"
REMOTE_DOCUMENT_SELECTION_MODULE = "/tmp/dolma3p5-runtime/document_selection.py"
# setup_worker_storage.sh mounts the worker's local NVMe here (DOLMA_WORK_MOUNT).
WORKER_LOCAL_MOUNT = "/mnt/dolma"
# Must equal the `plan.py --local-temp-root` used to build the plan. That value is
# recorded only inside each per-unit config YAML (as `local_tempdir`), never in
# config-index.csv or dataset-layout.json, so this script cannot derive it from the
# artifacts it loads. A different --local-temp-root therefore passes planning and
# only fails here, at worker bootstrap.
WORKER_LOCAL_TEMP_ROOT = f"{WORKER_LOCAL_MOUNT}/dolma3p5-resharding"
# Interpreter installed on every worker by poormanray's setup-dolma step.
WORKER_PYTHON_BIN = "$HOME/.venv/bin/python"
# Per-run status and log directory. The launcher payload writes it and the progress
# monitor reads it back, so both must resolve to exactly the same remote path.
WORKER_STATUS_ROOT = "$HOME/dolma3p5-resharding-status"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DOLMA_LOG_PREFIX = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [^\]]+ "
    r"(?P<level>TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s*"
)
COMPACT_LOG_PREFIX = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*"
)
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
WORKER_LOG_PAGE_BYTES = 16 * 1024
# EC2 accepts at most this many instance IDs per DescribeInstanceStatus call.
DESCRIBE_INSTANCE_STATUS_ID_LIMIT = 100
# Poormanray resolves a discovery group to instance IDs before applying explicit
# ones, so a group must stay under the limit above. Held 10 instances below it so
# a group that grows slightly cannot reach the hard limit.
PMR_DISCOVERY_WORKER_LIMIT = DESCRIBE_INSTANCE_STATUS_ID_LIMIT - 10
LIFECYCLE_MAX_ATTEMPTS = 5
# Argparse defaults, shared with the getattr fallbacks used for callers that build
# a Namespace without these attributes.
DEFAULT_PROVISION_BATCH_SIZE = 5
DEFAULT_PROVISION_BATCH_DELAY_SECONDS = 3.0
# EC2 applies tags asynchronously; wait this long for them to become readable.
TAG_PROPAGATION_TIMEOUT_SECONDS = 60
# One status/log page must come back before the next monitor poll is due.
WORKER_LOG_COMMAND_TIMEOUT_SECONDS = 120
# Instance IDs share a long common prefix, so only the tail identifies a worker.
WORKER_TAG_LENGTH = 6
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


@dataclass(frozen=True)
class PlannedWorkerGroup:
    instance_type: str
    storage_layout: str
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class MaterializationGroup:
    args: argparse.Namespace
    rows: tuple[dict[str, str], ...]
    script_dir: Path
    worker_count: int


@dataclass(frozen=True)
class WorkerAssignment:
    group: MaterializationGroup
    rows: tuple[dict[str, str], ...]
    script_dir: Path


_BatchItem = TypeVar("_BatchItem")


def _provision_batch_size(args: argparse.Namespace) -> int:
    """Return the lifecycle batch size, defaulted for hand-built Namespaces."""

    return int(getattr(args, "provision_batch_size", DEFAULT_PROVISION_BATCH_SIZE))


def _provision_batch_delay(args: argparse.Namespace) -> float:
    """Return the delay between lifecycle batches, defaulted for hand-built Namespaces."""

    return float(
        getattr(
            args,
            "provision_batch_delay_seconds",
            DEFAULT_PROVISION_BATCH_DELAY_SECONDS,
        )
    )


def _lifecycle_batches(
    items: Sequence[_BatchItem], batch_size: int
) -> list[list[_BatchItem]]:
    """Split one lifecycle request into bounded provider-API batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        list(items[offset : offset + batch_size])
        for offset in range(0, len(items), batch_size)
    ]


def _worker_tag(instance_id: str) -> str:
    """Return the short display tag for one worker; instance IDs share a long prefix."""

    return instance_id[-WORKER_TAG_LENGTH:]


def _remote_status_root(status_run_id: str) -> str:
    """Return the remote status directory shared by the launchers and the monitor."""

    return f"{WORKER_STATUS_ROOT}/{status_run_id}"


def _worker_log_line(
    tag: str,
    style: str,
    message: str,
    *,
    timestamp: str | None = None,
    bold: bool = False,
) -> Text:
    line = Text()
    rendered_timestamp = timestamp or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    line.append(f"[{rendered_timestamp}]", style="dim")
    line.append(" ")
    line.append(f"[{tag}]", style=style)
    line.append(" ")
    line.append(message, style="bold" if bold else None)
    return line


def _worker_log_parts(message: str) -> tuple[str | None, str]:
    """Extract the source timestamp and remove logger metadata from one worker line."""

    match = DOLMA_LOG_PREFIX.match(message)
    if match is not None:
        rendered = message[match.end() :]
        if match.group("level") in {"WARNING", "ERROR", "CRITICAL"}:
            rendered = f"{match.group('level').lower()} · {rendered}"
        return match.group("timestamp"), rendered

    compact_match = COMPACT_LOG_PREFIX.match(message)
    if compact_match is not None:
        return compact_match.group("timestamp"), message[compact_match.end() :]
    return None, message


def _worker_log_message(message: str) -> str:
    """Remove Dolma's module and level metadata from one worker line."""

    return _worker_log_parts(message)[1]


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
    """Apply bounded discovery, accounting, and cluster tags."""

    if not instance_ids:
        return
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("ec2", region_name=args.region)
    required_tags = {
        # Poormanray discovers existing AWS instances through this tag before
        # applying explicit instance IDs. Keep each discovery group below the
        # provider's 100-ID DescribeInstanceStatus limit.
        "project": _pmr_discovery_name(args),
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

    deadline = time.monotonic() + TAG_PROPAGATION_TIMEOUT_SECONDS
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
            all(
                observed.get(instance_id, {}).get(key) == value
                for key, value in required_tags.items()
            )
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
            return (
                f"{match.group('ready')} · {elapsed}"
                if elapsed
                else match.group("ready")
            )
        if line.startswith(("·", "•")):
            return None
    if line.startswith("[INFO]"):
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
    live: bool = True,
) -> int:
    """Run a command with one in-place status and a bounded failure log."""

    output = console or Console(stderr=True, highlight=False)
    started_at = time.monotonic()
    tail: deque[str] = deque(maxlen=PROCESS_TAIL_LINES)
    process: subprocess.Popen[str] | None = None
    live_status = (
        output.status(_stage_status(stage), spinner="dots")
        if output.is_terminal and not verbose and live
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
        output.print(
            Text.assemble(("✓", "bold green"), " ", stage, (f"  {duration}", "dim"))
        )
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
    console: Console | None = None,
    verbose: bool = False,
    live: bool = True,
) -> None:
    try:
        return_code = _run_compact_process(
            stage,
            command,
            console=console,
            verbose=verbose,
            live=live,
        )
    except OSError as exc:
        raise PreparationError(f"could not start {stage}: {exc}") from exc
    if return_code:
        raise PreparationError(f"{stage} failed with exit code {return_code}")


def _instance_options(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        # Poormanray selects existing AWS instances through the canonical
        # `project` tag supplied as --name. `ai2-project` remains the accounting
        # project and is passed independently as --project.
        "cluster": _pmr_discovery_name(args),
        "project": args.project,
        "region": args.region,
        "instance_ids": instance_ids,
        "parallelism": (
            min(args.parallelism, len(instance_ids))
            if instance_ids
            else args.parallelism
        ),
        "ssh_key_path": args.ssh_key_path,
    }


def _pmr_discovery_name(args: argparse.Namespace) -> str:
    """Return the bounded poormanray discovery group for one worker class."""

    instance_type = getattr(args, "instance_type", None)
    storage_layout = getattr(args, "storage_layout", None)
    if not instance_type or not storage_layout or storage_layout == "auto":
        return args.cluster
    suffix = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{instance_type}-{storage_layout}".lower(),
    ).strip("-")
    return f"{args.cluster}-{suffix}"


def _create_command(
    args: argparse.Namespace,
    number: int,
    *,
    detach: bool = False,
) -> list[str]:
    return build_poormanray_create_command(
        cluster=args.cluster,
        project=args.project,
        region=args.region,
        number=number,
        instance_type=args.instance_type,
        storage_type=args.root_storage_type,
        storage_size_gib=args.root_storage_size,
        parallelism=min(_provision_batch_size(args), number),
        detach=detach,
        ssh_key_path=args.ssh_key_path,
    )


def _provision_batches(worker_count: int, batch_size: int) -> tuple[int, ...]:
    """Split worker creation into bounded provider-API launch batches."""

    if worker_count < 0:
        raise ValueError("worker_count must not be negative")
    return tuple(
        len(batch) for batch in _lifecycle_batches(range(worker_count), batch_size)
    )


def _wait_command(args: argparse.Namespace, instance_ids: Sequence[str]) -> list[str]:
    return build_poormanray_instance_command(
        "wait",
        cluster=_pmr_discovery_name(args),
        project=args.project,
        region=args.region,
        instance_ids=instance_ids,
        ssh_key_path=args.ssh_key_path,
    )


def _resume_command(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    *,
    detach: bool = False,
) -> list[str]:
    options = _instance_options(args, instance_ids)
    options.pop("ssh_key_path")
    return build_poormanray_instance_command("resume", detach=detach, **options)


def _resume_workers_in_batches(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    *,
    detach: bool = True,
    console: Console | None = None,
) -> None:
    """Resume stopped workers without bursting the provider's start API."""

    ordered = sorted(set(instance_ids))
    if not ordered:
        return
    delay = _provision_batch_delay(args)
    batches = _lifecycle_batches(ordered, _provision_batch_size(args))
    output = console or Console(stderr=True, highlight=False)
    discovery_name = _pmr_discovery_name(args)

    for batch_index, batch in enumerate(batches, start=1):
        _resume_worker_batch(
            args,
            batch,
            stage=(
                f"resume {discovery_name} batch "
                f"{batch_index:,}/{len(batches):,}"
            ),
            detach=detach,
            console=output,
        )

        if batch_index < len(batches) and delay:
            time.sleep(delay)


def _worker_states(args: argparse.Namespace) -> dict[str, str]:
    """Return the provider's current state for every worker in the cluster."""

    return {
        instance.instance_id: instance.state
        for instance in _describe_cluster_instances(
            args.cluster, args.region, args.profile
        )
    }


def _retry_lifecycle_batch(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    *,
    run_attempt: Callable[[list[str]], bool],
    describe_states: Callable[[list[str]], dict[str, str]],
    retry_states: frozenset[str],
    on_exhausted: Callable[[list[str], int], None],
    on_retry: Callable[[list[str], float], None] | None = None,
) -> None:
    """Retry one provider-sized lifecycle batch until nothing is left to change.

    `run_attempt` reports True once the batch succeeded. After a failed attempt
    only the workers the provider still reports in `retry_states` are retried,
    backing off exponentially from the configured inter-batch delay.
    `on_exhausted` replaces the final retry and owns how that failure surfaces.
    """

    remaining = list(instance_ids)
    delay = _provision_batch_delay(args)
    for attempt in range(1, LIFECYCLE_MAX_ATTEMPTS + 1):
        if run_attempt(remaining):
            return
        states = describe_states(remaining)
        remaining = [
            instance_id
            for instance_id in remaining
            if states.get(instance_id) in retry_states
        ]
        if not remaining:
            return
        if attempt == LIFECYCLE_MAX_ATTEMPTS:
            on_exhausted(remaining, attempt)
            return
        retry_delay = max(1.0, delay) * (2 ** (attempt - 1))
        if on_retry is not None:
            on_retry(remaining, retry_delay)
        time.sleep(retry_delay)


def _resume_worker_batch(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    *,
    stage: str,
    detach: bool,
    console: Console,
) -> None:
    """Resume one provider-sized batch, retrying only workers still stopped."""

    def run_attempt(remaining: list[str]) -> bool:
        try:
            _run_lifecycle_command(
                stage,
                _resume_command(args, remaining, detach=detach),
                console=console,
                # Per-instance PMR messages overwhelm the materialization logs.
                verbose=False,
                live=False,
            )
        except PreparationError:
            return False
        return True

    def on_exhausted(remaining: list[str], attempt: int) -> None:
        raise PreparationError(
            f"Could not resume {len(remaining):,} worker(s) in {stage} "
            f"after {attempt:,} attempts: {', '.join(remaining)}"
        )

    def on_retry(remaining: list[str], retry_delay: float) -> None:
        console.print(
            Text.assemble(
                ("↻", "bold yellow"),
                f" {stage} incomplete; retrying {len(remaining):,} worker(s) ",
                (f"in {retry_delay:g}s", "dim"),
            )
        )

    _retry_lifecycle_batch(
        args,
        instance_ids,
        run_attempt=run_attempt,
        describe_states=lambda _remaining: _worker_states(args),
        retry_states=frozenset({"stopped"}),
        on_exhausted=on_exhausted,
        on_retry=on_retry,
    )


def _resume_worker_groups_in_batches(
    groups: Sequence[tuple[argparse.Namespace, Sequence[str]]],
    *,
    console: Console,
) -> None:
    """Resume all worker classes in globally bounded, round-robin waves."""

    if not groups:
        return
    batch_size = min(_provision_batch_size(group_args) for group_args, _ in groups)
    delay = max(_provision_batch_delay(group_args) for group_args, _ in groups)
    queues = deque(
        (group_args, deque(sorted(set(instance_ids))))
        for group_args, instance_ids in groups
        if instance_ids
    )
    waves: list[list[tuple[argparse.Namespace, list[str]]]] = []
    while queues:
        wave_by_group: dict[int, tuple[argparse.Namespace, list[str]]] = {}
        for _ in range(batch_size):
            if not queues:
                break
            group_args, instance_queue = queues.popleft()
            key = id(group_args)
            if key not in wave_by_group:
                wave_by_group[key] = (group_args, [])
            wave_by_group[key][1].append(instance_queue.popleft())
            if instance_queue:
                queues.append((group_args, instance_queue))
        waves.append(list(wave_by_group.values()))

    for wave_index, wave in enumerate(waves, start=1):
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [
                pool.submit(
                    _resume_worker_batch,
                    group_args,
                    instance_ids,
                    stage=(
                        f"resume batch {wave_index:,}/{len(waves):,} · "
                        f"{_pmr_discovery_name(group_args)}"
                    ),
                    detach=True,
                    console=console,
                )
                for group_args, instance_ids in wave
            ]
            for future in as_completed(futures):
                future.result()
        if wave_index < len(waves) and delay:
            time.sleep(delay)


def _pause_command(args: argparse.Namespace, instance_ids: Sequence[str]) -> list[str]:
    options = _instance_options(args, instance_ids)
    options.pop("ssh_key_path")
    return build_poormanray_instance_command("pause", **options)


def _worker_minimum_available_bytes(rows: Sequence[dict[str, str]]) -> int:
    largest_working_set = max(int(row["estimated_peak_local_bytes"]) for row in rows)
    # Integer ceiling of 110% of the largest working set: 10% headroom over the
    # plan's own estimate, rounded up so no worker is provisioned a byte short.
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
        f'python_bin="{WORKER_PYTHON_BIN}"; '
        'module_dir=$("$python_bin" -P -c \'import pathlib, dolma.tokenizer; '
        "print(pathlib.Path(dolma.tokenizer.__file__).parent)'); "
        'install -m 0644 /tmp/dolma3p5-runtime/reshard.py "$module_dir/reshard.py"; '
        "install -m 0644 /tmp/dolma3p5-runtime/document_selection.py "
        '"$module_dir/document_selection.py"; '
        '"$python_bin" -P -c \'from dolma.tokenizer.reshard import '
        "RESHARDING_MANIFEST_SCHEMA_VERSION; assert RESHARDING_MANIFEST_SCHEMA_VERSION == 2'; "
        f"s5cmd version; findmnt {WORKER_LOCAL_MOUNT}; "
        f"test -w {WORKER_LOCAL_TEMP_ROOT}"
    )
    return build_poormanray_run_command(
        remote_command=remote_command,
        **_instance_options(args, instance_ids),
    )


def _map_command(
    args: argparse.Namespace,
    script_dir: Path,
    instance_ids: Sequence[str] = (),
    *,
    spindown: bool = True,
) -> list[str]:
    return build_poormanray_map_command(
        cluster=_pmr_discovery_name(args),
        project=args.project,
        region=args.region,
        script_dir=script_dir,
        spindown=spindown,
        instance_ids=instance_ids,
        ssh_key_path=args.ssh_key_path,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(
            f"Required execution artifact is missing or unsafe: {path}"
        )
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
        "planned_output_shard_count",
        "estimated_peak_local_bytes",
        "worker_instance_type",
        "worker_storage_layout",
        "worker_vcpus",
        # Read only during post-run verification; validating it up front keeps a
        # stale plan from failing after the workers have already been paid for.
        "allowed_materialized_target_residual_uint32_values",
    }
    missing = required - set(rows[0])
    if missing:
        raise PreparationError("Execution index is missing columns: " + ", ".join(sorted(missing)))
    return rows


def _planned_worker_groups(args: argparse.Namespace, rows: Sequence[dict[str, str]]) -> list[PlannedWorkerGroup]:
    """Group selected units by their planned i4i worker configuration."""

    if args.instance_type:
        if args.storage_layout == "auto":
            planned_layouts = {
                row["worker_storage_layout"] for row in rows if row["worker_instance_type"] == args.instance_type
            }
            planned_types = {row["worker_instance_type"] for row in rows}
            if planned_types != {args.instance_type} or len(planned_layouts) != 1:
                raise PreparationError(
                    "--storage-layout must be explicit when --instance-type overrides " "the execution plan"
                )
            storage_layout = planned_layouts.pop()
        else:
            storage_layout = args.storage_layout
        return [
            PlannedWorkerGroup(
                instance_type=args.instance_type,
                storage_layout=storage_layout,
                rows=tuple(rows),
            )
        ]

    if args.storage_layout != "auto":
        raise PreparationError("--storage-layout can only override the plan together with --instance-type")
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        layout = row["worker_storage_layout"]
        if layout not in {"single", "raid0"}:
            raise PreparationError(f"Invalid planned storage layout for {row['unit_id']}: {layout}")
        key = (
            row["worker_instance_type"],
            layout,
            int(row["worker_vcpus"]),
        )
        grouped[key].append(row)
    return [
        PlannedWorkerGroup(instance_type, layout, tuple(group_rows))
        for (instance_type, layout, _), group_rows in sorted(
            grouped.items(), key=lambda item: (item[0][2], item[0][0])
        )
    ]


def _worker_counts_for_groups(
    groups: Sequence[PlannedWorkerGroup], maximum_workers: int
) -> list[int]:
    """Allocate the global worker limit while keeping every worker group active."""

    if not groups:
        return []
    if maximum_workers < len(groups):
        raise PreparationError(
            f"--parallelism must be at least {len(groups):,} to run all planned "
            "worker groups concurrently"
        )

    capacities = [
        min(len(group.rows), PMR_DISCOVERY_WORKER_LIMIT) for group in groups
    ]
    total_capacity = sum(capacities)
    remaining = min(maximum_workers, total_capacity) - len(groups)
    counts = [1] * len(groups)
    while remaining:
        allocated = False
        for index, group in enumerate(groups):
            if counts[index] >= capacities[index]:
                continue
            counts[index] += 1
            remaining -= 1
            allocated = True
            if not remaining:
                break
        if not allocated:
            break
    return counts


def _category_selector(row: dict[str, str]) -> str:
    return f"{row['mix_name']}::{row['category_name']}"


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
                "planned_output_shard_count": sum(int(row["planned_output_shard_count"]) for row in units),
                "largest_estimated_peak_local_bytes": max(int(row["estimated_peak_local_bytes"]) for row in units),
                "worker_instance_types": ",".join(sorted({row["worker_instance_type"] for row in units})),
            }
        )
    return sorted(output, key=lambda row: (int(row["leaf_id"].split(":", 1)[0]), row["leaf_id"]))


def _print_categories(rows: Sequence[dict[str, str]], filter_text: str) -> None:
    needle = filter_text.casefold()
    matches = [
        row
        for row in _category_rows(rows)
        if not needle
        or needle in " ".join((row["leaf_id"], row["mix_name"], row["category_name"], row["selector"])).casefold()
    ]
    if not matches:
        raise PreparationError(f"No categories match: {filter_text}")
    print("leaf_id\texecution_units\toutput_shards\tplanned_tokens\tlargest_unit\tworkers\tselector")
    for row in matches:
        print(
            f"{row['leaf_id']}\t{row['execution_units']:,}\t"
            f"{row['planned_output_shard_count']:,}\t"
            f"{_human_token_count(int(row['planned_uint32_values']))}\t"
            f"{_human_byte_count(int(row['largest_estimated_peak_local_bytes']))}\t"
            f"{row['worker_instance_types']}\t"
            f"{row['selector']}"
        )


def _select_units(
    args: argparse.Namespace, rows: Sequence[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    exclusions = list(getattr(args, "exclude_category", ()))
    if exclusions and not args.all:
        raise PreparationError("--exclude-category can only be used with --all")
    if args.all:
        excluded_unit_ids: set[str] = set()
        for selector in exclusions:
            excluded_unit_ids.update(
                row["unit_id"]
                for row in _filter_execution_units(rows, category=selector)
            )
        selected = [row for row in rows if row["unit_id"] not in excluded_unit_ids]
        if not selected:
            raise PreparationError("Category exclusions removed every execution unit")
        label = "all" if not exclusions else f"all-except-{len(exclusions)}-categories"
        return label, selected
    if args.unit:
        selected = _filter_execution_units(rows, unit=args.unit)
        return f"unit-{args.unit}", selected

    selector = str(args.category)
    selected = _filter_execution_units(rows, category=selector)
    return f"category-{selector}", selected


def _safe_slug(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    return "-".join(part for part in slug.split("-") if part)[:80] or "selection"


def _resolve_launcher(build: Path, row: dict[str, str]) -> Path:
    launcher = (build / row["launcher_path"]).resolve()
    launcher_root = (build / "01-plan/execution/launcher-scripts").resolve()
    if (
        launcher.parent != launcher_root
        or launcher.is_symlink()
        or not launcher.is_file()
    ):
        raise PreparationError(
            f"Unsafe or missing launcher for {row['unit_id']}: {launcher}"
        )
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
        raise PreparationError(
            f"Worker launcher is missing strict shell mode: {launcher}"
        )
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
            f'export DOLMA_STATUS_ROOT="{_remote_status_root(status_run_id)}"'
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
    for row, _, _, launcher_digest in sorted(
        launchers, key=lambda item: item[0]["unit_id"]
    ):
        digest.update(row["unit_id"].encode())
        digest.update(launcher_digest)
    selection_name = f"{_safe_slug(label)}-{digest.hexdigest()[:12]}"
    dispatch_root = build / "01-plan/execution/dispatch"
    if dispatch_root.exists() and (
        dispatch_root.is_symlink() or not dispatch_root.is_dir()
    ):
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
            raise PreparationError(
                f"Existing dispatch selection does not match the plan: {selection_dir}"
            )
        return selection_dir

    selection_dir.mkdir(exist_ok=False)
    for _, launcher, payload, _ in launchers:
        destination = selection_dir / launcher.name
        destination.write_bytes(payload)
        destination.chmod(launcher.stat().st_mode & 0o777)
    return selection_dir


def _stage_worker_assignments(
    group: MaterializationGroup,
) -> tuple[WorkerAssignment, ...]:
    """Create a largest-first queue of one-script poormanray directories."""

    assignments: list[WorkerAssignment] = []
    assignment_root = group.script_dir.parent / f"{group.script_dir.name}-assignments"
    if assignment_root.exists():
        if assignment_root.is_symlink() or not assignment_root.is_dir():
            raise PreparationError(
                f"Unsafe worker-assignment path: {assignment_root}"
            )
    else:
        assignment_root.mkdir(exist_ok=False)

    ordered = sorted(
        group.rows,
        key=lambda row: (
            int(row.get("estimated_work_uint32_values") or row["planned_uint32_values"]),
            row["unit_id"],
        ),
        reverse=True,
    )
    for row in ordered:
        unit_id = row["unit_id"]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", unit_id) or unit_id in {".", ".."}:
            raise PreparationError(f"Unsafe execution-unit ID: {unit_id!r}")
        launcher = group.script_dir / Path(row["launcher_path"]).name
        if launcher.is_symlink() or not launcher.is_file():
            raise PreparationError(
                f"Staged execution-unit launcher is missing or unsafe: {launcher}"
            )
        assignment_dir = assignment_root / unit_id
        destination = assignment_dir / launcher.name
        if assignment_dir.exists():
            if assignment_dir.is_symlink() or not assignment_dir.is_dir():
                raise PreparationError(
                    f"Unsafe worker-assignment directory: {assignment_dir}"
                )
            children = list(assignment_dir.iterdir())
            if (
                len(children) != 1
                or children[0] != destination
                or destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != launcher.read_bytes()
                or (destination.stat().st_mode & 0o777)
                != (launcher.stat().st_mode & 0o777)
            ):
                raise PreparationError(
                    f"Existing worker assignment does not match the plan: {assignment_dir}"
                )
        else:
            assignment_dir.mkdir(exist_ok=False)
            destination.write_bytes(launcher.read_bytes())
            destination.chmod(launcher.stat().st_mode & 0o777)
        assignments.append(WorkerAssignment(group, (row,), assignment_dir))
    return tuple(assignments)


def _require_preflight(build: Path, selected: Sequence[dict[str, str]]) -> str:
    summary_path = build / "02-preflight/preflight-summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise PreparationError(
            "A passing preflight is required before --execute. Add --preflight to this command "
            "or run python scripts/dolma3p5_resharding/preflight.py immediately before dispatch."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid preflight summary: {summary_path}") from exc
    blocking_fields = ("drifted_input_objects", "occupied_destinations", "errors")
    if not summary.get("passed") or any(
        int(summary.get(name, 0)) for name in blocking_fields
    ):
        raise PreparationError(f"Preflight did not pass: {summary_path}")
    selected_digest = _unit_selection_digest(selected)
    preflight_scope = summary.get("selection_scope")
    if (
        preflight_scope != "all"
        and summary.get("selected_unit_ids_sha256") != selected_digest
    ):
        raise PreparationError(
            "Preflight selection does not match this dispatch. Rerun preflight for the same "
            "materialization selection."
        )

    destination_rows = _read_csv(build / "02-preflight/destination-status.csv")
    statuses = {row["unit_id"]: row["status"] for row in destination_rows}
    invalid = [
        row["unit_id"] for row in selected if statuses.get(row["unit_id"]) != "empty"
    ]
    if invalid:
        raise PreparationError(
            f"Preflight does not show an empty destination for {len(invalid):,} selected unit(s)"
        )
    return str(summary.get("created_at", "unknown"))


def _run_selected_preflight(
    args: argparse.Namespace,
    build: Path,
    selected: Sequence[dict[str, str]],
) -> None:
    preflight_build(
        argparse.Namespace(
            build=build,
            profile=args.profile,
            region=args.region,
            max_workers=None,
            category=args.category,
            unit=args.unit,
            selected_unit_ids=tuple(row["unit_id"] for row in selected),
            quiet=True,
        )
    )


class _WorkerCleanupAbandoned(Exception):
    """Best-effort worker cleanup stopped after printing its own instructions."""


def _abandon_worker_cleanup(reason: str, command: Sequence[str]) -> NoReturn:
    """Print the manual command that must replace the failed cleanup, then stop."""

    print(
        f"WARNING: {reason}; run {shlex.join(command)} immediately",
        file=sys.stderr,
    )
    raise _WorkerCleanupAbandoned(reason)


def _pause_worker_batch(
    args: argparse.Namespace, instance_ids: Sequence[str], *, stage: str
) -> None:
    """Pause one provider-sized batch, retrying only workers still active."""

    def run_attempt(remaining: list[str]) -> bool:
        command = _pause_command(args, remaining)
        try:
            return_code = _run_compact_process(stage, command, verbose=False)
        except OSError as exc:
            _abandon_worker_cleanup(f"worker cleanup could not start: {exc}", command)
        return return_code == 0

    def describe_states(remaining: list[str]) -> dict[str, str]:
        try:
            return _worker_states(args)
        except Exception as exc:
            _abandon_worker_cleanup(
                f"worker cleanup state check failed: {exc}",
                _pause_command(args, remaining),
            )

    def on_exhausted(remaining: list[str], attempt: int) -> None:
        _abandon_worker_cleanup(
            f"worker cleanup failed after {attempt} attempts",
            _pause_command(args, remaining),
        )

    _retry_lifecycle_batch(
        args,
        instance_ids,
        run_attempt=run_attempt,
        describe_states=describe_states,
        retry_states=frozenset({"pending", "running"}),
        on_exhausted=on_exhausted,
    )


def _pause_workers_after_failure(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> None:
    if not instance_ids:
        return
    ordered = sorted(set(instance_ids))
    delay = _provision_batch_delay(args)
    batches = _lifecycle_batches(ordered, _provision_batch_size(args))
    try:
        for batch_index, batch in enumerate(batches, start=1):
            _pause_worker_batch(
                args,
                batch,
                stage=(
                    f"pause workers after failure batch {batch_index}/{len(batches)}"
                ),
            )
            if batch_index < len(batches) and delay:
                time.sleep(delay)
    except _WorkerCleanupAbandoned:
        return


def _prepare_workers(
    args: argparse.Namespace,
    worker_count: int,
    *,
    owned_instance_ids: Sequence[str] = (),
    wait_for_ready: bool = True,
    delay_after_last_batch: bool = False,
    deferred_resume_ids: list[str] | None = None,
) -> list[str]:
    """Resume stopped compatible workers and create missing workers in batches."""

    try:
        before = _describe_cluster_instances(args.cluster, args.region, args.profile)
    except Exception as exc:
        raise PreparationError(
            f"Could not inspect poormanray cluster {args.cluster!r} in {args.region}: {exc}"
        ) from exc

    owned = set(owned_instance_ids)
    busy = [
        instance
        for instance in before
        if instance.state != "stopped" and instance.instance_id not in owned
    ]
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
            if deferred_resume_ids is None:
                _resume_workers_in_batches(args, selected_ids, detach=False)
            else:
                deferred_resume_ids.extend(selected_ids)

        missing = worker_count - len(selected_ids)
        batches = _provision_batches(missing, _provision_batch_size(args))
        created_count = 0
        for batch_index, batch_count in enumerate(batches, start=1):
            _run_lifecycle_command(
                f"create worker batch {batch_index}/{len(batches)}",
                _create_command(args, batch_count, detach=True),
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
            new_created = sorted(
                instance.instance_id
                for instance in created
                if instance.instance_id not in selected_ids
            )
            if len(new_created) != batch_count:
                selected_ids.extend(new_created)
                raise PreparationError(
                    f"Expected poormanray to create {batch_count:,} worker(s) in batch "
                    f"{batch_index:,}/{len(batches):,}, but found {len(new_created):,}"
                )
            created_names = {
                instance_id: (
                    f"{args.cluster}-{next_name_index + created_count + offset:04d}"
                )
                for offset, instance_id in enumerate(new_created)
            }
            _retag_cluster_instances(args, new_created, names=created_names)
            selected_ids.extend(new_created)
            created_count += len(new_created)

            should_delay = batch_index < len(batches) or (
                delay_after_last_batch and batch_index == len(batches)
            )
            if should_delay:
                time.sleep(_provision_batch_delay(args))

        if len(selected_ids) != worker_count:
            raise PreparationError(
                f"Worker lifecycle selected {len(selected_ids):,} workers; expected {worker_count:,}"
            )
        if wait_for_ready:
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
    log_offsets: dict[tuple[str, str], int] | None = None,
    *,
    include_logs: bool = True,
) -> list[str]:
    status_root = _remote_status_root(status_run_id)
    active_instance_ids = set(instance_ids)
    offset_payload = json.dumps(
        {
            log_name: offset
            for (instance_id, log_name), offset in (log_offsets or {}).items()
            if instance_id in active_instance_ids
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    log_reader = r'''import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
offsets = json.loads(sys.argv[2])
budget = int(sys.argv[3])
for path in sorted(root.glob("*.log")):
    if budget <= 0:
        break
    size = path.stat().st_size
    offset = int(offsets.get(path.name, 0))
    if offset < 0 or offset > size:
        offset = 0
    if offset == size:
        continue
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(min(size - offset, budget))
    if offset + len(data) < size:
        boundary = max(data.rfind(b"\n"), data.rfind(b"\r"))
        if boundary >= 0:
            data = data[: boundary + 1]
    if not data:
        continue
    end = offset + len(data)
    print(f"@@DOLMA_LOG_BEGIN@@\t{path.name}\t{offset}\t{end}\t{size}")
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    print(f"@@DOLMA_LOG_END@@\t{path.name}\t{end}\tOK")
    budget -= len(data)
'''
    log_section = (
        f'''python_bin="${{DOLMA_PYTHON:-{WORKER_PYTHON_BIN}}}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=$(command -v python3.12 || command -v python3 || command -v python)
fi
"$python_bin" - "$status_root" {shlex.quote(offset_payload)} {WORKER_LOG_PAGE_BYTES} <<'PY'
{log_reader}
PY'''
        if include_logs
        else ""
    )
    remote_script = f"""status_root=\"{status_root}\"
shopt -s nullglob
status_files=(\"$status_root\"/*.status)
if (( ${{#status_files[@]}} == 0 )); then
  echo 'no unit status yet'
else
  for path in \"${{status_files[@]}}\"; do
    printf '@@DOLMA_STATUS@@\t%s\t' \"$(basename \"$path\" .status)\"
    tr -d '\n' < \"$path\"
    printf '\n'
  done
fi
{log_section}"""
    return build_poormanray_run_command(
        cluster=_pmr_discovery_name(args),
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
    log_offsets: dict[tuple[str, str], int] | None = None,
    *,
    include_logs: bool = True,
) -> dict[str, dict[str, Any]]:
    """Read one acknowledged page of new log bytes from every active worker."""

    try:
        result = subprocess.run(
            _worker_log_command(
                args,
                instance_ids,
                status_run_id,
                log_offsets,
                include_logs=include_logs,
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=WORKER_LOG_COMMAND_TIMEOUT_SECONDS,
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

    snapshots: dict[str, dict[str, Any]] = {}
    for instance_id, payload in instance_payloads.items():
        statuses: dict[str, str] = {}
        logs: dict[str, tuple[str, ...]] = {}
        offsets: dict[str, int] = {}
        current_log: str | None = None
        current_lines: list[str] = []
        for line in payload:
            if line.startswith("@@DOLMA_STATUS@@\t"):
                _, unit_id, status = line.split("\t", 2)
                statuses[unit_id] = status
                continue
            if line.startswith("@@DOLMA_LOG_BEGIN@@\t"):
                current_log = line.split("\t", 2)[1]
                current_lines = []
                continue
            if line.startswith("@@DOLMA_LOG_END@@\t"):
                fields = line.split("\t")
                if (
                    current_log is not None
                    and len(fields) == 4
                    and fields[1] == current_log
                    and fields[3] == "OK"
                ):
                    logs[current_log] = tuple(current_lines)
                    offsets[current_log] = int(fields[2])
                elif current_log is not None and len(fields) == 2:
                    # Accept envelopes written by the older full-log reader.
                    logs[current_log] = tuple(current_lines)
                current_log = None
                current_lines = []
                continue
            if current_log is not None:
                current_lines.append(line)
        snapshots[instance_id] = {
            "statuses": statuses,
            "logs": logs,
            "offsets": offsets,
        }
    return snapshots


@dataclass(frozen=True)
class _MonitorChannels:
    """Shared state the warm-worker dispatcher and the progress monitor exchange.

    Every field is optional so the monitor can also run standalone, with no
    dispatcher on the other side of these channels.
    """

    worker_stages: dict[str, str] | None = None
    stage_lock: Lock | None = None
    all_dispatched: Event | None = None
    abort: Event | None = None
    worker_args: dict[str, argparse.Namespace] | None = None
    unit_statuses: dict[tuple[str, str], str] | None = None
    unit_status_lock: Lock | None = None
    status_changed: Event | None = None

    @property
    def tracks_stages(self) -> bool:
        return self.worker_stages is not None

    @property
    def tracks_units(self) -> bool:
        return self.unit_statuses is not None

    @property
    def aborted(self) -> bool:
        return self.abort is not None and self.abort.is_set()

    @property
    def dispatch_complete(self) -> bool:
        return self.all_dispatched is None or self.all_dispatched.is_set()

    def stage_guard(self) -> AbstractContextManager[Any]:
        """Hold the dispatcher's stage lock, or nothing when running standalone."""

        return self.stage_lock if self.stage_lock is not None else nullcontext()

    def status_guard(self) -> AbstractContextManager[Any]:
        """Hold the dispatcher's unit-status lock, or nothing when standalone."""

        return (
            self.unit_status_lock
            if self.unit_status_lock is not None
            else nullcontext()
        )


@dataclass
class _WorkerOutputTracker:
    """Remember which worker statuses and log bytes have already been printed."""

    output: Console
    verbose: bool
    worker_tags: dict[str, tuple[str, str]]
    previous_statuses: dict[tuple[str, str], str] = field(default_factory=dict)
    emitted_log_lines: dict[tuple[str, str], int] = field(default_factory=dict)
    log_byte_offsets: dict[tuple[str, str], int] = field(default_factory=dict)
    announced_logs: set[tuple[str, str]] = field(default_factory=set)

    def record_statuses(
        self,
        channels: _MonitorChannels,
        instance_id: str,
        statuses: dict[str, str],
    ) -> bool:
        """Publish one worker's unit statuses; report whether any of them changed."""

        worker_tag, worker_style = self.worker_tags[instance_id]
        changed = False
        for unit_id, status in sorted(statuses.items()):
            assert isinstance(status, str)
            status_key = (instance_id, unit_id)
            if channels.unit_statuses is not None:
                with channels.status_guard():
                    if channels.unit_statuses.get(status_key) != status:
                        channels.unit_statuses[status_key] = status
                        changed = True
            if self.verbose and self.previous_statuses.get(status_key) != status:
                self.previous_statuses[status_key] = status
                self.output.print(
                    _worker_log_line(
                        worker_tag,
                        worker_style,
                        f"{unit_id} · {status}",
                        bold=True,
                    )
                )
        return changed

    def print_new_log_lines(self, instance_id: str, snapshot: dict[str, Any]) -> None:
        """Print only the log bytes this monitor has not already shown."""

        worker_tag, worker_style = self.worker_tags[instance_id]
        logs = snapshot["logs"]
        assert isinstance(logs, dict)
        offsets = snapshot.get("offsets", {})
        assert isinstance(offsets, dict)
        for log_name, log_lines in sorted(logs.items()):
            assert isinstance(log_lines, tuple)
            log_key = (instance_id, log_name)
            if log_name in offsets:
                new_lines = log_lines
            else:
                # Backward compatibility for old worker-log envelopes without
                # acknowledged byte offsets.
                emitted = self.emitted_log_lines.get(log_key, 0)
                if len(log_lines) < emitted:
                    emitted = 0
                new_lines = log_lines[emitted:]
            if new_lines:
                if log_key not in self.announced_logs:
                    self.announced_logs.add(log_key)
                    self.output.print(
                        _worker_log_line(
                            worker_tag,
                            worker_style,
                            f"unit {Path(log_name).stem}",
                            bold=True,
                        )
                    )
                for line in new_lines:
                    timestamp, message = _worker_log_parts(line)
                    self.output.print(
                        _worker_log_line(
                            worker_tag,
                            worker_style,
                            message,
                            timestamp=timestamp,
                        )
                    )
            if log_name in offsets:
                self.log_byte_offsets[log_key] = int(offsets[log_name])
            else:
                self.emitted_log_lines[log_key] = len(log_lines)


def _monitored_workers(
    describe_instances: Callable[[str, str, str | None], list[ClusterInstance]],
    args: argparse.Namespace,
    expected_ids: set[str],
) -> dict[str, ClusterInstance]:
    """Return the monitored workers, refusing to continue if any disappeared."""

    selected = {
        instance.instance_id: instance
        for instance in describe_instances(args.cluster, args.region, args.profile)
        if instance.instance_id in expected_ids
    }
    missing = expected_ids - set(selected)
    if missing:
        raise PreparationError(
            "Could not find selected materialization worker(s): "
            + ", ".join(sorted(missing))
        )
    return selected


def _reconcile_worker_stages(
    channels: _MonitorChannels, selected: dict[str, ClusterInstance]
) -> dict[str, str]:
    """Fold provider stop states into the shared stage map, then snapshot it."""

    stages = channels.worker_stages
    assert stages is not None
    with channels.stage_guard():
        for instance_id, instance in selected.items():
            base = stages.get(instance_id, "waiting").split(":", 1)[0]
            if base in {"materializing", "stopping"} and instance.state in {
                "stopping",
                "stopped",
            }:
                stages[instance_id] = instance.state
        return dict(stages)


def _unit_progress_detail(channels: _MonitorChannels, unit_count: int) -> str:
    """Describe unit completion, counting only what the dispatcher published."""

    if channels.unit_statuses is None:
        return f"{unit_count:,} units"
    with channels.status_guard():
        completed_units = sum(
            status == "succeeded" for status in channels.unit_statuses.values()
        )
    return f"{completed_units:,}/{unit_count:,} units"


def _lifecycle_progress_detail(
    channels: _MonitorChannels,
    lifecycle_counts: Counter[str],
    unit_count: int,
    started_at: float,
) -> str:
    """Summarize the dispatcher's own per-worker stages."""

    return " · ".join(
        (
            _unit_progress_detail(channels, unit_count),
            f"{lifecycle_counts['waiting']:,} waiting",
            f"{lifecycle_counts['queued']:,} queued",
            f"{lifecycle_counts['bootstrapping']:,} bootstrapping",
            f"{lifecycle_counts['materializing']:,} materializing",
            f"{lifecycle_counts['stopping']:,} stopping",
            f"{lifecycle_counts['stopped']:,} stopped",
            f"{lifecycle_counts['failed']:,} failed",
            _elapsed_time(started_at),
        )
    )


def _provider_progress_detail(
    state_counts: Counter[str], unit_count: int, started_at: float
) -> str:
    """Summarize provider instance states when no dispatcher stages exist."""

    return " · ".join(
        (
            f"{unit_count:,} units",
            f"{state_counts['running']:,} running",
            f"{state_counts['stopping']:,} stopping",
            f"{state_counts['stopped']:,} stopped",
            _elapsed_time(started_at),
        )
    )


def _materializing_worker_ids(
    channels: _MonitorChannels,
    selected: dict[str, ClusterInstance],
    stage_snapshot: dict[str, str],
) -> list[str]:
    """Return running workers whose logs and statuses are worth reading."""

    return sorted(
        instance.instance_id
        for instance in selected.values()
        if instance.state == "running"
        and (
            not channels.tracks_stages
            or stage_snapshot.get(instance.instance_id, "").split(":", 1)[0]
            == "materializing"
        )
    )


def _collect_worker_snapshots(
    args: argparse.Namespace,
    channels: _MonitorChannels,
    instance_ids: Sequence[str],
    status_run_id: str,
    log_offsets: dict[tuple[str, str], int],
    *,
    include_logs: bool,
) -> dict[str, dict[str, Any]]:
    """Read one status and log page per poormanray discovery group."""

    log_groups: dict[str, tuple[argparse.Namespace, list[str]]] = {}
    for instance_id in instance_ids:
        log_args = (channels.worker_args or {}).get(instance_id, args)
        selector = _pmr_discovery_name(log_args)
        if selector not in log_groups:
            log_groups[selector] = (log_args, [])
        log_groups[selector][1].append(instance_id)
    snapshots: dict[str, dict[str, Any]] = {}
    for log_args, group_instance_ids in log_groups.values():
        snapshots.update(
            _worker_log_snapshots(
                log_args,
                group_instance_ids,
                status_run_id,
                log_offsets,
                include_logs=include_logs,
            )
        )
    return snapshots


def _publish_worker_output(
    args: argparse.Namespace,
    channels: _MonitorChannels,
    tracker: _WorkerOutputTracker,
    running_ids: Sequence[str],
    status_run_id: str,
) -> None:
    """Print new worker output and hand fresh unit statuses to the dispatcher."""

    snapshots = _collect_worker_snapshots(
        args,
        channels,
        running_ids,
        status_run_id,
        tracker.log_byte_offsets,
        include_logs=tracker.verbose,
    )
    statuses_changed = False
    for instance_id in sorted(snapshots):
        snapshot = snapshots[instance_id]
        statuses = snapshot["statuses"]
        assert isinstance(statuses, dict)
        if tracker.record_statuses(channels, instance_id, statuses):
            statuses_changed = True
        if tracker.verbose:
            tracker.print_new_log_lines(instance_id, snapshot)
    if statuses_changed and channels.status_changed is not None:
        channels.status_changed.set()


def _await_next_worker_poll(
    args: argparse.Namespace,
    channels: _MonitorChannels,
    sleep: Callable[[float], None],
) -> None:
    """Wait out one poll interval, waking early when the dispatcher aborts."""

    poll_seconds = args.completion_poll_seconds
    if channels.tracks_units:
        poll_seconds = min(poll_seconds, args.readiness_poll_seconds)
    if channels.abort is not None:
        channels.abort.wait(poll_seconds)
    else:
        sleep(poll_seconds)


def _wait_for_workers_to_stop(
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    unit_count: int,
    status_run_id: str,
    *,
    describe: Callable[[str, str, str | None], list[ClusterInstance]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    console: Console | None = None,
    worker_stages: dict[str, str] | None = None,
    stage_lock: Lock | None = None,
    all_dispatched: Event | None = None,
    abort: Event | None = None,
    worker_args: dict[str, argparse.Namespace] | None = None,
    unit_statuses: dict[tuple[str, str], str] | None = None,
    unit_status_lock: Lock | None = None,
    status_changed: Event | None = None,
) -> None:
    """Monitor mixed worker stages until every dispatched worker has stopped."""

    # Every optional keyword is one shared-state channel to the dispatcher; they
    # are bundled once here so the poll helpers below take a single object.
    channels = _MonitorChannels(
        worker_stages=worker_stages,
        stage_lock=stage_lock,
        all_dispatched=all_dispatched,
        abort=abort,
        worker_args=worker_args,
        unit_statuses=unit_statuses,
        unit_status_lock=unit_status_lock,
        status_changed=status_changed,
    )
    describe_instances = describe or _describe_cluster_instances
    output = console or Console(stderr=True, highlight=False)
    expected_ids = set(instance_ids)
    started_at = time.monotonic()
    verbose = getattr(args, "verbose", False)
    previous_state_signature: tuple[tuple[str, int], ...] | None = None
    aborted = False
    tracker = _WorkerOutputTracker(
        output=output,
        verbose=verbose,
        worker_tags={
            instance_id: (
                _worker_tag(instance_id),
                WORKER_LOG_STYLES[(index - 1) % len(WORKER_LOG_STYLES)],
            )
            for index, instance_id in enumerate(sorted(expected_ids), start=1)
        },
    )
    live_status = (
        output.status(_stage_status("workers"), spinner="dots")
        if output.is_terminal
        else None
    )
    if live_status is None:
        output.print(Text.assemble(("…", "cyan"), " ", ("workers", "bold")))
    else:
        live_status.start()

    try:
        while True:
            if channels.aborted:
                aborted = True
                break
            selected = _monitored_workers(describe_instances, args, expected_ids)
            state_counts = Counter(instance.state for instance in selected.values())
            stopped = state_counts["stopped"]
            lifecycle_counts: Counter[str] | None = None
            stage_snapshot: dict[str, str] = {}
            if channels.tracks_stages:
                stage_snapshot = _reconcile_worker_stages(channels, selected)
                lifecycle_counts = Counter(
                    value.split(":", 1)[0] for value in stage_snapshot.values()
                )
                detail = _lifecycle_progress_detail(
                    channels, lifecycle_counts, unit_count, started_at
                )
            else:
                detail = _provider_progress_detail(state_counts, unit_count, started_at)
            if live_status is not None:
                live_status.update(_stage_status("workers", detail))
            else:
                state_signature = tuple(
                    sorted((lifecycle_counts or state_counts).items())
                )
                if state_signature != previous_state_signature:
                    output.print(Text(f"workers  {detail}", style="dim"))
                    previous_state_signature = state_signature
            running_ids = _materializing_worker_ids(channels, selected, stage_snapshot)
            if running_ids and (verbose or channels.tracks_units):
                _publish_worker_output(
                    args, channels, tracker, running_ids, status_run_id
                )
            if channels.dispatch_complete and stopped == len(expected_ids):
                break
            _await_next_worker_poll(args, channels, sleep)
    finally:
        if live_status is not None:
            live_status.stop()

    if not aborted:
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
    planned_output_shards = int(row["planned_output_shard_count"])
    if len(npys) != planned_output_shards:
        problems.append(
            f"found {len(npys):,} output shards; expected {planned_output_shards:,}"
        )
    invalid_npys = [
        obj for obj in npys if obj.size_bytes <= 0 or obj.size_bytes % UINT32_BYTES
    ]
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
    unexpected = [obj for obj in objects if not obj.key.endswith((".npy", ".csv.gz"))]
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
    live_status = (
        output.status(_stage_status("verify materialized outputs"), spinner="dots")
        if output.is_terminal
        else None
    )
    if live_status is None:
        output.print(
            Text.assemble(("…", "cyan"), " ", ("verify materialized outputs", "bold"))
        )
    else:
        live_status.start()

    checks: list[MaterializedUnitCheck] = []
    request_errors: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=min(args.parallelism, len(rows))) as pool:
            futures = {
                pool.submit(_check_materialized_unit, client, row): row for row in rows
            }
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

    batches = _provision_batches(worker_count, _provision_batch_size(args))
    create_commands = [
        (
            f"create worker batch {index}/{len(batches)}",
            _create_command(args, count, detach=True),
        )
        for index, count in enumerate(batches, start=1)
    ]
    return [
        *create_commands,
        ("upload storage setup", _storage_transfer_command(args, ())),
        ("prepare local NVMe", _storage_setup_command(args, (), rows)),
        ("install Dolma and s5cmd", _runtime_setup_command(args, ())),
        ("upload resharding runtime", _runtime_transfer_command(args, ())),
        (
            "install and validate resharding runtime",
            _runtime_validation_command(args, ()),
        ),
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
    instance_type: str | None = None,
    storage_layout: str | None = None,
) -> None:
    category_count = len({row["leaf_id"] for row in rows})
    planned_tokens = sum(int(row["planned_uint32_values"]) for row in rows)
    planned_output_shards = sum(int(row["planned_output_shard_count"]) for row in rows)
    largest_unit = max(int(row["estimated_peak_local_bytes"]) for row in rows)
    display_label = label.removeprefix("category-") if label.startswith("category-") else label

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", no_wrap=True)
    summary.add_column()
    summary.add_row("Selection", display_label)
    summary.add_row(
        "Work",
        " · ".join(
            (
                _count_label(category_count, "category", "categories"),
                _count_label(len(rows), "unit"),
                _count_label(worker_count, "worker"),
            )
        ),
    )
    summary.add_row(
        "Output",
        f"{_human_token_count(planned_tokens)} tokens · {planned_output_shards:,} shards",
    )
    summary.add_row("Largest working set", _human_byte_count(largest_unit))
    if instance_type:
        summary.add_row("Worker", instance_type)
    if storage_layout:
        summary.add_row("Local storage", storage_layout)
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
            f"{row['unit_id']}\t{_human_token_count(int(row['planned_uint32_values']))} tokens\t"
            f"{int(row['planned_output_shard_count']):,} shards\t"
            f"{_human_byte_count(int(row['estimated_peak_local_bytes']))}\t"
            f"{row['destination_prefix']}"
        )


def _ready_worker_ids(
    args: argparse.Namespace, instance_ids: Sequence[str]
) -> set[str]:
    """Return workers whose provider health checks have passed."""

    if not instance_ids:
        return set()
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("ec2", region_name=args.region)
    ready: set[str] = set()
    ordered = sorted(set(instance_ids))
    for batch in _lifecycle_batches(ordered, DESCRIBE_INSTANCE_STATUS_ID_LIMIT):
        response = client.describe_instance_status(
            InstanceIds=batch,
            IncludeAllInstances=True,
        )
        for status in response.get("InstanceStatuses", []):
            if (
                status.get("InstanceState", {}).get("Name") == "running"
                and status.get("InstanceStatus", {}).get("Status") == "ok"
                and status.get("SystemStatus", {}).get("Status") == "ok"
            ):
                instance_id = status.get("InstanceId")
                if isinstance(instance_id, str):
                    ready.add(instance_id)
    return ready


def _set_worker_stage(
    stages: dict[str, str], lock: Lock, instance_id: str, stage: str
) -> None:
    with lock:
        stages[instance_id] = stage


def _bootstrap_and_dispatch_worker(
    args: argparse.Namespace,
    instance_id: str,
    assignment: WorkerAssignment,
    stages: dict[str, str],
    stage_lock: Lock,
    console: Console | None = None,
) -> None:
    """Prepare one ready worker and dispatch only its assigned execution units."""

    tag = _worker_tag(instance_id)

    def run(stage: str, command: Sequence[str]) -> None:
        _set_worker_stage(stages, stage_lock, instance_id, f"bootstrapping:{stage}")
        _run_lifecycle_command(
            f"{tag} · {stage}",
            command,
            console=console,
            verbose=args.verbose,
            live=False,
        )

    group_args = assignment.group.args
    try:
        run("upload storage setup", _storage_transfer_command(group_args, [instance_id]))
        run(
            "prepare local NVMe",
            _storage_setup_command(group_args, [instance_id], assignment.group.rows),
        )
        run("install Dolma and s5cmd", _runtime_setup_command(group_args, [instance_id]))
        run(
            "upload resharding runtime",
            _runtime_transfer_command(group_args, [instance_id]),
        )
        run(
            "validate resharding runtime",
            _runtime_validation_command(group_args, [instance_id]),
        )
        run(
            f"dispatch {len(assignment.rows):,} unit(s)",
            _map_command(
                group_args,
                assignment.script_dir,
                [instance_id],
                spindown=False,
            ),
        )
        _set_worker_stage(stages, stage_lock, instance_id, "materializing")
    except BaseException:
        _set_worker_stage(stages, stage_lock, instance_id, "failed")
        raise


def _dispatch_assignment_to_worker(
    instance_id: str,
    assignment: WorkerAssignment,
    stages: dict[str, str],
    stage_lock: Lock,
    console: Console,
) -> None:
    """Dispatch the next unit without rebuilding or stopping the warm worker."""

    unit_id = assignment.rows[0]["unit_id"]
    _set_worker_stage(stages, stage_lock, instance_id, f"queued:{unit_id}")
    try:
        _run_lifecycle_command(
            f"{_worker_tag(instance_id)} · dispatch {unit_id}",
            _map_command(
                assignment.group.args,
                assignment.script_dir,
                [instance_id],
                spindown=False,
            ),
            console=console,
            verbose=assignment.group.args.verbose,
            live=False,
        )
        _set_worker_stage(stages, stage_lock, instance_id, "materializing")
    except BaseException:
        _set_worker_stage(stages, stage_lock, instance_id, "failed")
        raise


def _stop_worker_after_work(
    instance_id: str,
    group_args: argparse.Namespace,
    stages: dict[str, str],
    stage_lock: Lock,
    console: Console,
) -> None:
    """Stop one warm worker after its compatible unit queue is empty."""

    _set_worker_stage(stages, stage_lock, instance_id, "stopping")
    try:
        _run_lifecycle_command(
            f"{_worker_tag(instance_id)} · stop",
            _pause_command(group_args, [instance_id]),
            console=console,
            verbose=False,
            live=False,
        )
    except BaseException:
        _set_worker_stage(stages, stage_lock, instance_id, "failed")
        raise


@dataclass(frozen=True)
class _DispatchChannels:
    """Shared state the dispatcher owns and the progress monitor reads or fills."""

    worker_stages: dict[str, str]
    stage_lock: Lock = field(default_factory=Lock)
    unit_statuses: dict[tuple[str, str], str] = field(default_factory=dict)
    unit_status_lock: Lock = field(default_factory=Lock)
    status_changed: Event = field(default_factory=Event)
    all_dispatched: Event = field(default_factory=Event)
    abort: Event = field(default_factory=Event)

    def set_stage(self, instance_id: str, stage: str) -> None:
        _set_worker_stage(self.worker_stages, self.stage_lock, instance_id, stage)

    def stage_snapshot(self) -> dict[str, str]:
        with self.stage_lock:
            return dict(self.worker_stages)

    def status_snapshot(self) -> dict[tuple[str, str], str]:
        with self.unit_status_lock:
            return dict(self.unit_statuses)


def _harvest_finished_futures(futures: dict[Future[None], str]) -> None:
    """Re-raise the first failure and forget every future that already finished."""

    for future in [future for future in futures if future.done()]:
        future.result()
        del futures[future]


@dataclass
class _WarmWorkerPool:
    """The warm-worker state machine behind one materialization run.

    Each healthy worker starts one execution unit immediately, receives another
    compatible unit from its own group's queue whenever the unit it was running
    succeeds, and stays warm until that queue is empty, then stops.
    """

    args: argparse.Namespace
    pool: ThreadPoolExecutor
    console: Console
    channels: _DispatchChannels
    assignment_queues: list[deque[WorkerAssignment]]
    worker_group_index: dict[str, int]
    worker_args: dict[str, argparse.Namespace]
    pending: set[str]
    futures: dict[Future[None], str] = field(default_factory=dict)
    current_assignments: dict[str, WorkerAssignment] = field(default_factory=dict)
    completed_unit_ids: set[str] = field(default_factory=set)

    @property
    def busy(self) -> bool:
        """Report whether any worker still owes a bootstrap, a unit, or a stop."""

        return bool(
            self.pending
            or self.futures
            or self.current_assignments
            or any(self.assignment_queues)
        )

    def advance_completed_units(self) -> None:
        """Refill or stop every warm worker whose unit is no longer running."""

        inflight_ids = set(self.futures.values())
        status_snapshot = self.channels.status_snapshot()
        self.channels.status_changed.clear()
        stage_snapshot = self.channels.stage_snapshot()
        for instance_id, assignment in list(self.current_assignments.items()):
            if instance_id in inflight_ids:
                continue
            unit_id = assignment.rows[0]["unit_id"]
            status = status_snapshot.get((instance_id, unit_id))
            if status == "succeeded":
                self.completed_unit_ids.add(unit_id)
                del self.current_assignments[instance_id]
                self._dispatch_next_unit(instance_id)
            elif status is not None and status.startswith("failed"):
                self.channels.set_stage(instance_id, "failed")
                raise PreparationError(
                    f"Execution unit {unit_id} failed on {instance_id}: {status}"
                )
            elif stage_snapshot.get(instance_id) in {"stopping", "stopped"}:
                raise PreparationError(
                    f"Worker {instance_id} stopped before execution unit "
                    f"{unit_id} reported success"
                )

    def start_ready_workers(self) -> None:
        """Bootstrap and dispatch every worker the provider now reports healthy."""

        newly_ready = (
            _ready_worker_ids(self.args, sorted(self.pending)) if self.pending else set()
        )
        for instance_id in sorted(newly_ready):
            queue = self._queue_for(instance_id)
            if not queue:
                self.pending.remove(instance_id)
                self._stop_worker(instance_id)
                continue
            assignment = queue.popleft()
            self.pending.remove(instance_id)
            self.current_assignments[instance_id] = assignment
            self.channels.set_stage(instance_id, "queued")
            self._submit(
                instance_id,
                _bootstrap_and_dispatch_worker,
                self.args,
                instance_id,
                assignment,
                self.channels.worker_stages,
                self.channels.stage_lock,
                self.console,
            )

    def _dispatch_next_unit(self, instance_id: str) -> None:
        """Keep one worker warm with its next unit, or stop it when none remain."""

        queue = self._queue_for(instance_id)
        if not queue:
            self._stop_worker(instance_id)
            return
        assignment = queue.popleft()
        self.current_assignments[instance_id] = assignment
        self._submit(
            instance_id,
            _dispatch_assignment_to_worker,
            instance_id,
            assignment,
            self.channels.worker_stages,
            self.channels.stage_lock,
            self.console,
        )

    def _stop_worker(self, instance_id: str) -> None:
        self._submit(
            instance_id,
            _stop_worker_after_work,
            instance_id,
            self.worker_args[instance_id],
            self.channels.worker_stages,
            self.channels.stage_lock,
            self.console,
        )

    def _queue_for(self, instance_id: str) -> deque[WorkerAssignment]:
        """Return the queue of units this worker's instance type can run."""

        return self.assignment_queues[self.worker_group_index[instance_id]]

    def _submit(
        self, instance_id: str, function: Callable[..., None], *call_args: Any
    ) -> None:
        self.futures[self.pool.submit(function, *call_args)] = instance_id


def _await_dispatch_progress(
    args: argparse.Namespace,
    workers: _WarmWorkerPool,
    resume_futures: dict[Future[None], str],
) -> None:
    """Block until the next event that can change the dispatch loop's decisions."""

    if workers.pending:
        # Readiness is only visible through the provider, so this wait must also
        # time out on the readiness poll interval.
        active_futures = [*workers.futures, *resume_futures]
        if active_futures:
            wait(
                active_futures,
                timeout=args.readiness_poll_seconds,
                return_when=FIRST_COMPLETED,
            )
        else:
            time.sleep(args.readiness_poll_seconds)
    elif workers.futures:
        wait(workers.futures, return_when=FIRST_COMPLETED)
    elif resume_futures:
        wait(resume_futures, return_when=FIRST_COMPLETED)
    elif workers.current_assignments:
        workers.channels.status_changed.wait(args.readiness_poll_seconds)


def _execute_materialization_groups(
    args: argparse.Namespace,
    groups: Sequence[MaterializationGroup],
    selected: Sequence[dict[str, str]],
    status_run_id: str,
) -> None:
    """Dispatch each worker as soon as it becomes ready."""

    prepared: list[tuple[MaterializationGroup, list[str]]] = []
    all_worker_ids: list[str] = []
    deferred_resumes: list[tuple[argparse.Namespace, list[str]]] = []
    assignment_queues = [deque(_stage_worker_assignments(group)) for group in groups]
    try:
        for group_index, group in enumerate(groups):
            group_resume_ids: list[str] = []
            worker_ids = _prepare_workers(
                group.args,
                group.worker_count,
                owned_instance_ids=all_worker_ids,
                wait_for_ready=False,
                delay_after_last_batch=group_index < len(groups) - 1,
                deferred_resume_ids=group_resume_ids,
            )
            prepared.append((group, worker_ids))
            all_worker_ids.extend(worker_ids)
            if group_resume_ids:
                deferred_resumes.append((group.args, group_resume_ids))

        channels = _DispatchChannels(
            worker_stages={instance_id: "waiting" for instance_id in all_worker_ids}
        )
        lifecycle_console = Console(stderr=True, highlight=False)
        worker_group_index = {
            instance_id: group_index
            for group_index, (_, worker_ids) in enumerate(prepared)
            for instance_id in worker_ids
        }
        worker_args = {
            instance_id: group.args
            for group, worker_ids in prepared
            for instance_id in worker_ids
        }
        resume_pool = ThreadPoolExecutor(max_workers=1)
        resume_futures = (
            {
                resume_pool.submit(
                    _resume_worker_groups_in_batches,
                    deferred_resumes,
                    console=lifecycle_console,
                ): "resume workers"
            }
            if deferred_resumes
            else {}
        )
        monitor_pool = ThreadPoolExecutor(max_workers=1)
        monitor = monitor_pool.submit(
            _wait_for_workers_to_stop,
            args,
            all_worker_ids,
            len(selected),
            status_run_id,
            worker_stages=channels.worker_stages,
            stage_lock=channels.stage_lock,
            all_dispatched=channels.all_dispatched,
            abort=channels.abort,
            console=lifecycle_console,
            worker_args=worker_args,
            unit_statuses=channels.unit_statuses,
            unit_status_lock=channels.unit_status_lock,
            status_changed=channels.status_changed,
        )
        try:
            bootstrap_workers = min(args.bootstrap_parallelism, len(all_worker_ids))
            with ThreadPoolExecutor(max_workers=bootstrap_workers) as pool:
                workers = _WarmWorkerPool(
                    args=args,
                    pool=pool,
                    console=lifecycle_console,
                    channels=channels,
                    assignment_queues=assignment_queues,
                    worker_group_index=worker_group_index,
                    worker_args=worker_args,
                    pending=set(all_worker_ids),
                )
                while workers.busy or resume_futures:
                    _harvest_finished_futures(resume_futures)
                    _harvest_finished_futures(workers.futures)
                    workers.advance_completed_units()
                    workers.start_ready_workers()
                    _await_dispatch_progress(args, workers, resume_futures)

            channels.all_dispatched.set()
            monitor.result()
        except BaseException:
            channels.abort.set()
            raise
        finally:
            monitor_pool.shutdown(wait=True)
            resume_pool.shutdown(wait=True)

        if any(assignment_queues):
            raise PreparationError("Not every execution-unit assignment was dispatched")
        if len(workers.completed_unit_ids) != len(selected):
            raise PreparationError(
                f"Only {len(workers.completed_unit_ids):,}/{len(selected):,} execution units "
                "reported success"
            )
        _verify_materialized_units(args, selected)
    except BaseException:
        for group, worker_ids in prepared:
            _pause_workers_after_failure(group.args, worker_ids)
        raise


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()
        args.region = normalize_region(args.region)
        if args.preflight and not args.execute:
            raise PreparationError("--preflight requires --execute")
        build = args.build.resolve()
        rows = _load_execution_units(build)
        if args.list_categories is not None:
            _print_categories(rows, args.list_categories)
            return

        label, selected = _select_units(args, rows)
        worker_groups = _planned_worker_groups(args, selected)
        status_run_id = f"{int(time.time())}-{uuid.uuid4().hex[:12]}" if args.execute else None
        if args.execute:
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
            if args.preflight:
                _run_selected_preflight(args, build, selected)
            preflight_created_at = _require_preflight(build, selected)
            print(f"preflight=passed created_at={preflight_created_at}")

        worker_counts = _worker_counts_for_groups(worker_groups, args.parallelism)
        materialization_groups: list[MaterializationGroup] = []
        for group, worker_count in zip(worker_groups, worker_counts):
            group_args = copy.copy(args)
            group_args.instance_type = group.instance_type
            group_args.storage_layout = group.storage_layout
            group_rows = list(group.rows)
            group_label = label if len(worker_groups) == 1 else f"{label}-{group.instance_type}"
            script_dir = _stage_selection(build, group_label, group_rows, status_run_id)
            lifecycle_commands = _dry_run_lifecycle_commands(
                group_args,
                group_rows,
                script_dir,
                worker_count,
            )
            _print_dispatch(
                group_label,
                group_rows,
                script_dir,
                lifecycle_commands,
                worker_count,
                args.execute,
                cluster=args.cluster,
                project=args.project,
                region=args.region,
                instance_type=group.instance_type,
                storage_layout=group.storage_layout,
            )
            materialization_groups.append(
                MaterializationGroup(
                    args=group_args,
                    rows=tuple(group_rows),
                    script_dir=script_dir,
                    worker_count=worker_count,
                )
            )

        if args.execute:
            assert status_run_id is not None
            _execute_materialization_groups(
                args,
                materialization_groups,
                selected,
                status_run_id,
            )
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
