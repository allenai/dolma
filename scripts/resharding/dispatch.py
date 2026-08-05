"""Build and validate poormanray dispatches for resharding jobs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class PoormanrayDispatchError(RuntimeError):
    """A poormanray dispatch cannot safely run as configured."""


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
POORMANRAY_RUNNER = [
    "uv",
    "run",
    "--isolated",
    "--no-project",
    "--with",
    "poormanray",
    "--",
    "pmr",
]


def build_poormanray_map_command(
    *,
    cluster: str,
    script_dir: str | Path,
    region: str | None = None,
    project: str | None = None,
    spindown: bool,
) -> list[str]:
    """Build a poormanray command that maps executable scripts over a cluster."""

    if not cluster.strip():
        raise ValueError("cluster must not be empty")
    command = [*POORMANRAY_RUNNER, "map", "--name", cluster]
    if project:
        command.extend(("--project", project))
    if region:
        command.extend(("--region", region))
    command.extend(("--script", str(script_dir)))
    if spindown:
        command.append("--spindown")
    return command


def require_spindown_coverage(
    *,
    cluster: str,
    region: str,
    script_count: int,
    project: str | None = None,
    cloud: str = "aws",
    gcp_project: str | None = None,
) -> int:
    """Require every active cluster instance to receive at least one script."""

    if script_count <= 0:
        raise ValueError("script_count must be positive")
    if cloud not in {"aws", "gcp"}:
        raise ValueError(f"Unsupported poormanray cloud: {cloud}")

    command = [*POORMANRAY_RUNNER, "list", "--name", cluster, "--region", region]
    if project:
        command.extend(("--project", project))
    if cloud != "aws":
        command.extend(("--cloud", cloud))
    if gcp_project:
        command.extend(("--gcp-project", gcp_project))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise PoormanrayDispatchError(f"Could not inspect poormanray cluster {cluster!r}: {detail}")

    active_instances = 0
    current_instance = False
    for raw_line in result.stdout.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line).strip()
        if line.startswith("Id: ") or line.startswith("Id/Name: "):
            current_instance = True
        elif current_instance and line.startswith("State:"):
            state = line.removeprefix("State:").strip().casefold()
            if re.search(r"\b(?:pending|running)\b", state):
                active_instances += 1
            current_instance = False

    if not active_instances:
        raise PoormanrayDispatchError(
            f"No active poormanray instances found for cluster {cluster!r} in {region}"
        )
    if active_instances > script_count:
        raise PoormanrayDispatchError(
            f"Cluster {cluster!r} has {active_instances:,} active instances but this dispatch has "
            f"only {script_count:,} scripts. poormanray does not apply --spindown to instances "
            "that receive no script. Use a cluster with no more instances than scripts."
        )
    return active_instances
