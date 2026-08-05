"""Build and validate poormanray dispatches for resharding jobs."""

from __future__ import annotations

from pathlib import Path


class PoormanrayDispatchError(RuntimeError):
    """A poormanray dispatch cannot safely run as configured."""


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
    command = ["pmr", "map", "--name", cluster]
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
    cloud: str = "aws",
    gcp_project: str | None = None,
) -> int:
    """Require every active cluster instance to receive at least one script."""

    if script_count <= 0:
        raise ValueError("script_count must be positive")
    if cloud == "aws":
        try:
            from poormanray.aws_instance import InstanceInfo
        except ImportError as exc:
            raise PoormanrayDispatchError(
                "The poormanray Python package is required to verify cluster size before dispatch"
            ) from exc
        describe_options = {"region": region, "project": cluster}
    elif cloud == "gcp":
        try:
            from poormanray.gcp_instance import InstanceInfo
        except ImportError as exc:
            raise PoormanrayDispatchError(
                "The poormanray GCP package is required to verify cluster size before dispatch"
            ) from exc
        describe_options = {
            "region": region,
            "project": cluster,
            "gcp_project": gcp_project,
        }
    else:
        raise ValueError(f"Unsupported poormanray cloud: {cloud}")

    instances = InstanceInfo.describe_instances(**describe_options)
    if not instances:
        raise PoormanrayDispatchError(
            f"No active poormanray instances found for cluster {cluster!r} in {region}"
        )
    if len(instances) > script_count:
        raise PoormanrayDispatchError(
            f"Cluster {cluster!r} has {len(instances):,} active instances but this dispatch has "
            f"only {script_count:,} scripts. poormanray does not apply --spindown to instances "
            "that receive no script. Use a cluster with no more instances than scripts."
        )
    return len(instances)
