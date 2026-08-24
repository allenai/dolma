"""Build poormanray commands for a complete resharding worker lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def _base_command(
    action: str,
    *,
    cluster: str,
    region: str | None,
    project: str | None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    if not cluster.strip():
        raise ValueError("cluster must not be empty")
    command = [*(runner or ("pmr",)), action, "--name", cluster]
    if project:
        command.extend(("--project", project))
    if region:
        command.extend(("--region", region))
    return command


def _add_instance_options(
    command: list[str],
    *,
    instance_ids: Sequence[str] = (),
    parallelism: int | None = None,
    ssh_key_path: str | Path | None = None,
) -> list[str]:
    for instance_id in instance_ids:
        if not instance_id.strip():
            raise ValueError("instance IDs must not be empty")
        command.extend(("--instance-id", instance_id))
    if parallelism is not None:
        if parallelism <= 0:
            raise ValueError("parallelism must be positive")
        command.extend(("--parallelism", str(parallelism)))
    if ssh_key_path is not None:
        command.extend(("--ssh-key-path", str(ssh_key_path)))
    return command


def build_poormanray_create_command(
    *,
    cluster: str,
    project: str,
    region: str,
    number: int,
    instance_type: str,
    storage_type: str,
    storage_size_gib: int,
    parallelism: int | None = None,
    detach: bool = False,
    ssh_key_path: str | Path | None = None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    """Build the command that creates missing workers for a cluster."""

    if number <= 0:
        raise ValueError("number must be positive")
    if storage_size_gib <= 0:
        raise ValueError("storage_size_gib must be positive")
    command = _base_command(
        "create", cluster=cluster, project=project, region=region, runner=runner
    )
    command.extend(
        (
            "--number",
            str(number),
            "--instance-type",
            instance_type,
            "--storage-type",
            storage_type,
            "--storage-size",
            str(storage_size_gib),
        )
    )
    if detach:
        command.append("--detach")
    return _add_instance_options(
        command,
        parallelism=parallelism,
        ssh_key_path=ssh_key_path,
    )


def build_poormanray_instance_command(
    action: str,
    *,
    cluster: str,
    project: str,
    region: str,
    instance_ids: Sequence[str],
    parallelism: int | None = None,
    detach: bool = False,
    ssh_key_path: str | Path | None = None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    """Build a wait, resume, or pause command scoped to explicit workers."""

    if action not in {"wait", "resume", "pause"}:
        raise ValueError(f"unsupported instance action: {action}")
    command = _base_command(
        action, cluster=cluster, project=project, region=region, runner=runner
    )
    if detach and action != "wait":
        command.append("--detach")
    return _add_instance_options(
        command,
        instance_ids=instance_ids,
        parallelism=parallelism if action != "wait" else None,
        ssh_key_path=ssh_key_path if action == "wait" else None,
    )


def build_poormanray_transfer_command(
    *,
    cluster: str,
    project: str,
    region: str,
    transfers: Sequence[tuple[str | Path, str]],
    instance_ids: Sequence[str] = (),
    parallelism: int | None = None,
    ssh_key_path: str | Path | None = None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    """Build a transfer command for worker bootstrap files."""

    if not transfers:
        raise ValueError("at least one transfer is required")
    command = _base_command(
        "transfer", cluster=cluster, project=project, region=region, runner=runner
    )
    for source, destination in transfers:
        command.extend(("--source", f"{source}:{destination}"))
    return _add_instance_options(
        command,
        instance_ids=instance_ids,
        parallelism=parallelism,
        ssh_key_path=ssh_key_path,
    )


def build_poormanray_run_command(
    *,
    cluster: str,
    project: str,
    region: str,
    remote_command: str,
    instance_ids: Sequence[str] = (),
    parallelism: int | None = None,
    ssh_key_path: str | Path | None = None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    """Build a synchronous command to run on selected workers."""

    if not remote_command.strip():
        raise ValueError("remote_command must not be empty")
    command = _base_command(
        "run", cluster=cluster, project=project, region=region, runner=runner
    )
    command.extend(("--command", remote_command))
    return _add_instance_options(
        command,
        instance_ids=instance_ids,
        parallelism=parallelism,
        ssh_key_path=ssh_key_path,
    )


def build_poormanray_setup_dolma_command(
    *,
    cluster: str,
    project: str,
    region: str,
    instance_ids: Sequence[str] = (),
    parallelism: int | None = None,
    ssh_key_path: str | Path | None = None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    """Build the command that installs Dolma and s5cmd on selected workers."""

    command = _base_command(
        "setup-dolma-python",
        cluster=cluster,
        project=project,
        region=region,
        runner=runner,
    )
    return _add_instance_options(
        command,
        instance_ids=instance_ids,
        parallelism=parallelism,
        ssh_key_path=ssh_key_path,
    )


def build_poormanray_map_command(
    *,
    cluster: str,
    script_dir: str | Path,
    region: str | None = None,
    project: str | None = None,
    spindown: bool,
    instance_ids: Sequence[str] = (),
    parallelism: int | None = None,
    ssh_key_path: str | Path | None = None,
    runner: Sequence[str] | None = None,
) -> list[str]:
    """Build a poormanray command that maps scripts over selected workers."""

    command = _base_command(
        "map", cluster=cluster, project=project, region=region, runner=runner
    )
    command.extend(("--script", str(script_dir)))
    if spindown:
        command.append("--spindown")
    return _add_instance_options(
        command,
        instance_ids=instance_ids,
        parallelism=parallelism,
        ssh_key_path=ssh_key_path,
    )
