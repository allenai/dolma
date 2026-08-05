"""Build the complete Dolma 3.5 sampling and materialization plan."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        PreparationError,
        collect_inventory,
        plan_build,
        propose_configs,
    )
except ImportError:
    from workflow import (
        DEFAULT_BUILD_PATH,
        DEFAULT_REGION,
        PreparationError,
        collect_inventory,
        plan_build,
        propose_configs,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs/dolma3p5-resharding/14t"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mix",
        type=Path,
        default=CONFIG_ROOT / "inputs/dolma3p5-14t-optimal-mix.yaml",
        help="authoritative mix YAML",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CONFIG_ROOT / "inputs/dolma3p5-reference-all-npy-s3-paths.csv",
        help="reference superset of S3 NPY paths",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=CONFIG_ROOT / "settings.yaml",
        help="preparation settings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BUILD_PATH,
        help="local preparation build directory",
    )
    parser.add_argument("--profile", help="optional AWS profile")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="AWS region; override when the source store is in another region",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="override concurrent read-only exact-object requests",
    )
    parser.add_argument(
        "--destination-root",
        required=True,
        help="new object-store prefix below which every execution unit gets a unique destination",
    )
    parser.add_argument(
        "--local-temp-root",
        type=Path,
        required=True,
        help="absolute worker-local directory used while materializing one execution unit",
    )
    parser.add_argument(
        "--max-unit-working-bytes",
        type=int,
        required=True,
        help="maximum estimated local working set for one execution unit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if shutil.which("s5cmd") is None:
            raise PreparationError("s5cmd is required for planning and was not found on PATH")
        plan_build(args)
        collect_inventory(
            argparse.Namespace(
                build=args.output,
                profile=args.profile,
                region=args.region,
                max_workers=args.max_workers,
            )
        )
        propose_configs(
            argparse.Namespace(
                build=args.output,
                destination_root=args.destination_root,
                local_temp_root=args.local_temp_root,
                max_unit_working_bytes=args.max_unit_working_bytes,
            )
        )
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
