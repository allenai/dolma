"""Resolve the Dolma 3.5 mix, inventory sources, and build the sampling plan."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .workflow import (
        DEFAULT_BUILD_PATH,
        PreparationError,
        collect_inventory,
        plan_build,
    )
except ImportError:
    from workflow import (
        DEFAULT_BUILD_PATH,
        PreparationError,
        collect_inventory,
        plan_build,
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
    parser.add_argument("--region", help="optional AWS region")
    parser.add_argument(
        "--max-workers",
        type=int,
        help="override concurrent read-only exact-object requests",
    )
    return parser


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if shutil.which("s5cmd") is None:
            raise PreparationError(
                "s5cmd is required for planning and was not found on PATH"
            )
        plan_build(args)
        collect_inventory(
            argparse.Namespace(
                build=args.output,
                profile=args.profile,
                region=args.region,
                max_workers=args.max_workers,
            )
        )
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
