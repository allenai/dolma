"""Build the local Dolma 3.5 14T resharding preparation plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow import DEFAULT_BUILD_PATH, PreparationError, plan_build

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
    return parser


def main() -> None:
    parser = build_parser()
    try:
        plan_build(parser.parse_args())
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
