"""Validate the locally prepared Dolma 3.5 resharding artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow import DEFAULT_BUILD_PATH, PreparationError, validate_build


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
    return parser


def main() -> None:
    parser = build_parser()
    try:
        validate_build(parser.parse_args())
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
