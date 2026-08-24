"""Run read-only source-drift and destination checks before materialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow import DEFAULT_BUILD_PATH, DEFAULT_REGION, PreparationError, preflight_build


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
    parser.add_argument("--profile", help="optional AWS profile")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="AWS region; override when the source or destination store is in another region",
    )
    parser.add_argument("--max-workers", type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="check every execution unit; this is also the default selection",
    )
    selection.add_argument(
        "--category",
        help="check one exact mix name, leaf ID, or MIX_NAME::CATEGORY_NAME selector",
    )
    selection.add_argument(
        "--unit",
        help="check one exact execution-unit ID",
    )
    return parser


def main() -> None:
    parser = build_parser()
    try:
        preflight_build(parser.parse_args())
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
