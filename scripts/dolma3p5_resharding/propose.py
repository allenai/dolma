"""Create reviewable Dolma 3.5 resharding configs and exact manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow import DEFAULT_BUILD_PATH, PreparationError, propose_configs


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
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--local-temp-root", required=True)
    parser.add_argument(
        "--max-unit-working-bytes",
        type=int,
        required=True,
        help=(
            "maximum estimated input-plus-output bytes for one worker unit; "
            "choose a value below usable worker disk capacity"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    try:
        propose_configs(parser.parse_args())
    except PreparationError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
