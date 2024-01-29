"""
Script to identify which documents files have missing attributes
files.

Author: Luca Soldaini (@soldni)
"""

import sys
from typing import Generator

try:
    import click
    from dolma.core.paths import _get_fs, _pathify, glob_path, join_path, sub_prefix
except ImportError as e:
    raise ImportError("Missing dependency; plese run `pip install dolma click`.") from e


def find_missing(prefix: str, attribute_name: str) -> Generator[str, None, None]:
    """
    Find all files in the given prefix that are missing the given attribute.
    """
    fs = _get_fs(prefix)
    protocol, _ = _pathify(prefix)
    document_prefix = join_path(protocol, prefix, "documents")

    count_all_ = count_miss = 0

    for root, directories, filenames in fs.walk(document_prefix):
        if directories:
            # ignore directories
            continue

        subpath = sub_prefix(join_path(protocol, root), document_prefix)

        for fn in filenames:
            attribute_fn = join_path(protocol, prefix, "attributes", attribute_name, subpath, fn)
            documents_fn = join_path(protocol, root, fn)
            count_all_ += 1
            if not fs.exists(attribute_fn):
                count_miss += 1
                yield documents_fn

    print(f"Total documents: {count_all_:,}", file=sys.stderr)
    print(f"Missing attrs:   {count_miss:,}", file=sys.stderr)


@click.command()
@click.argument("attribute-path", type=str, required=True)
@click.option("--separator", type=str, default="\n", help="Separator to use between paths")
def main(attribute_path: str, separator: str) -> None:
    """
    Find all files in the given prefix that are missing the given attribute.
    """

    if "/attributes/" not in attribute_path:
        raise ValueError("Attribute path must contain 'attributes'")

    prefix, attribute = attribute_path.split("/attributes/", 1)

    if not attribute.strip():
        raise ValueError("Attribute name must not be empty")

    for missing in find_missing(prefix, attribute):
        print(missing, end=separator, flush=True, file=sys.stdout)


if __name__ == "__main__":
    main()
