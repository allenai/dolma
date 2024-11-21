import argparse
import sys
import os

from . import index, query

CLI_DESCRIPTION = "Dolma Search CLI"


def main():
    parser = argparse.ArgumentParser(CLI_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index subparser
    index_parser = subparsers.add_parser("index", help=index.INDEX_DESCRIPTION)
    index.make_index_parser(index_parser)

    # Query subparser
    query_parser = subparsers.add_parser("query", help=query.QUERY_DESCRIPTION)
    query.make_search_parser(query_parser)

    args = parser.parse_args()

    if args.command == "index":
        index.index_data(args)
    elif args.command == "query":
        query.search_data_flex(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
