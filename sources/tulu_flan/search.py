import argparse
from contextlib import ExitStack
from pathlib import Path
import re
import json
import os

import smart_open
import tqdm
from dolma_decontamination.search.common import create_index
from dolma_decontamination.search.index import list_paths
from dolma_decontamination.search.query import HitsTuple


def make_search_parser():
    parser = argparse.ArgumentParser("Interactive search tool on a tantivy index")
    parser.add_argument(
        "-i",
        "--index-path",
        type=str,
        required=True,
        help="The path to the index."
    )
    parser.add_argument(
        "-d",
        "--documents",
        type=str,
        required=True,
        nargs="+",
        help="The paths to documents to use as queries."
    )
    parser.add_argument(
        "-n",
        "--num-hits",
        type=int,
        default=10,
        help="The number of hits to return."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="A directory to write the output to."
    )
    return parser


class TextNormalizer:
    def __init__(self):
        self.whitespace_re = re.compile(r"\s+")
        self.non_alnum_re = re.compile(r"[^a-zA-Z0-9\s]+")

    def __call__(self, text: str) -> str:
        text = self.whitespace_re.sub(" ", self.non_alnum_re.sub("", text.strip()))
        return text.replace("AND", "and").replace("OR", "or").replace("NOT", "not").replace("IN", "in")


def search_data(args: argparse.Namespace):
    index = create_index(args.index_path, reuse=True)
    searcher = index.searcher()

    paths = list_paths(args.documents)
    norm = TextNormalizer()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm.tqdm(paths, unit="files", unit_scale=True))
        docs_pbar = stack.enter_context(tqdm.tqdm(unit=" docs", unit_scale=True))
        queries_pbar = stack.enter_context(tqdm.tqdm(unit=" queries", unit_scale=True))

        output_id = 0

        output_path = f"{args.output}/{output_id:06d}.jsonl.zst"
        output_file = stack.enter_context(smart_open.open(output_path, "wt", encoding="utf-8"))

        for path in files_pbar:
            f = stack.enter_context(smart_open.open(path, "rt", encoding="utf-8"))
            for line in f:
                document = json.loads(line)

                for start, end, score in document.get("attributes", {}).get("dedupe_ngrams_8_1", []):
                    text = document["text"][start:end]
                    normalized_text = norm(text)

                    parsed_query = index.parse_query(normalized_text)
                    hits = searcher.search(parsed_query, limit=args.num_hits).hits
                    parsed_hits = HitsTuple.from_hits(hits, searcher)

                    output = {
                        "query": normalized_text,
                        "hits": [h.to_dict() for h in parsed_hits],
                        "document": document,
                        "span_score": score
                    }
                    queries_pbar.update(1)
                    output_file.write(json.dumps(output) + "\n")

                    if queries_pbar.n % 50_000 == 0:
                        output_file.close()
                        output_id += 1
                        output_path = f"{args.output}/{output_id:06d}.jsonl.zst"
                        output_file = stack.enter_context(
                            smart_open.open(output_path, "wt", encoding="utf-8")
                        )

                docs_pbar.update(1)

            files_pbar.update(1)


if __name__ == "__main__":
    search_data(make_search_parser().parse_args())
