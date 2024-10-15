import argparse

from .common import create_index

from markdownify import markdownify as md
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tantivy import SnippetGenerator


def make_search_parser():
    parser = argparse.ArgumentParser("Interactive search tool on a tantivy index")
    parser.add_argument(
        "-i",
        "--index-path",
        type=str,
        help="The path to the index. If not provided, an in-memory index will be used."
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="The query to search for."
    )
    parser.add_argument(
        "-n",
        "--num-hits",
        type=int,
        default=10,
        help="The number of hits to return."
    )
    parser.add_argument(
        "-s",
        "--show-snippets",
        action="store_true",
        help="Show snippets in the search results."
    )
    return parser


def query_iterator(query: str | None):
    if query is None:
        while True:
            try:
                query = input("Enter a query: ")
                yield query
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    elif query == "-":
        for line in sys.stdin:
            yield line.strip()
    else:
        yield query


def search_data(args: argparse.Namespace):
    index = create_index(args.index_path, reuse=True)
    searcher = index.searcher()

    console = Console()

    for query in query_iterator(args.query):
        parsed_query = index.parse_query(query)
        hits = searcher.search(parsed_query, limit=args.num_hits).hits
        table = Table(title="Search Results", show_header=True, header_style="bold", show_lines=True)
        table.add_column("Score", justify="right", style="green")
        table.add_column("ID", style="magenta")
        table.add_column("Text", style="blue")
        for hit_score, hit_doc_address in hits:
            document = searcher.doc(hit_doc_address)
            hit_id = document["id"][0]



            if args.show_snippets:
                snippet_generator = SnippetGenerator.create(
                searcher, parsed_query, index.schema, "text"
                )
                snippet = snippet_generator.snippet_from_doc(document)
                hit_text = Markdown(md(snippet.to_html()).strip())
            else:
                hit_text = Text(document["text"][0].strip().replace("\n", " ⮑"))

            table.add_row(f"{hit_score:.2f}", hit_id, hit_text)

        console.print(table)



if __name__ == "__main__":
    search_data(make_search_parser().parse_args())
