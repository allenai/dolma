from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig, make_workdirs
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path
from dolma.core.registry import TaggerRegistry
from dolma.core.runtime import create_and_run_tagger


# def parse_args() -> argparse.Namespace:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--sources", nargs="+", required=True)
#     ap.add_argument("--destination", required=True)
#     ap.add_argument("--tokenizer-id", type=str, default="allenai/eleuther-ai-gpt-neox-20b-pii-special")
#     ap.add_argument("--metadata-dir", type=str, default=None)
#     ap.add_argument("--num-tokenizers", type=int, default=1)
#     ap.add_argument("--num-writers", type=int, default=1)
#     ap.add_argument("--max-size", type=int, default=1024 * 1024 * 1024)
#     return ap.parse_args()


@dataclass
class TokenizerConfig:
    documents: List[str] = field(
        default=[],
        help="One or more document paths to process; Can be either local or S3 paths. Globs are supported.",
    )
    destination: Optional[List[str]] = field(
        default=None,
        help=(
            "Destination paths to save the outputs; should match the number of document paths. "
            "If not provided, destination will be derived from the document path."
        ),
    )
    tokenizer_id: Optional[str] = field(
        default=None,
        help="Name of the tokenizer to use.",
    )
    taggers: List[str] = field(
        default=[],
        help="List of taggers to run.",
    )
    experiment: Optional[str] = field(
        default=None,
        help="Name of the experiment.",
    )
    processes: int = field(
        default=1,
        help="Number of parallel processes to use.",
    )
    batch_size: int = field(
        default=10_000,
        help="Number of sequences to tokenize before writing to disk.",
    )
    ring_size: int = field(
        default=8,
        help="Number of files to open in parallel for tokenization."
    )

    debug: bool = field(
        default=False,
        help="Whether to run in debug mode.",
    )
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    dryrun: bool = field(
        default=False,
        help="If true, only print the configuration and exit without running the taggers.",
    )


class TaggerCli(BaseCli):
    CONFIG = TaggerConfig
    DESCRIPTION = (
        "Tag documents or spans of documents using one or more taggers. "
        "For a list of available taggers, run `dolma list`."
    )

    @classmethod
    def run(cls, parsed_config: TaggerConfig):
        logger = get_logger("tagger")

        with make_workdirs(parsed_config.work_dir) as work_dirs:
            documents = [str(p) for p in parsed_config.documents]
            taggers = [str(p) for p in parsed_config.taggers]

            # perform some path validation to make sure we don't call the mixer with invalid config
            total_matching_documents = 0
            for document in documents:
                current_matching_documents = sum(1 for _ in glob_path(document))
                if current_matching_documents == 0:
                    # only raise a warning if no documents are found for a single path
                    logger.warning("No documents found for path %s", document)
                total_matching_documents += current_matching_documents

            if total_matching_documents == 0:
                # but raise an error if no documents are found for all paths
                raise DolmaConfigError(f"No documents found for paths {documents}.")

            print_config(parsed_config)
            if parsed_config.dryrun:
                logger.info("Exiting due to dryrun.")
                return

            create_and_run_tagger(
                documents=documents,
                destination=parsed_config.destination,
                metadata=work_dirs.output,
                taggers=taggers,
                ignore_existing=parsed_config.ignore_existing,
                num_processes=parsed_config.processes,
                experiment=parsed_config.experiment,
                debug=parsed_config.debug,
                profile_enable=parsed_config.profile.enable,
                profile_output=parsed_config.profile.output,
                profile_steps=parsed_config.profile.steps,
                profile_sort_key=parsed_config.profile.sort_key,
            )


@dataclass
class ListTaggerConfig:
    ...


class ListTaggerCli(BaseCli):
    CONFIG = ListTaggerConfig
    DESCRIPTION = "List available taggers."

    @classmethod
    def run(cls, parsed_config: ListTaggerConfig):
        table = Table(title="dolma taggers", style="bold")
        table.add_column("name", justify="left", style="cyan")
        table.add_column("class", justify="left", style="magenta")

        for tagger_name, tagger_cls in sorted(TaggerRegistry.taggers()):
            tagger_repr = f"{tagger_cls.__module__}.{tagger_cls.__name__}"
            table.add_row(tagger_name, tagger_repr)

        console = Console()
        console.print(table)
