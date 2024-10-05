from dataclasses import dataclass
from pstats import SortKey
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
from dolma.core.utils import import_modules


@dataclass
class ProfilerConfig:
    enable: bool = field(
        default=False,
        help="Whether to enable profiling.",
    )
    output: Optional[str] = field(
        default=None,
        help="Path to save the profiling output; if not provided, the output will be printed to stdout.",
    )
    steps: Optional[int] = field(
        default=None,
        help="List of steps to profile; if not provided, all steps will be profiled.",
    )
    sort_key: str = field(
        default="tottime",
        choices=[str(k) for k in SortKey._value2member_map_],  # pylint: disable=no-member,protected-access
        help="Sort key for the profiling output.",
    )
    lines: int = field(
        default=100,
        help="Number of rows to print in the profiling output.",
    )


@dataclass
class TaggerConfig:
    documents: List[str] = field(
        default=[],
        help="One or more document paths to process; Can be either local or S3 paths. Globs are supported.",
    )
    destination: Optional[List[str]] = field(
        default=None,
        nargs="*",
        help=(
            "Destination paths to save the outputs; should match the number of document paths. "
            "If not provided, destination will be derived from the document path."
        ),
    )
    tagger_modules: List[str] = field(
        default=[],
        help=(
            "Additional modules to import taggers from; this is useful for taggers that are not part of Dolma. "
            "Modules must be available in $PYTHONPATH or a path to module. Taggers should be registered using the "
            "@dolma.add_tagger(...) decorator."
        ),
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
    ignore_existing: bool = field(
        default=False,
        help="Whether to ignore existing outputs and re-run the taggers.",
    )
    debug: bool = field(
        default=False,
        help="Whether to run in debug mode.",
    )
    profile: ProfilerConfig = field(
        default=ProfilerConfig(),
        help="Whether to run in profiling mode.",
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
                taggers_modules=parsed_config.tagger_modules,
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
    tagger_modules: List[str] = field(
        default=[],
        help="List of Python modules $PYTHONPATH to import custom taggers from.",
    )


class ListTaggerCli(BaseCli):
    CONFIG = ListTaggerConfig
    DESCRIPTION = "List available taggers."

    @classmethod
    def run(cls, parsed_config: ListTaggerConfig):
        # import tagger modules
        import_modules(parsed_config.tagger_modules)

        table = Table(title="dolma taggers", style="bold")
        table.add_column("name", justify="left", style="cyan")
        table.add_column("class", justify="left", style="magenta")

        for tagger_name, tagger_cls in sorted(TaggerRegistry.items()):
            tagger_repr = f"{tagger_cls.__module__}.{tagger_cls.__name__}"
            table.add_row(tagger_name, tagger_repr)

        console = Console()
        console.print(table)
