from dataclasses import dataclass
from typing import List

from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig, make_workdirs
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path
from dolma.warc import create_and_run_warc_pipeline


@dataclass
class TaggerConfig:
    taggers: List[str] = field(
        default=[],
        help="List of taggers to run.",
    )
    skip: bool = field(
        default=False,
        help="Whether to skip if taggers returns no output.",
    )


@dataclass
class WarcExtractorConfig:
    documents: List[str] = field(
        default=[],
        help="One or more document paths to process; Can be either local or S3 paths. Globs are supported.",
    )
    destination: List[str] = field(
        default=[],
        nargs="*",
        help=(
            "Destination paths to save the outputs; should match the number of document paths. "
            "If not provided, destination will be derived from the document path."
        ),
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
    source_name: str = field(help="Name to assign to the source.")
    linearizer: str = field(
        default="resiliparse",
        help="Name of the HTML linearizer to use.",
    )
    pre: TaggerConfig = field(default=TaggerConfig(), help="Configuration for pre-extraction taggers.")
    post: TaggerConfig = field(default=TaggerConfig(), help="Configuration for post-extraction taggers.")
    store_html_in_metadata: bool = field(
        default=False,
        help="Whether to store the HTML content in the metadata.",
    )

    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    dryrun: bool = field(
        default=False,
        help="If true, only print the configuration and exit without running the taggers.",
    )


class WarcExtractorCli(BaseCli):
    CONFIG = WarcExtractorConfig
    DESCRIPTION = "Extract documents from WARC files and parse HTML out."

    @classmethod
    def run(cls, parsed_config: WarcExtractorConfig):
        logger = get_logger("warc")

        with make_workdirs(parsed_config.work_dir) as work_dirs:
            documents = [str(p) for p in parsed_config.documents]
            destination = [str(p) for p in parsed_config.destination]

            source_name = parsed_config.source_name
            if not isinstance(source_name, str):
                raise ValueError(f"source_name must be a string, not {source_name} ({type(source_name)})")

            # perform some path validation to make sure we don't call
            # the extractor with invalid config
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

            create_and_run_warc_pipeline(
                documents=(documents[0] if len(documents) == 1 else documents),
                destination=(destination[0] if len(destination) == 1 else destination),
                metadata=work_dirs.output,
                num_processes=parsed_config.processes,
                ignore_existing=parsed_config.ignore_existing,
                debug=parsed_config.debug,
                source_name=source_name,
                pre_taggers=parsed_config.pre.taggers,
                skip_no_pre_taggers=parsed_config.pre.skip,
                post_taggers=parsed_config.post.taggers,
                skip_no_post_taggers=parsed_config.post.skip,
                store_html_in_metadata=parsed_config.store_html_in_metadata,
                linearizer_name=parsed_config.linearizer,
            )
