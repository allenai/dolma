from dataclasses import dataclass
from typing import List, Optional

from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig, make_workdirs
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path
from dolma.warc import create_and_run_warc_pipeline


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
    skip_unknown_license: bool = field(
        default=False,
        help="Whether to skip documents with unknown licenses.",
    )
    html_extractor: str = field(
        default="trafilatura",
        help="HTML extractor to use.",
    )
    html_kwargs: Optional[dict] = field(
        default=None,
        help="HTML extractor arguments.",
    )
    license_extractor: str = field(
        default="cc_regex_fast",
        help="License extractor to use.",
    )
    license_kwargs: Optional[dict] = field(
        default=None,
        help="License extractor arguments.",
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
                skip_unknown_license=parsed_config.skip_unknown_license,
                html_extractor=parsed_config.html_extractor,
                html_kwargs=parsed_config.html_kwargs,
                license_extractor=parsed_config.license_extractor,
                license_kwargs=parsed_config.license_kwargs,
            )


@dataclass
class ListTaggerConfig:
    tagger_modules: List[str] = field(
        default=[],
        help="List of Python modules $PYTHONPATH to import custom taggers from.",
    )
