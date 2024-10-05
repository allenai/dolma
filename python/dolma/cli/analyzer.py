from dataclasses import dataclass
from typing import List, Optional

from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig, make_workdirs
from dolma.core.analyzer import create_and_run_analyzer
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path


@dataclass
class AnalyzerConfig:
    attributes: List[str] = field(
        default=[],
        help="One or more attributes paths to process; Can be either local or S3 paths. Globs are supported.",
    )
    report: Optional[str] = field(
        default=None,
        help=(
            "Path where to save the report. Can be either local or S3 path. "
            "If not provided, the report will be printed to stdout."
        ),
    )
    bins: int = field(
        default=1_000,
        help="Number of bins to use for the histograms.",
    )
    processes: int = field(
        default=1,
        help="Number of parallel processes to use.",
    )
    seed: int = field(
        default=0,
        help="Seed to use for reproducibility.",
    )
    total: bool = field(
        default=False,
        help="Whether to include total count and sum in the report.",
    )
    debug: bool = field(
        default=False,
        help="Whether to run in debug mode.",
    )
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    regex: Optional[str] = field(
        default=None,
        help="Regex to use for filtering the attributes by name.",
    )


class AnalyzerCli(BaseCli):
    CONFIG = AnalyzerConfig
    DESCRIPTION = "Analyze the distribution of attributes values in a dataset."

    @classmethod
    def run(cls, parsed_config: AnalyzerConfig):
        logger = get_logger("analyzer")

        # perform some path validation to make sure we don't call the mixer with invalid config
        total_matching_documents = 0
        for document in parsed_config.attributes:
            current_matching_documents = sum(1 for _ in glob_path(document))
            if current_matching_documents == 0:
                # only raise a warning if no documents are found for a single path
                logger.warning("No documents found for path %s", document)
            total_matching_documents += current_matching_documents

        if total_matching_documents == 0:
            # but raise an error if no documents are found for all paths
            raise DolmaConfigError(f"No documents found for paths {parsed_config.attributes}.")

        print_config(parsed_config)

        with make_workdirs(parsed_config.work_dir) as work_dirs:
            create_and_run_analyzer(
                attributes=parsed_config.attributes,
                report=parsed_config.report,
                summaries_path=work_dirs.output,
                metadata_path=work_dirs.input,
                debug=parsed_config.debug,
                seed=parsed_config.seed,
                num_bins=parsed_config.bins,
                num_processes=parsed_config.processes,
                name_regex=parsed_config.regex,
                show_total=parsed_config.total,
            )
