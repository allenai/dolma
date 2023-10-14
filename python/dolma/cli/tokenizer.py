from dataclasses import dataclass
from typing import List, Optional

from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig, make_workdirs
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path
from dolma.tokenizer import tokenize_in_parallel


@dataclass
class TokenizerConfig:
    documents: List[str] = field(
        default=[],
        help=(
            "One or more document paths to process; Can be either local or S3 paths. "
            "Globs are supported. Required"
        ),
    )
    destination: Optional[str] = field(
        default=None,
        help=(
            "Destination paths to save the outputs; should match the number of document paths. "
            "If not provided, destination will be derived from the document path. Required."
        ),
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        help="Name or path of the tokenizer to use. Must be a HuggingFace-compatible tokenizer. Required.",
    )
    processes: int = field(
        default=1,
        help="Number of parallel processes to use.",
    )
    files_per_process: Optional[int] = field(
        default=None,
        help="Number of files to process per process.",
    )
    batch_size: int = field(
        default=10_000,
        help="Number of sequences to tokenize before writing to disk.",
    )
    ring_size: int = field(default=8, help="Number of files to open in parallel for tokenization.")
    max_size: int = field(
        default=1024 * 1024 * 1024,
        help="Maximum size of a file in bytes.",
    )
    dtype: str = field(
        default="uint16",
        help="Data type for the memmap file; must be a valid numpy dtype.",
    )
    debug: bool = field(
        default=False,
        help="Whether to run in debug mode.",
    )
    seed: int = field(
        default=3920,
        help="Seed for random number generation.",
    )
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    dryrun: bool = field(
        default=False,
        help="If true, only print the configuration and exit without running the taggers.",
    )


class TokenizerCli(BaseCli):
    CONFIG = TokenizerConfig
    DESCRIPTION = "Tokenize documents using the provided tokenizer."

    @classmethod
    def run(cls, parsed_config: TokenizerConfig):
        logger = get_logger("tagger")

        with make_workdirs(parsed_config.work_dir) as work_dirs:
            documents = [str(p) for p in parsed_config.documents]

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

            if parsed_config.destination is None:
                raise DolmaConfigError("Destination must be provided.")

            if parsed_config.tokenizer_name_or_path is None:
                raise DolmaConfigError("Tokenizer ID must be provided.")

            tokenize_in_parallel(
                sources=documents,
                destination=parsed_config.destination,
                num_writers=parsed_config.processes,
                num_readers=parsed_config.files_per_process,
                local_shuffle=parsed_config.batch_size,
                ring_size=parsed_config.ring_size,
                tokenizer_name_or_path=parsed_config.tokenizer_name_or_path,
                seed=parsed_config.seed,
                metadata_dir=work_dirs.output,
                max_size=parsed_config.max_size,
                debug=parsed_config.debug,
            )
