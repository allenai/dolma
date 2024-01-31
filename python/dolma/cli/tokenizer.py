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
    name_or_path: Optional[str] = field(
        default=None,
        help="Name or path of the tokenizer to use. Must be a HuggingFace-compatible tokenizer. Required.",
    )
    bos_token_id: Optional[int] = field(
        default=None, help="The token ID corresponding to the 'beginning-of-sentence' token."
    )
    eos_token_id: Optional[int] = field(
        default=None,
        help="The token ID corresponding to the 'end-of-sentence' token.",
    )
    pad_token_id: Optional[int] = field(
        default=None,
        help="The token ID corresponding to the 'padding' token.",
    )
    segment_before_tokenization: bool = field(
        default=False,
        help=(
            "Whether to segment documents by paragraph before tokenization. "
            "This is useful for tokenizers like Llama that are very slow on long documents. "
            "Might not be needed once this bugfix is merged https://github.com/huggingface/tokenizers/pull/1413"
        ),
    )

    def __post__init__(self):
        logger = get_logger(__file__)

        if self.eos_token_id is None:
            logger.warning("NO EOS TOKEN PROVIDED. Are you sure this is what you want?")

        if self.bos_token_id is None:
            logger.warning("NO BOS TOKEN PROVIDED. Are you sure this is what you want?")

        if self.pad_token_id is None:
            logger = get_logger(__file__)
            logger.warning("No pad token ID provided; using EOS token ID.")
            self.pad_token_id = self.eos_token_id

        if self.segment_before_tokenization:
            logger.warning(
                "EXPERIMENTAL FEATURE: segmenting before tokenization is enabled. "
                "This option has only been tested with Llama and GPT-NeoX tokenizers. "
                "USE AT YOUR OWN RISK."
            )

    @classmethod
    def deprecated_init(cls, tokenizer_name_or_path: str) -> "TokenizerConfig":
        logger = get_logger(__file__)
        logger.warning(
            "The `tokenizer_name_or_path` argument is deprecated and will be removed in a future release. "
            "Please use --tokenizer.name_or_path, and provide --tokenizer.eos_token_id as well "
            "(and, optionally, --tokenizer.pad_token_id)."
        )
        from tokenizers import Tokenizer

        # before options to pass eos_token_id and pad_token_id were added, eos was set to
        # the last token in the vocab. We need to do the same here to maintain compatibility
        legacy_tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path)
        old_eos_token_id = len(legacy_tokenizer.get_vocab()) - 1

        return cls(
            name_or_path=tokenizer_name_or_path,
            eos_token_id=old_eos_token_id,
            segment_before_tokenization=False,
        )


@dataclass
class TokenizationConfig:
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
        help="Deprecated. Use --tokenizer.name_or_path instead.",
    )
    tokenizer: Optional[TokenizerConfig] = field(
        default=None,
        help="Configuration for the tokenizer.",
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
    CONFIG = TokenizationConfig
    DESCRIPTION = "Tokenize documents using the provided tokenizer."

    @classmethod
    def run(cls, parsed_config: TokenizationConfig):
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

            # must handle new and deprecated way to get tokenizer config
            if parsed_config.tokenizer is None:
                if parsed_config.tokenizer_name_or_path is None:
                    raise DolmaConfigError("Tokenizer configuration is missing.")
                else:
                    parsed_config.tokenizer = TokenizerConfig.deprecated_init(parsed_config.tokenizer_name_or_path)

            if parsed_config.tokenizer.name_or_path is None:
                raise DolmaConfigError("Tokenizer name or path must be provided.")

            tokenize_in_parallel(
                sources=documents,
                destination=parsed_config.destination,
                num_writers=parsed_config.processes,
                num_readers=parsed_config.files_per_process,
                local_shuffle=parsed_config.batch_size,
                ring_size=parsed_config.ring_size,
                tokenizer_name_or_path=parsed_config.tokenizer.name_or_path,
                bos_token_id=parsed_config.tokenizer.bos_token_id,
                eos_token_id=parsed_config.tokenizer.eos_token_id,
                pad_token_id=parsed_config.tokenizer.pad_token_id,
                segment_before_tokenization=parsed_config.tokenizer.segment_before_tokenization,
                seed=parsed_config.seed,
                metadata_dir=work_dirs.output,
                max_size=parsed_config.max_size,
                debug=parsed_config.debug,
            )
