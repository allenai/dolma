from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf as om

from dolma import deduper
from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path


@dataclass
class ParagraphDedupeConfig:
    attribute_name: str = field(help="Name of the output field in the tagger")


@dataclass
class DocumentDedupeConfig:
    attribute_name: str = field(help="Name of the output field in the tagger")
    key: str = field(help="Name of the input field to use for deduplication, e.g. `$.metadata.url`")


@dataclass
class BloomFilterConfig:
    file: str = field(help="Path where to read/write the bloom filter file to/from. Required.")
    size_in_bytes: int = field(
        default=-1,
        help=(
            "Size of the bloom filter in bytes. Either this value is provided, or both estimated_doc_count "
            "and desired_false_positive_rate."
        ),
    )
    read_only: bool = field(help="If true, the bloom filter will be read from the file and not updated. Required.")
    estimated_doc_count: int = field(
        default=-1,
        help=(
            "Estimated number of documents to be added to the bloom filter. Either this value is provided, "
            "or both size_in_bytes and desired_false_positive_rate."
        ),
    )
    desired_false_positive_rate: float = field(
        default=-1.0,
        help=(
            "Desired false positive rate. Either this value is provided, or both size_in_bytes and "
            "estimated_doc_count."
        ),
    )


@dataclass
class DedupeConfig:
    name: str = field(help="Name of the deduper. Required.")
    documents: Optional[DocumentDedupeConfig] = field(
        default=None, help="Configuration for document deduplication"
    )
    paragraphs: Optional[ParagraphDedupeConfig] = field(
        default=None, help="Configuration for paragraph deduplication"
    )
    skip_empty: Optional[bool] = field(default=False, help="If true, empty documents/paragraphs will be skipped")


@dataclass
class DeduperConfig:
    documents: List[str] = field(default=[], help="Paths to the documents to be deduplicated. Required.")
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    dedupe: DedupeConfig = field(help="Deduplication configuration. Required.")
    bloom_filter: BloomFilterConfig = field(help="Bloom filter configuration. Required.")
    processes: int = field(
        default=1, help="Number of processes to use for deduplication. If 1, no multiprocessing will be used."
    )


class DeduperCli(BaseCli):
    CONFIG = DeduperConfig
    DESCRIPTION = "Deduplicate documents or paragraphs using a bloom filter."

    @classmethod
    def run(cls, parsed_config: DeduperConfig):
        logger = get_logger("tagger")

        dict_config: Dict[str, Any] = {}

        dict_config["dedupe"] = {"name": parsed_config.dedupe.name, "skip_empty": parsed_config.dedupe.skip_empty}
        if parsed_config.dedupe.documents is not None:
            dict_config["dedupe"]["documents"] = om.to_container(parsed_config.dedupe.documents)
        elif parsed_config.dedupe.paragraphs is not None:
            dict_config["dedupe"]["paragraphs"] = om.to_container(parsed_config.dedupe.paragraphs)
        else:
            raise ValueError("Either dedupe.documents or dedupe.paragraphs must be specified")

        # perform some path validation to make sure we don't call the mixer with invalid config
        total_matching_documents = 0
        for document in parsed_config.documents:
            if document.count("*") > 1:
                raise DolmaConfigError("Only one wildcard is allowed in the document path")

            current_matching_documents = sum(1 for _ in glob_path(document))
            if current_matching_documents == 0:
                # only raise a warning if no documents are found for a single path
                logger.warn(f"No documents found for path {document}")
            total_matching_documents += current_matching_documents

        if total_matching_documents == 0:
            # but raise an error if no documents are found for all paths
            raise DolmaConfigError(f"No documents found for the paths {parsed_config.documents}.")

        dict_config["bloom_filter"] = {
            "file": parsed_config.bloom_filter.file,
            "read_only": parsed_config.bloom_filter.read_only,
            "size_in_bytes": getattr(parsed_config.bloom_filter, "size_in_bytes", 0),
            "estimated_doc_count": getattr(parsed_config.bloom_filter, "estimated_doc_count", 0),
            "desired_false_positive_rate": getattr(parsed_config.bloom_filter, "desired_false_positive_rate", 0),
        }

        if dict_config["bloom_filter"]["size_in_bytes"] <= 0 and (
            dict_config["bloom_filter"]["estimated_doc_count"] <= 0
            or dict_config["bloom_filter"]["desired_false_positive_rate"] <= 0
        ):
            raise ValueError(
                "Either bloom_filter.size_in_bytes or bloom_filter.estimated_doc_count and "
                "bloom_filter.desired_false_positive_rate must be specified"
            )

        dict_config["work_dir"] = {"input": parsed_config.work_dir.input, "output": parsed_config.work_dir.output}
        dict_config["processes"] = parsed_config.processes
        dict_config["documents"] = list(om.to_container(parsed_config.documents))  # pyright: ignore

        if len(dict_config["documents"]) == 0:
            raise ValueError("At least one document must be specified")

        print_config(dict_config)
        return deduper(dict_config)
