from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf as om

from dolma import deduper
from dolma.cli import BaseCli, field, make_parser, namespace_to_nested_omegaconf


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
class WorkDirConfig:
    input: str = field(
        default=str(Path(gettempdir()) / "dolma" / "deduper" / "input"),
        help="Path to the input directory. Required.",
    )
    output: str = field(
        default=str(Path(gettempdir()) / "dolma" / "deduper" / "output"),
        help="Path to the output directory. Required.",
    )


@dataclass
class DeduperConfig:
    documents: List[str] = field(help="Paths to the documents to be deduplicated. Required.")
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Path to the working directory")
    dedupe: DedupeConfig = field(help="Deduplication configuration. Required.")
    bloom_filter: BloomFilterConfig = field(help="Bloom filter configuration. Required.")
    processes: int = field(
        default=1, help="Number of processes to use for deduplication. If 1, no multiprocessing will be used."
    )


class DeduperCli(BaseCli):
    @classmethod
    def make_parser(cls, parser: ArgumentParser):
        make_parser(parser, DeduperConfig)

    @classmethod
    def run_from_args(cls, args: Namespace, config: Optional[dict] = None):
        parsed_config = namespace_to_nested_omegaconf(args=args, structured=DeduperConfig, config=config)
        dict_config: Dict[str, Any] = {}

        dict_config["dedupe"] = {"name": parsed_config.dedupe.name, "skip_empty": parsed_config.dedupe.skip_empty}
        if parsed_config.dedupe.documents is not None:
            dict_config["dedupe"]["documents"] = om.to_container(parsed_config.dedupe.documents)
        elif parsed_config.dedupe.paragraphs is None:
            dict_config["dedupe"]["paragraphs"] = om.to_container(parsed_config.dedupe.paragraphs)
        else:
            raise ValueError("Either dedupe.documents or dedupe.paragraphs must be specified")

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

        print(dict_config)

        deduper(dict_config)
