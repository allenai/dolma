from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf as om
from argparse import ArgumentParser, Namespace
from tempfile import gettempdir

from dolma.cli import BaseCli, field, namespace_to_nested_omegaconf, make_parser
from dolma import deduper


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
        )
    )
    read_only: bool = field(
        help="If true, the bloom filter will be read from the file and not updated. Required."
    )
    estimated_doc_count: int = field(
        default=-1,
        help=(
            "Estimated number of documents to be added to the bloom filter. Either this value is provided, "
            "or both size_in_bytes and desired_false_positive_rate."
        )
    )
    desired_false_positive_rate: float = field(
        default=-1.,
        help=(
            "Desired false positive rate. Either this value is provided, or both size_in_bytes and "
            "estimated_doc_count."
        )
    )


@dataclass
class DedupeConfig:
    name: str
    documents: Optional[DocumentDedupeConfig] = field(
        default=None,
        help="Configuration for document deduplication"
    )
    paragraphs: Optional[ParagraphDedupeConfig] = field(
        default=None,
        help="Configuration for paragraph deduplication"
    )
    skip_empty: Optional[bool] = field(
        default=False,
        help="If true, empty documents/paragraphs will be skipped"
    )


@dataclass
class WorkDirConfig:
    input: str = field(
        default=str(Path(gettempdir()) / 'dolma' / 'deduper' / 'input'),
        help="Path to the input directory. Required."
    )
    output: str = field(
        default=str(Path(gettempdir()) / 'dolma' / 'deduper' / 'output'),
        help="Path to the output directory. Required."
    )


@dataclass
class DeduperConfig:
    documents: List[str] = field(
        help="Paths to the documents to be deduplicated. Required."
    )
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Path to the working directory")
    dedupe: DedupeConfig = field(help="Deduplication configuration. Required.")
    bloom_filter: BloomFilterConfig = field(help="Bloom filter configuration. Required.")
    processes: int = field(
        default=1,
        help="Number of processes to use for deduplication. If 1, no multiprocessing will be used."
    )


class DeduperCli(BaseCli):
    @classmethod
    def make_parser(cls, parser: ArgumentParser):
        make_parser(parser, DeduperConfig)

    @classmethod
    def run_from_args(cls, args: Namespace):
        config = namespace_to_nested_omegaconf(args, DeduperConfig)
        dict_config: Dict[str, Any] = {}

        dict_config['dedupe'] = {
            'name': config.dedupe.name,
            'skip_empty': config.dedupe.skip_empty
        }
        if config.dedupe.documents is not None:
            dict_config['dedupe']['documents'] = om.to_container(config.dedupe.documents)
        elif config.dedupe.paragraphs is None:
            dict_config['dedupe']['paragraphs'] = om.to_container(config.dedupe.paragraphs)
        else:
            raise ValueError("Either dedupe.documents or dedupe.paragraphs must be specified")

        dict_config['bloom_filter'] = {
            'file': config.bloom_filter.file,
            'read_only': config.bloom_filter.read_only
        }
        if config.bloom_filter.size_in_bytes > 0:
            dict_config['bloom_filter']['size_in_bytes'] = config.bloom_filter.size_in_bytes
        if config.bloom_filter.estimated_doc_count > 0 and config.bloom_filter.desired_false_positive_rate > 0:
            dict_config['bloom_filter'].update({
                'estimated_doc_count': config.bloom_filter.estimated_doc_count,
                'desired_false_positive_rate': config.bloom_filter.desired_false_positive_rate
            })
        else:
            raise ValueError(
                "Either bloom_filter.size_in_bytes or bloom_filter.estimated_doc_count and "
                "bloom_filter.desired_false_positive_rate must be specified"
            )

        dict_config['work_dir'] = {
            'input': config.work_dir.input,
            'output': config.work_dir.output
        }
        dict_config['num_processes'] = config.processes

        deduper(dict_config)
