from typing import List, Optional
from omegaconf import DictConfig, OmegaConf as om
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace

from dolma.cli import BaseCli
from dolma import deduper


@dataclass
class ParagraphDedupeConfig:
    attribute_name: str


@dataclass
class DocumentDedupeConfig:
    attribute_name: str
    key: str


@dataclass
class BloomFilterConfig:
    file: str
    size_in_bytes: int
    read_only: bool
    estimated_doc_count: int
    desired_false_positive_rate: float


@dataclass
class DedupeConfig:
    name: str
    documents: Optional[DocumentDedupeConfig] = None
    paragraphs: Optional[ParagraphDedupeConfig] = None
    skip_empty: Optional[bool] = None


@dataclass
class DeduperConfig:
    documents: List[str]
    work_dir: str
    dedupe: DedupeConfig
    bloom_filter: BloomFilterConfig
    processes: int


class DeduperCli(BaseCli):
    @classmethod
    def make_parser(cls, parser: ArgumentParser):
        parser.add_argument(
            '-c',
            '--config',
            required=True,
            type=str,
            help="Path to the config file",
        )

    @classmethod
    def run_from_args(cls, args: Namespace):
        config = om.load(args.config)
        assert isinstance(config, DictConfig)

        if config.dedupe.documents is None and config.dedupe.paragraphs is None:
            raise ValueError("Either dedupe.documents or dedupe.paragraphs must be specified")

        config_dict = om.to_container(config)
        assert isinstance(config_dict, dict)
        deduper(config_dict)
