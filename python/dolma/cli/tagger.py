from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Optional, Union

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from dolma.cli import BaseCli, field
from dolma.cli.shared import WorkDirConfig
from dolma.core.runtime import TaggerProcessor


@dataclass
class TaggerConfig:
    documents: List[str] = field(
        default=[],
        help="One or more document paths to process; Can be either local or S3 paths. Globs are supported.",
    )
    destination: Optional[List[str]] = field(
        default=None,
        help=(
            "Destination paths to save the outputs; should match the number of document paths. "
            "If not provided, destination will be derived from the document path."
        ),
    )
    taggers: List[str] = field(
        default=[],
        help="List of taggers to run.",
    )
    processes: int = field(
        default=1,
        help="Number of parallel processes to use.",
    )
    work_dir: Optional[WorkDirConfig] = field(
        default=WorkDirConfig(), help="Configuration for temporary work directories."
    )


class TaggerCli(BaseCli):
    CONFIG = TaggerConfig

    @classmethod
    def run(cls, parsed_config: TaggerConfig):
        raise NotImplementedError("TaggerCli.run() is not implemented yet.")
