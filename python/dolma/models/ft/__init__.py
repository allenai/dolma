from dataclasses import dataclass
from typing import List

from omegaconf import OmegaConf as om

from ...cli import BaseCli, field
from ...core.errors import DolmaConfigError
from .config import FastTextTrainerConfig
from .trainer import FastTextTrainer


@dataclass
class FastTextCliConfig(FastTextTrainerConfig):
    action: List[str] = field(default=[], help="Action to perform (train/valid/test)")


class FastTextCli(BaseCli):
    CONFIG = FastTextCliConfig
    DESCRIPTION = "Subcommand for training and evaluating fasttext-based taggers."

    @classmethod
    def run(cls, parsed_config: FastTextCliConfig):
        if om.is_missing(parsed_config, "action"):
            raise DolmaConfigError("At least one action must be provided")

        if om.is_missing(parsed_config, "model_path"):
            raise DolmaConfigError("model_path must be provided")

        if parsed_config.data is None or parsed_config.streams is None:
            raise ValueError("At least one of `data` or `streams` must be provided")

        trainer = FastTextTrainer(parsed_config)
        if "train" in parsed_config.action:
            trainer.do_train()
        if "valid" in parsed_config.action:
            trainer.do_valid()
        if "test" in parsed_config.action:
            trainer.do_test()
