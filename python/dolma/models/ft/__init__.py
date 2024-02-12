from dataclasses import dataclass
from typing import List, Type, Union

from omegaconf import OmegaConf as om

from ...cli import BaseCli, field
from ...core.errors import DolmaConfigError
from ..config import BaseTrainerConfig
from ..trainer import BaseTrainer
from .config import FastTextSupervisedTrainerConfig, FastTextUnsupervisedTrainerConfig
from .trainer import FastTextTrainer


@dataclass
class _FastTextCliConfig(BaseTrainerConfig):
    action: List[str] = field(default=[], help="Action to perform (train/valid/test)")


@dataclass
class FastTextSupervisedCliConfig(_FastTextCliConfig, FastTextSupervisedTrainerConfig):
    pass


@dataclass
class FastTextUnsupervisedCliConfig(_FastTextCliConfig, FastTextUnsupervisedTrainerConfig):
    pass


class _FastTextCli(BaseCli):
    CONFIG: Type[_FastTextCliConfig]
    TRAINER: Type[BaseTrainer]
    DESCRIPTION: str

    @classmethod
    def run(cls, parsed_config: _FastTextCliConfig):
        if om.is_missing(parsed_config, "action"):
            raise DolmaConfigError("At least one action must be provided")

        if om.is_missing(parsed_config, "model_path"):
            raise DolmaConfigError("model_path must be provided")

        if parsed_config.data is None and parsed_config.streams is None:
            raise ValueError("At least one of `data` or `streams` must be provided")

        trainer = cls.TRAINER(parsed_config)
        if "train" in parsed_config.action:
            trainer.do_train()
        if "valid" in parsed_config.action:
            trainer.do_valid()
        if "test" in parsed_config.action:
            trainer.do_test()


class FastTextSupervisedCli(_FastTextCli):
    CONFIG = FastTextSupervisedCliConfig
    TRAINER = FastTextTrainer
    DESCRIPTION = "Subcommand for training and evaluating supervised fasttext-based taggers."


class FastTextUnsupervisedCli(_FastTextCli):
    CONFIG = FastTextUnsupervisedCliConfig
    TRAINER = FastTextTrainer
    DESCRIPTION = "Subcommand for training and evaluating unsupervised fasttext-based taggers."
