from dataclasses import dataclass
from typing import List, Type, Union

from omegaconf import OmegaConf as om

from dolma.core.loggers import get_logger

from ...cli import BaseCli, field, print_config
from ...core.errors import DolmaConfigError
from ..config import BaseTrainerConfig
from ..trainer import BaseTrainer
from .config import (
    FastTextQuantizerTrainerConfig,
    FastTextSupervisedTrainerConfig,
    FastTextUnsupervisedTrainerConfig,
)
from .trainer import (
    FastTextQuantizerTrainer,
    FastTextTrainer,
    FastTextUnsupervisedTrainer,
)


@dataclass
class _FastTextCliConfig(BaseTrainerConfig):
    """Base CLI configuration to derive both supervised and unsupervised configurations."""

    action: List[str] = field(default=[], help="Action to perform (train/valid/test)")
    dryrun: bool = field(default=False, help="Dry run (do not actually run the command)")


@dataclass
class FastTextSupervisedCliConfig(FastTextSupervisedTrainerConfig, _FastTextCliConfig):
    """Configuration for the supervised fasttext CLI; most options come
    from FastTextSupervisedTrainerConfig."""

    pass


@dataclass
class FastTextUnsupervisedCliConfig(FastTextUnsupervisedTrainerConfig, _FastTextCliConfig):
    """Configuration for the unsupervised fasttext CLI; most options come
    from FastTextUnsupervisedTrainerConfig."""

    pass


@dataclass
class FastTexQuantizerCliConfig(FastTextQuantizerTrainerConfig, _FastTextCliConfig):
    """Configuration for the unsupervised fasttext CLI; most options come
    from FastTextUnsupervisedTrainerConfig."""

    pass


class _FastTextCli(BaseCli):
    CONFIG: Type[_FastTextCliConfig]
    TRAINER: Type[BaseTrainer]
    DESCRIPTION: str

    @classmethod
    def run(cls, parsed_config: _FastTextCliConfig):
        logger = get_logger(cls.__name__)

        if om.is_missing(parsed_config, "action"):
            raise DolmaConfigError("At least one action must be provided")

        if om.is_missing(parsed_config, "model_path"):
            raise DolmaConfigError("model_path must be provided")

        if parsed_config.data is None and parsed_config.streams is None:
            raise ValueError("At least one of `data` or `streams` must be provided")

        if any(action not in ["train", "valid", "test"] for action in parsed_config.action):
            raise ValueError(f"valid actions are 'train', 'valid', and 'test'; got: {parsed_config.action}")

        print_config(parsed_config)
        if parsed_config.dryrun:
            logger.info("Exiting due to dryrun.")
            return

        trainer = cls.TRAINER(parsed_config)
        if "train" in parsed_config.action:
            logger.info("Starting training...")
            trainer.do_train()
            logger.info("Training complete.")
        if "valid" in parsed_config.action:
            logger.info("Starting validation...")
            trainer.do_valid()
            logger.info("Validation complete.")
        if "test" in parsed_config.action:
            logger.info("Starting testing...")
            trainer.do_test()
            logger.info("Testing complete.")


class FastTextSupervisedCli(_FastTextCli):
    CONFIG = FastTextSupervisedCliConfig
    TRAINER = FastTextTrainer
    DESCRIPTION = "Subcommand for training and evaluating supervised fasttext-based taggers."


class FastTextUnsupervisedCli(_FastTextCli):
    CONFIG = FastTextUnsupervisedCliConfig
    TRAINER = FastTextUnsupervisedTrainer
    DESCRIPTION = "Subcommand for training and evaluating unsupervised fasttext-based taggers."


class FastTextQuantizeCli(_FastTextCli):
    CONFIG = FastTexQuantizerCliConfig
    TRAINER = FastTextQuantizerTrainer
    DESCRIPTION = "Subcommand for quantizing fasttext models."
