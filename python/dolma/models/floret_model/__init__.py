from dataclasses import dataclass

from ..fasttext_model import _FastTextCli, _FastTextCliConfig
from .config import FloretSupervisedTrainerConfig, FloretUnsupervisedTrainerConfig
from .trainer import FloretTrainer, FloretUnsupervisedTrainer


@dataclass
class FloretSupervisedCliConfig(FloretSupervisedTrainerConfig, _FastTextCliConfig):
    """Configuration for the supervised floret CLI; most options come
    from FloretSupervisedTrainerConfig."""


@dataclass
class FloretUnsupervisedCliConfig(FloretUnsupervisedTrainerConfig, _FastTextCliConfig):
    """Configuration for the unsupervised floret CLI; most options come
    from FloretUnsupervisedTrainerConfig."""


class FloretSupervisedCli(_FastTextCli):
    CONFIG = FloretSupervisedCliConfig
    TRAINER = FloretTrainer
    DESCRIPTION = "Subcommand for training and evaluating supervised floret-based taggers."


class FloretUnsupervisedCli(_FastTextCli):
    CONFIG = FloretUnsupervisedCliConfig
    TRAINER = FloretUnsupervisedTrainer
    DESCRIPTION = "Subcommand for training and evaluating unsupervised floret-based taggers."
