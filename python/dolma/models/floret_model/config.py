from dataclasses import dataclass

from ...cli import field
from ..fasttext_model.config import (
    FastTextSupervisedModelConfig,
    FastTextSupervisedTrainerConfig,
    FastTextUnsupervisedModelConfig,
    FastTextUnsupervisedTrainerConfig,
)


@dataclass
class FloretSupervisedModelConfig(FastTextSupervisedModelConfig):
    mode: str = field(
        help="Whether to run floret in fasttext-compatible mode (default) or in floret mode (floret)",
        default="floret",
    )
    hash_count: int = field(
        help="Number of hashes (1-4) per word/subword. Choose between 1 and 4. More hashes mean longer training, but also more accuracy. Defaults to 1",
        default=1,
    )


@dataclass
class FloretSupervisedTrainerConfig(FastTextSupervisedTrainerConfig):
    model: FloretSupervisedModelConfig = field(help="Model configuration", default=FloretSupervisedModelConfig())


@dataclass
class FloretUnsupervisedModelConfig(FastTextUnsupervisedModelConfig):
    mode: str = field(
        help="Whether to run floret in fasttext-compatible mode (default) or in floret mode (floret).",
        default="floret",
    )
    hash_count: int = field(
        help="Number of hashes (1-4) per word/subword. Choose between 1 and 4. More hashes mean longer training, but also more accuracy. Defaults to 1.",
        default=1,
    )


@dataclass
class FloretUnsupervisedTrainerConfig(FastTextUnsupervisedTrainerConfig):
    model: FloretUnsupervisedModelConfig = field(
        help="Model configuration", default=FloretUnsupervisedModelConfig()
    )
