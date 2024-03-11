from argparse import Namespace
from typing import TYPE_CHECKING, Optional, Union

from necessary import necessary

from ...core.loggers import get_logger
from ..data import FastTextUnsupervisedDataConverter
from ..fasttext_model.trainer import FastTextTrainer, FastTextUnsupervisedTrainer
from .config import FloretSupervisedTrainerConfig

with necessary("floret", soft=True) as FLORET_AVALIABLE:
    if TYPE_CHECKING or FLORET_AVALIABLE:
        import floret
        from floret.floret import _floret as FloretModel


LOGGER = get_logger(__name__)


class FloretTrainer(FastTextTrainer):
    def __init__(self, config: FloretSupervisedTrainerConfig):
        if not FLORET_AVALIABLE:
            raise ImportError("fasttext is not available. Install it using `pip install fasttext-wheel`")

        if config.model.mode not in ("floret", "default"):
            raise ValueError(f"mode must be one of 'floret' or 'default', got {config.model.mode}")

        if config.model.hash_count not in range(1, 5):
            raise ValueError(f"hash_count must be in range 1-4, got {config.model.hash_count}")

        super().__init__(config=config)

    def _train_unsupervised(self, **kwargs) -> FloretModel:
        kwargs = {
            "mode": self.config.model.mode,
            "hashCount": self.config.model.hash_count,
            **kwargs,
        }
        return floret.train_unsupervised(**kwargs)

    def _train_supervised(self, **kwargs) -> FloretModel:
        kwargs = {
            "mode": self.config.model.mode,
            "hashCount": self.config.model.hash_count,
            **kwargs,
        }
        return floret.train_supervised(**kwargs)

    def _load_model(self, path: str, args: Optional[dict] = None) -> FloretModel:
        parsed_args = Namespace(**args) if args is not None else None
        return FloretModel(model_path=path, args=parsed_args)


class FloretUnsupervisedTrainer(FloretSupervisedTrainerConfig, FastTextUnsupervisedTrainer):
    def fit(self, data_path: str, save_path: str, validation_path: Union[str, None] = None):
        return FastTextUnsupervisedTrainer.fit(self, data_path, save_path, validation_path)

    @property
    def data_factory_cls(self):
        return FastTextUnsupervisedDataConverter
