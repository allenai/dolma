from functools import partial
from typing import NamedTuple, Type

import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .loggers import get_logger
from .utils import get_local_gpu_rank, sanitize_model_name


class Prediction(NamedTuple):
    label: str
    score: float


class BaseQualityClassifier:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, model_name: str, device: str, dtype: str, compile: bool = False):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=True,
        ).to(torch.device(device))

        if compile:
            self.model = torch.compile(self.model)  # pyright: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()


        if len(self.model.config.id2label) > 1:
            label_name_fn = lambda label: f"{sanitize_model_name(model_name)}_{sanitize_model_name(label)}"
        else:
            label_name_fn = lambda label: sanitize_model_name(model_name)

        self.labels_map = {
            id_: label_name_fn(label)
            for id_, label in self.model.config.id2label.items()
        }

    @property
    def device(self) -> torch.device:
        return self.model.device

    def score(self, **batch: torch.Tensor) -> list[list[Prediction]]:
        outputs = self.model(**batch)
        scores = (
            F.softmax(outputs.logits, dim=-1) if outputs.logits.size(-1) != 1 else outputs.logits
        )
        return [
            [Prediction(label=self.labels_map[i], score=float(score)) for i, score in enumerate(row)]
            for row in scores.float().cpu().numpy()
        ]


class Registry:
    _registry: dict[str, Type[BaseQualityClassifier]] = {}
    _logger = get_logger("ModelRegistry")

    def __new__(cls, *args, **kwargs):
        return cls

    @classmethod
    def add(cls, classifier_name: str):
        def _add(classifier: Type[BaseQualityClassifier]):
            cls._registry[classifier_name] = classifier
        return _add

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseQualityClassifier:
        if model_name not in cls._registry:
            cls._logger.warning(f"Classifier {model_name} not found in registry; using default classifier")
            return BaseQualityClassifier(model_name=model_name, **kwargs)
        else:
            return cls._registry[model_name](model_name=model_name, **kwargs)


@Registry.add("HuggingFaceFW/fineweb-edu-classifier")
class FineWebEduClassifier(BaseQualityClassifier):
    pass
