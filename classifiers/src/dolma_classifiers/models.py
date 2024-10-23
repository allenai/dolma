from functools import partial
import torch
from typing import Type

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from .utils import get_local_gpu_rank
from .loggers import get_logger


class BaseQualityClassifier:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, model_name: str, device: str, dtype: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=True,
        ).to(torch.device(device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def score(self, **batch: torch.Tensor) -> list[float]:
        outputs = self.model(**batch)
        return outputs.logits.squeeze(-1).float().detach().tolist()


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
        return cls._registry[model_name](**kwargs)
