from functools import partial
from typing import NamedTuple, Type

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .loggers import get_logger
from .utils import get_local_gpu_rank, sanitize_model_name


class Prediction(NamedTuple):
    label: str
    score: float


class BaseQualityClassifier:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    input_template: str = ".text"
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        compile: bool = False,
        trust_remote_code: bool = False,
    ):
        self.model = self._make_model(
            model_name=model_name,
            device=device,
            dtype=dtype,
            compile=compile,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if len(self.model.config.id2label) > 1:
            label_name_fn = lambda label: f"{sanitize_model_name(model_name)}_{sanitize_model_name(label)}"
        else:
            label_name_fn = lambda label: sanitize_model_name(model_name)

        self.labels_map = {
            id_: label_name_fn(label)
            for id_, label in self.model.config.id2label.items()
        }

    def _make_model(
        self,
        model_name: str,
        device: str,
        dtype: str,
        compile: bool,
        trust_remote_code: bool,
    ) -> PreTrainedModel:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=trust_remote_code,
        )
        model = model.to(torch.device(device))

        if compile:
            model = torch.compile(model)  # pyright: ignore

        model.eval()  # pyright: ignore

        return model  # pyright: ignore

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
    def add(cls, classifier_name: str ):
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


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    @property
    def device(self):
        return self.model.device

    def forward(self, input_ids, attention_mask, **kwargs):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return SequenceClassifierOutput(logits=outputs[:, 0, :])

#Alexw classifier
@Registry.add("data-delve/gte-base-en-v1.5_topic-v3.8_url1")
class DataDelveClassifier(BaseQualityClassifier):
    input_template = ".metadata.url\n.text"
    pass


@Registry.add("nvidia/quality-classifier-deberta")
class DebertaQualityClassifier(BaseQualityClassifier):
    def _make_model(
        self,
        model_name: str,
        device: str,
        dtype: str,
        compile: bool,
        trust_remote_code: bool,
    ) -> PreTrainedModel:
        model = QualityModel.from_pretrained(model_name)
        model = model.to(getattr(torch, dtype))
        model = model.to(torch.device(device))

        if compile:
            model = torch.compile(model)  # pyright: ignore

        model.eval()  # pyright: ignore

        # for some reason the config is not loaded automatically; need to set it manually
        model.config = AutoConfig.from_pretrained(model_name)  # pyright: ignore

        return model  # pyright: ignore
