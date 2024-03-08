import re
from contextlib import ExitStack
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import smart_open
from necessary import necessary
from tqdm import tqdm

from dolma.core.loggers import get_logger

from ...core.paths import cached_path
from ..data import FastTextDataConverter, FastTextUnsupervisedDataConverter
from ..trainer import BaseTrainer
from .config import (
    FastTextQuantizerTrainerConfig,
    FastTextSupervisedTrainerConfig,
    FastTextUnsupervisedTrainerConfig,
)

with necessary(("fasttext", "0.9.2"), soft=True) as FASTTEXT_AVAILABLE:
    if TYPE_CHECKING or FASTTEXT_AVAILABLE:
        import fasttext
        from fasttext.FastText import _FastText as FastTextModel

with necessary("sklearn", soft=True) as SKLEARN_AVAILABLE:
    if TYPE_CHECKING or SKLEARN_AVAILABLE:
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import MultiLabelBinarizer


LOGGER = get_logger(__name__)


class FastTextTrainer(BaseTrainer):
    def __init__(self, config: FastTextSupervisedTrainerConfig):
        if not FASTTEXT_AVAILABLE:
            raise ImportError("fasttext is not available. Install it using `pip install fasttext-wheel`")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available. Install it using `pip install scikit-learn")

        super().__init__(config=config)

    @property
    def data_factory_cls(self):
        return FastTextDataConverter

    def fit(self, data_path: str, save_path: str, validation_path: Optional[str] = None):
        pretrained_vectors = (
            cached_path(self.config.model.pretrained_vectors)
            if self.config.model.pretrained_vectors is not None
            else ""
        )

        autotune_config = {}
        if self.config.model.autotune.enabled and validation_path is not None:
            autotune_config = {
                "autotuneValidationFile": cached_path(validation_path),
                "autotuneMetric": self.config.model.autotune.metric,
                "autotunePredictions": self.config.model.autotune.number_predictions,
                "autotuneDuration": self.config.model.autotune.duration,
                "autotuneModelSize": self.config.model.autotune.model_size,
            }

        model = fasttext.train_supervised(
            input=data_path,
            lr=self.config.model.learning_rate,
            dim=self.config.model.word_vector_size,
            ws=self.config.model.context_window_size,
            epoch=self.config.model.epochs,
            minCount=self.config.model.min_word_occurrences,
            minCountLabel=self.config.model.min_label_occurrences,
            minn=self.config.model.min_char_ngram_length,
            maxn=self.config.model.max_char_ngram_length,
            neg=self.config.model.negatives_samples,
            wordNgrams=self.config.model.max_word_ngram_length,
            loss=self.config.model.loss_function,
            bucket=self.config.model.number_of_buckets,
            thread=1 if self.config.debug else self.config.num_processes,
            lrUpdateRate=self.config.model.learning_rate_update_rate,
            t=self.config.model.sampling_threshold,
            label="__label__",
            verbose=2,
            pretrainedVectors=pretrained_vectors,
            **autotune_config,
        )
        model.save_model(save_path)
        return model

    def get_labels_and_text(self, path: str) -> Generator[Tuple[List[str], str], None, None]:
        with ExitStack() as stack:
            # handy progress bar
            path_short = path if len(path) < 30 else f"{path[:10]}...{path[-10:]}"
            pbar = stack.enter_context(tqdm(desc=f"Reading {path_short}", unit="lines", unit_scale=True))

            # this is the file I'm reading from
            f = stack.enter_context(smart_open.open(path, "rt"))

            # iterate over the lines
            for ln in f:
                # because we might have more than one label, we need to find the end of the labels
                # and then split the labels and the text
                labels_match = re.match(r"^(__label__\S+\s+)+", ln)

                if labels_match is None:
                    raise ValueError(f"{path} not the fasttext format, no labels found!")

                labels = labels_match.group(0).strip().split(" ")
                text = ln[labels_match.end() :].strip()
                yield labels, text
                pbar.update(1)

    def predict(self, data_path: str, load_path: str):
        # load the model
        LOGGER.info(f"Loading model from {load_path}")
        model = FastTextModel(load_path)

        # run the models
        y_true, y_pred = [], []
        for labels, text in self.get_labels_and_text(data_path):
            pred, _ = model.predict(text, k=len(labels))
            y_true.append(labels)
            y_pred.append(pred)

        # have to binarize the labels
        binarizer = MultiLabelBinarizer()
        y_true = binarizer.fit_transform(y_true)
        y_pred = binarizer.transform(y_pred)

        # calculate metrics, and print the report
        report = classification_report(y_true=y_true, y_pred=y_pred, target_names=binarizer.classes_)
        print(report)

        return y_true, y_pred


class FastTextUnsupervisedTrainer(FastTextTrainer):
    def __init__(self, config: FastTextUnsupervisedTrainerConfig):
        super().__init__(config=config)  # type: ignore[arg-type]

    @property
    def data_factory_cls(self):
        return FastTextUnsupervisedDataConverter

    def fit(self, data_path: str, save_path: str, validation_path: Optional[str] = None):
        model = fasttext.train_unsupervised(
            input=data_path,
            model=self.config.model.algorithm,
            lr=self.config.model.learning_rate,
            dim=self.config.model.word_vector_size,
            ws=self.config.model.context_window_size,
            epoch=self.config.model.epochs,
            minCount=self.config.model.min_word_occurrences,
            minn=self.config.model.min_char_ngram_length,
            maxn=self.config.model.max_char_ngram_length,
            neg=self.config.model.negatives_samples,
            wordNgrams=self.config.model.max_word_ngram_length,
            loss=self.config.model.loss_function,
            bucket=self.config.model.number_of_buckets,
            thread=1 if self.config.debug else self.config.num_processes,
            lrUpdateRate=self.config.model.learning_rate_update_rate,
            t=self.config.model.sampling_threshold,
            verbose=2,
        )
        model.save_model(save_path)
        return model


class FastTextQuantizerTrainer(FastTextTrainer):
    def __init__(self, config: FastTextQuantizerTrainerConfig):
        super().__init__(config=config)  # type: ignore[arg-type]

    @property
    def data_factory_cls(self):
        return FastTextDataConverter

    def fit(self, data_path: str, save_path: str, validation_path: Optional[str] = None):
        if self.config.model.model_path is None:
            LOGGER.warning("model_path is not provided, using save_path")
        model_path = self.config.model.model_path or save_path
        model = FastTextModel(model_path)
        model.quantize(
            input=data_path,
            cutoff=self.config.model.features_cutoff,
            retrain=self.config.model.retrain,
            epoch=self.config.model.epochs,
            lr=self.config.model.learning_rate,
            thread=1 if self.config.debug else self.config.num_processes,
            verbose=2,
            dsub=self.config.model.subvector_size,
            qnorm=self.config.model.quantize_norm,
        )
        model.save_model(save_path)
        return model
