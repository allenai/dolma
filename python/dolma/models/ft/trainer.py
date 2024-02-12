import re
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import smart_open
from necessary import necessary

from ..data import FastTextDataConverter
from ..trainer import BaseTrainer
from .config import FastTextTrainerConfig

with necessary(("fasttext", "0.9.2"), soft=True) as FASTTEXT_AVAILABLE:
    if TYPE_CHECKING or FASTTEXT_AVAILABLE:
        import fasttext
        from fasttext.FastText import _FastText as FastTextModel

with necessary("sklearn", soft=True) as SKLEARN_AVAILABLE:
    if TYPE_CHECKING or SKLEARN_AVAILABLE:
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import MultiLabelBinarizer


class FastTextTrainer(BaseTrainer):
    def __init__(self, config: FastTextTrainerConfig, cache_dir: Optional[str] = None):
        if not FASTTEXT_AVAILABLE:
            raise ImportError("fasttext is not available. Install it using `pip install fasttext-wheel`")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available. Install it using `pip install scikit-learn")

        super().__init__(config=config, cache_dir=cache_dir)

    @property
    def data_factory_cls(self):
        return FastTextDataConverter

    def fit(self, data_path: str, save_path: str):
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
            verbose=2 if self.config.debug else 1,
            pretrainedVectors=self.config.model.pretrained_vectors or "",
        )
        model.save_model(save_path)
        return model

    def get_labels_and_text(self, path: str) -> Generator[Tuple[List[str], str], None, None]:
        with smart_open.open(path, "rt") as f:
            for ln in f:
                # because we might have more than one label, we need to find the end of the labels
                # and then split the labels and the text
                labels_match = re.match(r"^(__label__\S+\s+)+", ln)

                if labels_match is None:
                    raise ValueError(f"{path} not the fasttext format, no labels found!")

                labels = labels_match.group(0).strip().split(" ")
                text = ln[labels_match.end() :].strip()
                yield labels, text

    def predict(self, data_path: str, load_path: str):
        # load the model
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
