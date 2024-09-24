"""

Filters.

@kylel, @soldni

"""

from typing import Iterable, List, Tuple

from tokenizers import normalizers, pre_tokenizers

from ..core.data_types import TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry


@TaggerRegistry.add("dclm-oh-eli5")
class DclmQualityClassifier(BaseFastTextTagger):
    MODEL_PATH = "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        # Note: This slice should always be the entire document

        # Clean the input text by joining all lines into a single string
        text = " ".join(text_slice.doc.strip().splitlines())
        pred = self.classifier.predict(text)

        # Extract the predicted label and its probability
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]
        probability_score = pred_prob[0]

        # If the predicted label is 'CC', adjust the probability of it being 'Wikipedia'
        if pred_label == "__label__cc":
            probability_score = 1 - probability_score

        label = pred_label.replace("__label__", "").replace("cc", "score").replace("hq", "score")

        return [Prediction(label=label, score=probability_score)]


@TaggerRegistry.add("dolma17-quality")
class Dolma17QualityClassifier(BaseFastTextTagger):
    MODEL_PATH = "https://dolma-artifacts.org/fasttext_models/dolma-1_7/cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_books_openhermes.bin"  # noqa: E501

    def __init__(self):
        self._normer = normalizers.Strip()
        self._splitter = pre_tokenizers.WhitespaceSplit()
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def preprocess(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Tokenize the text"""
        normalized_text = self._normer.normalize_str(text)
        tokens = self._splitter.pre_tokenize_str(normalized_text)
        return tokens

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        tokens, _ = zip(*self.preprocess(text_slice.text))
        preds = self.classifier.predict(" ".join(tokens), k=-1)
        out = [
            Prediction(label=label.replace("__label__", ""), score=score)
            for label, score in sorted(zip(*preds), key=lambda x: x[1], reverse=True)
        ]
        return out
