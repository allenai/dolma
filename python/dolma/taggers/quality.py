"""

Filters.

@kylel, @soldni

"""

from typing import Iterable, List, Tuple

from tokenizers import normalizers, pre_tokenizers

from ..core.data_types import TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry


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
