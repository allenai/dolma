from typing import Iterable

from dolma.core.data_types import TextSlice
from dolma.core.registry import TaggerRegistry
from dolma.taggers.models.ft import FastTextPrediction, FastTextTagger
from dolma.models.word_tokenizers import TokenizerRegistry


class BaseQualityTagger(FastTextTagger):
    MODEL_PATH: str
    MODEL_MODE: str
    TOKENIZER_MODE: str

    def __init__(self):
        assert hasattr(self, "MODEL_PATH"), f"{self.__class__} is missing MODEL_PATH"
        assert hasattr(self, "MODEL_MODE"), f"{self.__class__} is missing MODEL_MODE"
        assert hasattr(self, "TOKENIZER_MODE"), f"{self.__class__} is missing TOKENIZER_MODE"

        super().__init__(path=self.MODEL_PATH, mode=self.MODEL_MODE)
        self.word_tokenizer = TokenizerRegistry.get(self.TOKENIZER_MODE)()

    def predict_slice(self, text_slice: TextSlice) -> Iterable[FastTextPrediction]:
        text = " ".join(self.word_tokenizer(text_slice.text))
        preds = self.classifier.predict(text, k=-1)
        out = [
            FastTextPrediction(label=label.replace("__label__", ""), score=score)
            for label, score in sorted(zip(*preds), key=lambda x: x[1], reverse=True)
        ]
        return out


@TaggerRegistry.add("cc_multi_bin")
class Dolma17BinaryCommonCrawlWiki(BaseQualityTagger):
    MODEL_PATH = "https://dolma-artifacts.org/fasttext_models/dolma-1_7/cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_books_openhermes.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws"
