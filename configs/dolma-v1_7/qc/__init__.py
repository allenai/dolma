from typing import Iterable

from dolma.core.data_types import TextSlice
from dolma.core.registry import TaggerRegistry
from dolma.taggers.models.ft import FastTextPrediction, FastTextTagger
from dolma.models.word_tokenizers import TokenizerRegistry


@TaggerRegistry.add("cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_300dSW")
class FT1(FastTextTagger):
    MODEL_PATH = "/home/ubuntu/fasttext_models/cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_300dSW.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws"

    def __init__(self):
        super().__init__(path=self.MODEL_PATH, mode=self.MODEL_MODE)
        self.word_tokenizer = TokenizerRegistry.get(self.TOKENIZER_MODE)()

    def predict_slice(self, text_slice: TextSlice) -> Iterable[FastTextPrediction]:
        text = " ".join(self.word_tokenizer(text_slice.text))
        preds = self.classifier.predict(text, k=-1)
        out = [
            FastTextPrediction(label=label.replace("__label__", ""), score=score)
            for label, score in sorted(zip(*preds), key=lambda x: x[1], reverse=True)
        ]
        # print(text)
        # print('-----------------')
        # print(out)
        # input('=================\n\n')
        return out


@TaggerRegistry.add("cc_wiki_wikiref_owt2_pes2o_books")
class FT2(FT1):
    MODEL_PATH = "/home/ubuntu/fasttext_models/cc_wiki_wikiref_owt2_pes2o_books.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws_lower"


@TaggerRegistry.add("cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_binary_ws_lower")
class FT3(FT1):
    MODEL_PATH = "/home/ubuntu/fasttext_models/cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_binary_ws_lower.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws_lower"


@TaggerRegistry.add("cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_binary_ws")
class FT4(FT1):
    MODEL_PATH = "/home/ubuntu/fasttext_models/cc_wiki_wikiref_sw_pes2o_adult_fakenews_math_binary_ws.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws"


@TaggerRegistry.add("q2_6m-cc_w_wr_sw_s2_nsfw_fake_math_bin_ws_300_2Ms")
class FT5(FT1):
    MODEL_PATH = "/home/ubuntu/fasttext_models/q2_6m-cc_w_wr_sw_s2_nsfw_fake_math_bin_ws_300_2Ms.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws"


@TaggerRegistry.add("rw_hrms_bin_v2")
class FT6(FT1):
    MODEL_PATH = "/home/ubuntu/fasttext_models/rw_hrms_bin.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws"

    def predict_slice(self, text_slice: TextSlice) -> Iterable[FastTextPrediction]:
        out = super().predict_slice(text_slice)
        return [label for label in out if label.label in {"lq", "hq"}]


@TaggerRegistry.add("suchin_whose_quality_v2")
class FT7(FT1):
    MODEL_PATH = "/home/ubuntu/fasttext_models/suchin_whose_quality_v2.bin"
    MODEL_MODE = "document"
    TOKENIZER_MODE = "ws"
