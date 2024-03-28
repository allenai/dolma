"""

Filters.

@kylel, @soldni

"""

from typing import Iterable

from ..core.data_types import TextSlice
from ..core.registry import TaggerRegistry
from .models.ft import FastTextPrediction, FastTextTagger


@TaggerRegistry.add("jigsaw_hatespeech_document_v2")
class FastTextJigsawHatespeechDocumentTagger(FastTextTagger):
    MODEL_PATH = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin"  # noqa: E501
    MODEL_MODE = "document"

    def __init__(self):
        super().__init__(path=self.MODEL_PATH, mode=self.MODEL_MODE)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[FastTextPrediction]:
        labels, probs = self.classifier.predict(text_slice.text.replace("\n", " ").strip(), k=-1)
        label_index = 1 if "non" in labels[0] else 0  # pyright: ignore
        return (
            FastTextPrediction(label=labels[label_index], score=probs[label_index]),
            FastTextPrediction(label=labels[1 - label_index], score=probs[1 - label_index]),
        )


@TaggerRegistry.add("jigsaw_hatespeech_sentence_v2")
class FastTextJigsawHatespeechSentenceTagger(FastTextJigsawHatespeechDocumentTagger):
    MODEL_MODE = "sentence"


@TaggerRegistry.add("jigsaw_nsfw_document_v1")
class FastTextJigsawNsfwDocumentTagger(FastTextJigsawHatespeechDocumentTagger):
    MODEL_PATH = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"  # noqa: E501


@TaggerRegistry.add("jigsaw_nsfw_sencence_v2")
class FastTextJigsawNsfwSentenceTagger(FastTextJigsawHatespeechSentenceTagger):
    MODEL_PATH = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"  # noqa: E501
