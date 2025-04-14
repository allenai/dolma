"""

Filters.

@kylel, @soldni

"""

from typing import Iterable, List, Tuple

from tokenizers import normalizers, pre_tokenizers
import math

from ..core.data_types import TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry
from ..core.utils import split_words


def log_cap(x: float, cap: float = 1e-38) -> float:
    return math.log(max(x, 1e-38))


@TaggerRegistry.add("dclm-oh-eli5")
class DclmQualityClassifier(BaseFastTextTagger):
    MODEL_PATH = "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_text(self, text: str) -> float:
        # Clean the input text by joining all lines into a single string
        text = " ".join(text.strip().splitlines())
        pred = self.classifier.predict(text)

        # Extract the predicted label and its probability
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]
        probability_score = pred_prob[0]

        # If the predicted label is 'CC', adjust the probability of it being 'Wikipedia'
        if pred_label == "__label__cc":
            probability_score = 1 - probability_score
            
        return probability_score

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        # Note: This slice should always be the entire document
        return [Prediction(label="score", score=self.predict_text(text_slice.text))]



@TaggerRegistry.add("dclm-oh-eli5-log")
class DclmQualityClassifierLog(DclmQualityClassifier):
    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.predict_text(text_slice.text)
        return [Prediction(label="score", score=log_cap(pred))]


@TaggerRegistry.add("dclm-oh-eli5-log-chunk1k")
class DclmQualityClassifierLogChunk(DclmQualityClassifier):
    CHUNK_SIZE = 1000
    MIN_CHUNK_SIZE = 250

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        # Average number of tokens per sentence in DCLM RefinedWeb: ~800 tokens = ~600 words
        # Let's put a slight bias towards longer documents by using a chunk size of 1000 words
        # Chunk the text into chunks of at least 1000 words, 
        # only the last chunk is always between 250 - 1250 words to avoid an unnecessarily chunk
        words = split_words(text_slice.text)
        boundaries = list(range(0, len(words) - self.MIN_CHUNK_SIZE, self.CHUNK_SIZE)) + [len(words)]
        chunks = [
            TextSlice(text_slice.text, words[boundaries[i]].start, words[boundaries[i+1]-1].end)
            for i in range(len(boundaries) - 1)
        ]

        num_words = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
        total_num_words = sum(num_words)
        weights = [n / total_num_words for n in num_words]

        # Calculate the probability of the document being high quality
        # P_{\text{doc}} = 1 - \prod_{i=1}^N (1 - p_i)
        # where p_i is the probability of the i-th chunk being high quality
        # and N is the number of chunks
        # Use log-probabilities to avoid underflow
        doc_pred_noisy_or = 1 - sum(
            (len(weights) * weight) * log_cap(1 - self.predict_text(chunk.text))
            for chunk, weight in zip(chunks, weights)
        )

        doc_pred_avg = sum(
            weight * self.predict_text(chunk.text)
            for chunk, weight in zip(chunks, weights)
        )

        return [
            Prediction(label="noisy_or", score=log_cap(doc_pred_noisy_or)),
            Prediction(label="average", score=log_cap(doc_pred_avg)),
        ]



@TaggerRegistry.add("dclm-oh-eli5-log-chunk2k")
class DclmQualityClassifierLogChunk2k(DclmQualityClassifierLogChunk):
    CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 500


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
