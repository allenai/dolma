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

import random
import numpy as np
def log_cap(x: float, cap: float = 1e-38) -> float:
    return math.log(max(x, 1e-38))


@TaggerRegistry.add("dclm-oh-eli5")
class DclmQualityClassifier(BaseFastTextTagger):
    MODEL_PATH = "/home/ec2-user/dolma/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"

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


def float_or_zero(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


@TaggerRegistry.add("fineweb-edu-fasttext-gt2")
class FinewebEduBinaryClassifier(BaseFastTextTagger):
    MODEL_PATH = "/home/ec2-user/dolma/fineweb_edu_gt2_bigram_200k.bin"
    NUM_CLASSES = 2

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_text(self, text: str) -> float:
        # Clean the input text by joining all lines into a single string
        text = " ".join(text.strip().splitlines())
        pred = self.classifier.predict(text, k=self.NUM_CLASSES)

        # Extract the predicted label and its probability
        (labels, probs) = pred
        
        label_score = np.array([float_or_zero(label.removeprefix("__label__")) for label in labels])
            
        mean_prediction = np.dot(label_score, probs).item()
        return mean_prediction

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        return [Prediction(label="score", score=self.predict_text(text_slice.text))]



@TaggerRegistry.add("weborganizer-edu")
class WebOrganizerEduClassifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/weborganizer_0.25edu_bigram_200k.bin"


@TaggerRegistry.add("weborganizer-dclm")
class WebOrganizerDclmClassifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/weborganizer_0.25fasttext_bigram_200k.bin"

@TaggerRegistry.add("weborganizer-edu-regmix")
class WebOrganizerEduRegmixClassifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/weborganizer_0.25edu-regmix_bigram_200k.bin"


@TaggerRegistry.add("weborganizer-dclm-regmix")
class WebOrganizerDclmRegmixClassifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/weborganizer_0.25fasttext-regmix_bigram_200k.bin"


@TaggerRegistry.add("luca-fineweb2")
class LucaFineweb2Classifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/whitespace-fineweb2_lr05_ng3_n3M6.bin"

@TaggerRegistry.add("oh-uc-wc-eli5-edu2")
class OhUcWcEli5Edu2Classifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/oh_uc_wc_eli5_edu2_fasttext_model_bigram_200k.bin"

@TaggerRegistry.add("oh-uc-wc-eli5")
class OhUcWcEli5Classifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/oh_uc_wc_eli5_fasttext_model_bigram_200k.bin"

@TaggerRegistry.add("fineweb-edu-fasttext-5way")
class FinewebEdu5WayClassifier(FinewebEduBinaryClassifier):
    MODEL_PATH = "/home/ec2-user/dolma/fineweb_edu_5way_bigram_200k.bin"
    NUM_CLASSES = 6
    
@TaggerRegistry.add("dclm-oh-eli5-log")
class DclmQualityClassifierLog(DclmQualityClassifier):
    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.predict_text(text_slice.text)
        return [Prediction(label="score", score=log_cap(pred))]


@TaggerRegistry.add("dclm-oh-eli5-log-single500")
class DclmQualityClassifierLog(DclmQualityClassifier):
    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:

        words = split_words(text_slice.text)
        # Define chunk size and create boundaries
        chunk_size = 500
        if len(words) <= chunk_size:
            # If text is shorter than 500 words, use the entire text
            chunk = text_slice
        else:
            # Select a random starting point for a 500-word chunk
            max_start = len(words) - chunk_size
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + chunk_size
            
            # Create a single chunk with the randomly selected 500 words
            chunk = TextSlice(text_slice.text, words[start_idx].start, words[end_idx-1].end)

        pred = self.predict_text(chunk.text)
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
        boundaries = list(range(0, max(len(words) - self.MIN_CHUNK_SIZE, 1), self.CHUNK_SIZE)) + [len(words)]
        
        chunks = [
            TextSlice(text_slice.text, words[boundaries[i]].start, words[boundaries[i+1]-1].end)
            for i in range(len(boundaries) - 1)
        ]
        # print(len(chunks), len(words))

        num_words = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
        total_num_words = sum(num_words)
        weights = [n / total_num_words for n in num_words]

        chunk_scores = [self.predict_text(chunk.text) for chunk in chunks]

        # Calculate the probability of the document being high quality
        # P_{\text{doc}} = 1 - \prod_{i=1}^N (1 - p_i)
        # where p_i is the probability of the i-th chunk being high quality
        # and N is the number of chunks
        # Use log-probabilities to avoid underflow
        doc_pred_noisy_or = 1 - math.exp(len(weights) * sum(
            weight * log_cap(1 - chunk_score)
            for chunk_score, weight in zip(chunk_scores, weights)
        ))

        doc_pred_avg = sum(
            weight * chunk_score
            for chunk_score, weight in zip(chunk_scores, weights)
        )

        return [
            Prediction(label="noisy_or", score=log_cap(doc_pred_noisy_or)),
            Prediction(label="average", score=log_cap(doc_pred_avg)),
            Prediction(label="n_chunks", score=len(chunk_scores)),
            Prediction(label="min_score", score=log_cap(min(chunk_scores))),
            Prediction(label="max_score", score=log_cap(max(chunk_scores))),
        ]



@TaggerRegistry.add("dclm-oh-eli5-log-chunk500")
class DclmQualityClassifierLogChunk500(DclmQualityClassifierLogChunk):
    CHUNK_SIZE = 500
    MIN_CHUNK_SIZE = 250


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
