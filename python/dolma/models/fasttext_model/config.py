from dataclasses import dataclass
from typing import Optional

from ...cli import field
from ..config import BaseModelConfig, BaseTrainerConfig
from ..word_tokenizers import TokenizerRegistry


@dataclass
class FastTextQuantizerConfig(BaseModelConfig):
    features_cutoff: int = field(help="Number of words and ngrams (features) to retain.", default=0)
    retrain: bool = field(help="Whether to finetune embeddings if a cutoff is applied.", default=True)
    epochs: int = field(help="Number of epochs", default=1)
    learning_rate: float = field(help="Learning rate", default=0.1)
    subvector_size: int = field(help="Size of each sub-vector", default=2)
    quantize_norm: bool = field(help="Whether to quantize norm separately", default=True)
    quantize_classifier: bool = field(help="Whether to quantize classifier", default=False)
    model_path: Optional[str] = field(help="Path to paths to quantize vectors", default=None)


@dataclass
class FastTextAutotuneConfig:
    enabled: bool = field(help="Whether to autotune. Off by default.", default=False)
    metric: str = field(help="Metric to use for autotuning", default="f1")
    number_predictions: int = field(help="Number of prediction to autotune for", default=1)
    duration: int = field(help="Duration of autotuning in seconds", default=300)
    model_size: str = field(help="Constrained size to use for autotuning", default="")


@dataclass
class FastTextSupervisedModelConfig(BaseModelConfig):
    learning_rate: float = field(help="Learning rate", default=0.1)
    word_vector_size: int = field(help="Size of the word vectors", default=100)
    context_window_size: int = field(help="Size of the context window", default=5)
    epochs: int = field(help="Number of epochs", default=5)
    min_word_occurrences: int = field(help="Minimum word occurrences", default=1)
    min_label_occurrences: int = field(help="Minimum label occurrences", default=1)
    min_char_ngram_length: int = field(help="Minimum character ngram length", default=0)
    max_char_ngram_length: int = field(help="Maximum character ngram length", default=0)
    negatives_samples: int = field(help="Number of negative samples", default=5)
    max_word_ngram_length: int = field(help="Maximum word ngram length", default=1)
    loss_function: str = field(help="Loss function to use. Can be one of ns, hs, softmax, ova", default="softmax")
    number_of_buckets: int = field(help="Number of buckets", default=2_000_000)
    learning_rate_update_rate: int = field(help="Learning rate update rate", default=100)
    sampling_threshold: float = field(help="Sampling threshold", default=0.0001)
    pretrained_vectors: str = field(help="Path to pretrained vectors", default="")
    autotune: FastTextAutotuneConfig = field(help="Autotune configuration", default=FastTextAutotuneConfig())


@dataclass
class FastTextSupervisedTrainerConfig(BaseTrainerConfig):
    model: FastTextSupervisedModelConfig = field(
        help="Model configuration", default=FastTextSupervisedModelConfig()
    )
    word_tokenizer: str = field(
        help=f"Tokenizer used to extract words; must be one of {TokenizerRegistry.s()}", default="punct"
    )


@dataclass
class FastTextUnsupervisedModelConfig(BaseModelConfig):
    algorithm: str = field(help="Type fasttext model (cbow, skipgram)", default="skipgram")
    learning_rate: float = field(help="Learning rate", default=0.05)
    word_vector_size: int = field(help="Size of the word vectors", default=100)
    context_window_size: int = field(help="Size of the context window", default=5)
    epochs: int = field(help="Number of epochs", default=5)
    min_word_occurrences: int = field(help="Minimum word occurrences", default=5)
    min_char_ngram_length: int = field(help="Minimum character ngram length", default=3)
    max_char_ngram_length: int = field(help="Maximum character ngram length", default=6)
    negatives_samples: int = field(help="Number of negative samples", default=5)
    max_word_ngram_length: int = field(help="Maximum word ngram length", default=1)
    loss_function: str = field(help="Loss function to use. Can be one of ns, hs, softmax, ova", default="ns")
    number_of_buckets: int = field(help="Number of buckets", default=2_000_000)
    learning_rate_update_rate: int = field(help="Learning rate update rate", default=100)
    sampling_threshold: float = field(help="Sampling threshold", default=0.0001)


@dataclass
class FastTextUnsupervisedTrainerConfig(BaseTrainerConfig):
    model: FastTextUnsupervisedModelConfig = field(
        help="Model configuration", default=FastTextUnsupervisedModelConfig()
    )
    word_tokenizer: str = field(
        help=f"Tokenizer used to extract words; must be one of {TokenizerRegistry.s()}", default="punct"
    )


@dataclass
class FastTextQuantizerTrainerConfig(BaseTrainerConfig):
    model: FastTextQuantizerConfig = field(help="Model configuration", default=FastTextQuantizerConfig())
    word_tokenizer: str = field(
        help=f"Tokenizer used to extract words; must be one of {TokenizerRegistry.s()}", default="punct"
    )
