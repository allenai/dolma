from dataclasses import dataclass

from ...cli import field
from ..config import BaseModelConfig, BaseTrainerConfig


@dataclass
class FastTextModelConfig(BaseModelConfig):
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


@dataclass
class FastTextTrainerConfig(BaseTrainerConfig):
    model: FastTextModelConfig = field(help="Model configuration", default=FastTextModelConfig())
