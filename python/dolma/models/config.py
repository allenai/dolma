from dataclasses import dataclass
from typing import List, Optional

from ..cli import field


@dataclass
class SamplingConfig:
    train: float = field(help="Percentage of the data to sample for training", default=0.8)
    dev: float = field(help="Percentage of the data to sample for development", default=0.1)
    test: float = field(help="Percentage of the data to sample for testing", default=0.1)


@dataclass
class StreamConfig:
    documents: List[str] = field(help="List of documents to use for this stream", default_factory=list)
    label: str = field(
        help=(
            "Label for this stream. Can be a selector, e.g. $.<value>, to derive the label "
            "from the document, or a fixed value."
        ),
        default="$.source",
    )
    text: str = field(
        help="Text for this stream. Must, e.g. $.<value>, to derive the text from the document", default="$.text"
    )
    sample: SamplingConfig = field(help="Sampling configuration for this stream", default=SamplingConfig())


@dataclass
class DataConfig:
    train: Optional[str] = field(help="Path to the training data", default=None)
    dev: Optional[str] = field(help="Path to the development data", default=None)
    test: Optional[str] = field(help="Path to the testing data", default=None)


@dataclass
class BaseModelConfig:
    pass


@dataclass
class BaseTrainerConfig:
    model_path: str = field(help="Path to location where to save&load the model")
    model: BaseModelConfig = field(help="Model configuration", default=BaseModelConfig())
    streams: List[StreamConfig] = field(help="List of stream configurations", default=[])
    data: Optional[DataConfig] = field(help="Data configuration", default=None)
    num_processes: int = field(help="Number of processes to use for training", default=1)
    debug: bool = field(help="Enable debug mode", default=False)


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
