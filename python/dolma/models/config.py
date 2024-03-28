from dataclasses import dataclass
from typing import List, Optional

from ..cli import field
from .word_tokenizers import TokenizerRegistry


@dataclass
class SamplingConfig:
    train: float = field(help="Percentage of the data to sample for training", default=0)
    dev: float = field(help="Percentage of the data to sample for development", default=0)
    test: float = field(help="Percentage of the data to sample for testing", default=0)


@dataclass
class StreamConfig:
    name: Optional[str] = field(help="Name of the stream", default=None)
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

    @classmethod
    def from_dir(cls, path: str) -> "DataConfig":
        return cls(train=f"{path}/train.txt", dev=f"{path}/dev.txt", test=f"{path}/test.txt")


@dataclass
class BaseModelConfig:
    pass


@dataclass
class BaseTrainerConfig:
    model_path: str = field(help="Path to location where to save or load the model")
    save_path: Optional[str] = field(
        help="Extra path to save the model; if not provided, model_path is used", default=None
    )
    model: BaseModelConfig = field(help="Model configuration", default=BaseModelConfig())
    word_tokenizer: str = field(
        help=f"Tokenizer used to extract words; must be one of {TokenizerRegistry.s()}", default="noop"
    )
    streams: List[StreamConfig] = field(help="List of stream configurations", default=[])
    data: Optional[DataConfig] = field(help="Data configuration", default=None)
    num_processes: int = field(help="Number of processes to use for training", default=1)
    cache_dir: Optional[str] = field(help="Cache directory", default=None)
    debug: bool = field(help="Enable debug mode", default=False)
    reprocess_streams: bool = field(help="Reprocess streams", default=False)
