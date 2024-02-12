from dataclasses import dataclass
from typing import List, Optional

from ..cli import field


@dataclass
class SamplingConfig:
    train: float = field(help="Percentage of the data to sample for training", default=0)
    dev: float = field(help="Percentage of the data to sample for development", default=0)
    test: float = field(help="Percentage of the data to sample for testing", default=0)


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
    lowercase: bool = field(help="Lowercase the text before training the model", default=False)


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
