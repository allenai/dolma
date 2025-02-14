from dataclasses import dataclass
from typing import TYPE_CHECKING

from necessary import necessary

from . import BaseCli, DolmaConfigError, field, print_config
from ..core.loggers import get_logger
from ..core.utils import is_valid_jq_expr


with necessary("dolma-classifiers", soft=True) as DOLMA_CLASSIFIERS_AVAILABLE:
    if DOLMA_CLASSIFIERS_AVAILABLE or TYPE_CHECKING:
        from dolma_classifiers import run_inference


@dataclass
class WandbConfig:
    project: str | None = field(default=None, help="Weights & Biases project name")
    entity: str | None = field(default=None, help="Weights & Biases entity name")
    name: str | None = field(default=None, help="Name of this run")


@dataclass
class ModelConfig:
    name_or_path: str | None = field(default=None, help="Hugging Face model name or path. Required.")
    batch_size: int = field(default=32, help="Batch size for processing (default: 1)")
    max_length: int | None = field(default=None, help="Maximum sequence length for tokenization (default: None)")
    compile: bool = field(default=False, help="Compile the model using torch.compile")
    dtype: str = field(default="float16", help="Data type for model")
    device: str = field(default="cuda", help="Device to run the model on")


@dataclass
class DataConfig:
    text_key: str = field(
        default=".text",
        help="JQ key to extract text from documents",
    )
    id_key: str = field(
        default=".id",
        help="JQ key to extract id from documents",
    )
    attribute_suffix: str | None = field(
        default=None,
        help="Optional suffix for attribute keys",
    )
    prefetch_factor: int = field(
        default=2,
        help="Prefetch factor for DataLoader",
    )


@dataclass
class DolmaClassifierInferenceConfig:
    documents: list[str] = field(
        default=[],
        help="One or more document paths to process; Can be either local or S3 paths. Globs are supported.",
    )
    destination: list[str] | None = field(
        default=None,
        nargs="*",
        help=(
            "Destination paths to save the outputs; should match the number of document paths. "
            "If not provided, destination will be derived from the document path."
        ),
    )
    model: ModelConfig = field(
        default=ModelConfig(),
        help="Hugging Face model configuration; At least `model.name` is required.",
    )
    wandb: WandbConfig = field(
        default=WandbConfig(),
        help="Weights & Biases configuration; if no options are provided, Weights & Biases will not be used."
    )
    data: DataConfig = field(
        default=DataConfig(),
        help="Data configuration for processing documents.",
    )
    skip_existing: bool = field(
        default=False,
        help="Whether to ignore existing outputs and re-run the classifier on all files. Default is False.",
    )
    debug: bool = field(
        default=False,
        help="Whether to run in debug mode.",
    )
    processes: int = field(
        default=1,
        help="Number of parallel processes to use.",
    )
    log_every: int = field(
        default=10_000,
        help="Log every n documents",
    )
    dryrun: bool = field(
        default=False,
        help="If true, only print the configuration and exit without running the classifier.",
    )


class DolmaClassifierInferenceCli(BaseCli):
    CONFIG = DolmaClassifierInferenceConfig
    DESCRIPTION = "Run a HuggingFace model on documents for inference."

    @classmethod
    def run(cls, parsed_config: DolmaClassifierInferenceConfig):
        if not DOLMA_CLASSIFIERS_AVAILABLE:
            raise DolmaConfigError(
                "dolma-classifiers is not available; please install it: 'pip install dolma-classifiers'"
            )

        documents = [str(doc) for doc in parsed_config.documents]
        destinations = [str(dest) for dest in parsed_config.destination] if parsed_config.destination else None

        if len(documents) == 0:
            raise DolmaConfigError("At least one documents path must be specified")

        if destinations is not None and len(destinations) != len(documents):
            raise DolmaConfigError("Number of documents and destinations must match")
        elif destinations is None and any("/documents/" not in doc for doc in documents):
            raise DolmaConfigError("Destination must be provided if documents do not contain '/documents/'")

        if parsed_config.model.name_or_path is None:
            raise DolmaConfigError("Hugging Face model name or path is required")

        if parsed_config.model.max_length is not None and parsed_config.model.max_length <= 0:
            raise DolmaConfigError("max_length must be > 0")

        if parsed_config.model.batch_size <= 0:
            raise DolmaConfigError("batch_size must be > 0")

        model_max_length = (
            int(parsed_config.model.max_length) if parsed_config.model.max_length is not None else None
        )

        if not is_valid_jq_expr(parsed_config.data.text_key):
            raise DolmaConfigError(f"Invalid text_key: {parsed_config.data.text_key}")

        if not is_valid_jq_expr(parsed_config.data.id_key):
            raise DolmaConfigError(f"Invalid id_key: {parsed_config.data.id_key}")

        if parsed_config.data.prefetch_factor < 0:
            raise DolmaConfigError("prefetch_factor must be >= 0")

        data_attribute_suffix = (
            str(parsed_config.data.attribute_suffix) if parsed_config.data.attribute_suffix else None
        )

        print_config(parsed_config)
        if parsed_config.dryrun:
            get_logger("dolma-classifier").info("Exiting due to dryrun.")
            return

        run_inference(
            documents=documents,
            destinations=destinations,
            model_name_or_path=str(parsed_config.model.name_or_path),
            model_batch_size=int(parsed_config.model.batch_size),
            model_max_length=model_max_length,
            model_compile=bool(parsed_config.model.compile),
            model_dtype=parsed_config.model.dtype,
            model_device=parsed_config.model.device,
            data_text_key=str(parsed_config.data.text_key),
            data_id_key=str(parsed_config.data.id_key),
            data_attribute_suffix=data_attribute_suffix,
            data_prefetch_factor=int(parsed_config.data.prefetch_factor),
            wandb_project=str(parsed_config.wandb.project) if parsed_config.wandb.project else None,
            wandb_entity=str(parsed_config.wandb.entity) if parsed_config.wandb.entity else None,
            wandb_name=str(parsed_config.wandb.name) if parsed_config.wandb.name else None,
            skip_existing=bool(parsed_config.skip_existing),
            debug=bool(parsed_config.debug),
            processes=int(parsed_config.processes),
        )
