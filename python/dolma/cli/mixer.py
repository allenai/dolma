from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dolma import mixer
from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path


@dataclass
class StreamOutputConfig:
    path: str = field(help="Path where to write the mixed documents to. Required.")
    max_size_in_bytes: int = field(
        default=2 * 2**30, help="Maximum size of the output file in bytes. Defaults to 2GB."
    )
    discard_fields: List[str] = field(default=[], help="List of fields to discard from the output documents.")


@dataclass
class FilterConfig:
    include: List[str] = field(default=[], help="JSONPath expressions to include documents")
    exclude: List[str] = field(default=[], help="JSONPath expressions to exclude documents")


@dataclass
class SpanReplacementConfig:
    span: str = field(help="JSONPath expression for the span to replace")
    min_score: float = field(default=0.5, help="Minimum score for the span to be replaced")
    replacement: str = field(default="", help="Replacement for the span")


@dataclass
class StreamConfig:
    name: str = field(help="Name of the stream. Required.")
    documents: List[str] = field(default=[], help="Paths to the documents to be mixed. Required.")
    output: StreamOutputConfig = field(
        default=StreamOutputConfig(), help="Configuration for the output of the stream."
    )
    attributes: List[str] = field(default=[], help="List of attributes files to used for mixing.")
    filter: Optional[FilterConfig] = field(  # pyright: ignore
        default=None, help="Configuration for filtering documents."
    )
    span_replacement: List[SpanReplacementConfig] = field(default=[], help="Configuration for replacing spans.")


@dataclass
class MixerConfig:
    streams: List[StreamConfig] = field(default=[], help="List configurations of streams to be mixed")
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    processes: int = field(default=1, help="Number of processes to use for mixing. By default 1 process is used.")


class MixerCli(BaseCli):
    CONFIG = MixerConfig
    DESCRIPTION = "Mix documents from multiple streams."

    @classmethod
    def run(cls, parsed_config: MixerConfig):
        logger = get_logger("mixer")

        dict_config: Dict[str, Any] = {
            "work_dir": {"input": parsed_config.work_dir.input, "output": parsed_config.work_dir.output},
            "processes": parsed_config.processes,
            "streams": [],
        }

        for stream_config in parsed_config.streams:
            stream_config_dict: Dict[str, Any] = {}

            if stream_config.filter is not None:
                if not stream_config.filter.include and not stream_config.filter.exclude:
                    raise DolmaConfigError("Either `include` or `exclude` must be specified for filter")

                stream_config_dict["filter"] = {
                    "include": list(stream_config.filter.include),
                    "exclude": list(stream_config.filter.exclude),
                }

            for span_replacement in stream_config.span_replacement:
                stream_config_dict.setdefault("span_replacement", []).append(
                    {
                        "span": span_replacement.span,
                        "min_score": span_replacement.min_score,
                        "replacement": span_replacement.replacement,
                    }
                )

            if "span_replacement" not in stream_config_dict and "filter" not in stream_config_dict:
                raise DolmaConfigError("Either `filter` or `span_replacement` must be specified")

            # perform some path validation to make sure we don't call the mixer with invalid config
            total_matching_documents = 0
            for document in stream_config.documents:
                if document.count("*") > 1:
                    raise DolmaConfigError("Only one wildcard is allowed in the document path")

                current_matching_documents = sum(1 for _ in glob_path(document))
                if current_matching_documents == 0:
                    # only raise a warning if no documents are found for a single path
                    logger.warn(f"No documents found for path {document}")
                total_matching_documents += current_matching_documents

            if total_matching_documents == 0:
                # but raise an error if no documents are found for all paths
                raise DolmaConfigError(f"No documents found for the paths for {stream_config.name} config.")

            # populate the stream config dict
            stream_config_dict["name"] = stream_config.name
            stream_config_dict["documents"] = list(stream_config.documents)
            stream_config_dict["attributes"] = list(stream_config.attributes)
            stream_config_dict["output"] = {
                "path": stream_config.output.path,
                "max_size_in_bytes": stream_config.output.max_size_in_bytes,
            }

            if stream_config.output.discard_fields:
                stream_config_dict["output"]["discard_fields"] = list(stream_config.output.discard_fields)

            if len(stream_config_dict["documents"]) == 0:
                raise ValueError("No documents to mix")

            dict_config["streams"].append(stream_config_dict)

        if len(dict_config["streams"]) == 0:
            raise DolmaConfigError("No streams to mix")

        print_config(dict_config)
        return mixer(dict_config)
