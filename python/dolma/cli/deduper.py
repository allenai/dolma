from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import smart_open
from omegaconf import OmegaConf as om

from dolma import deduper
from dolma.cli import BaseCli, field, print_config
from dolma.cli.shared import WorkDirConfig, get_path_to_temp_file, make_workdirs
from dolma.core.errors import DolmaConfigError
from dolma.core.loggers import get_logger
from dolma.core.paths import glob_path, is_local


@dataclass
class ParagraphDedupeConfig:
    attribute_name: Optional[str] = field(help="Name of the output field in the tagger")


@dataclass
class DocumentDedupeConfig:
    attribute_name: Optional[str] = field(help="Name of the output field in the tagger")
    key: str = field(help="Name of the input field to use for deduplication, e.g. `$.metadata.url`")


@dataclass
class BloomFilterConfig:
    file: str = field(help="Path where to read/write the bloom filter file to/from. Required.")
    size_in_bytes: int = field(
        default=0,
        help=(
            "Size of the bloom filter in bytes. Either this value is provided, or both estimated_doc_count "
            "and desired_false_positive_rate."
        ),
    )
    read_only: bool = field(help="If true, the bloom filter will be read from the file and not updated. Required.")
    estimated_doc_count: int = field(
        default=0,
        help=(
            "Estimated number of documents to be added to the bloom filter. Either this value is provided, "
            "or both size_in_bytes and desired_false_positive_rate."
        ),
    )
    desired_false_positive_rate: float = field(
        default=0,
        help=(
            "Desired false positive rate. Either this value is provided, or both size_in_bytes and "
            "estimated_doc_count."
        ),
    )


@dataclass
class DedupeConfig:
    name: str = field(help="Name of the deduper. Required.")
    documents: Optional[DocumentDedupeConfig] = field(
        default=None, help="Configuration for document deduplication"
    )
    paragraphs: Optional[ParagraphDedupeConfig] = field(
        default=None, help="Configuration for paragraph deduplication"
    )
    skip_empty: Optional[bool] = field(default=False, help="If true, empty documents/paragraphs will be skipped")
    min_length: Optional[int] = field(default=0, help="Minimum length of documents/paragraphs to be deduplicated")
    min_words: Optional[int] = field(
        default=0, help="Minimum number of uniseg word units in documents/paragraphs to be deduplicated"
    )


@dataclass
class DeduperConfig:
    documents: List[str] = field(default=[], help="Paths to the documents to be deduplicated. Required.")
    work_dir: WorkDirConfig = field(default=WorkDirConfig(), help="Configuration for temporary work directories.")
    dedupe: DedupeConfig = field(help="Deduplication configuration. Required.")
    bloom_filter: BloomFilterConfig = field(help="Bloom filter configuration. Required.")
    processes: int = field(
        default=1, help="Number of processes to use for deduplication. If 1, no multiprocessing will be used."
    )
    dryrun: bool = field(
        default=False,
        help="If true, only print the configuration and exit without running the deduper.",
    )


class DeduperCli(BaseCli):
    CONFIG = DeduperConfig
    DESCRIPTION = "Deduplicate documents or paragraphs using a bloom filter."

    @classmethod
    def run(cls, parsed_config: DeduperConfig):
        logger = get_logger("tagger")

        dict_config: Dict[str, Any] = {}

        with ExitStack() as stack:
            work_dirs = stack.enter_context(make_workdirs(parsed_config.work_dir))

            # create a dedupe config to populate
            dedupe_dict_config: Dict[str, Any] = {
                "skip_empty": parsed_config.dedupe.skip_empty,
                "min_length": parsed_config.dedupe.min_length,
                "min_words": parsed_config.dedupe.min_words,
            }
            try_name = parsed_config.dedupe.name if not om.is_missing(parsed_config.dedupe, "name") else None

            if dedupe_dict_config["min_length"] < 0:
                raise ValueError("min_length must be >= 0")

            if dedupe_dict_config["min_words"] < 0:
                raise ValueError("min_words must be >= 0")

            # add either the document or paragraph dedupe config
            if not (
                om.is_missing(parsed_config.dedupe.documents, "attribute_name")
                and om.is_missing(parsed_config.dedupe.documents, "key")
            ):
                cfg = om.to_container(parsed_config.dedupe.documents)
                assert isinstance(cfg, dict), "Expected dedupe.documents to be a dict"
                dedupe_dict_config["documents"] = cfg
                try_name = try_name or cfg["attribute_name"]
            elif not om.is_missing(parsed_config.dedupe.paragraphs, "attribute_name"):
                cfg = om.to_container(parsed_config.dedupe.paragraphs)
                assert isinstance(cfg, dict), "Expected dedupe.paragraphs to be a dict"
                dedupe_dict_config["paragraphs"] = cfg
                try_name = try_name or cfg["attribute_name"]
            else:
                raise ValueError("Either dedupe.documents or dedupe.paragraphs must be specified")

            if try_name is None:
                raise ValueError("dedupe.name must be specified")
            dedupe_dict_config["name"] = try_name

            # add the dedupe config to the main config
            dict_config["dedupe"] = dedupe_dict_config

            # perform some path validation to make sure we don't call the mixer with invalid config
            total_matching_documents = 0
            for document in parsed_config.documents:
                dict_config.setdefault("documents", []).append(str(document))

                if document.count("*") > 1:
                    raise DolmaConfigError("Only one wildcard is allowed in the document path")

                current_matching_documents = sum(1 for _ in glob_path(document))
                if current_matching_documents == 0:
                    # only raise a warning if no documents are found for a single path
                    logger.warning("No documents found for path %s", document)
                total_matching_documents += current_matching_documents

            if total_matching_documents == 0:
                # but raise an error if no documents are found for all paths
                raise DolmaConfigError(f"No documents found for the paths {dict_config['documents']}.")

            # The rust deduper does not work with remote files, so we need to download the bloom filter
            # if it is not local. If the remote file does not exists, and the bloom filter is read-only,
            # we raise an error.
            if not (path_is_local := is_local(parsed_config.bloom_filter.file)):
                local_bloom_file = stack.enter_context(get_path_to_temp_file())
                try:
                    with smart_open.open(parsed_config.bloom_filter.file, "rb") as f:
                        contents: bytes = f.read()  # pyright: ignore
                    local_bloom_file.write_bytes(contents)
                except (FileNotFoundError, OSError) as ex:
                    if parsed_config.bloom_filter.read_only:
                        raise ex
            else:
                local_bloom_file = Path(parsed_config.bloom_filter.file)

            dict_config["bloom_filter"] = {
                "file": str(local_bloom_file),
                "read_only": bool(parsed_config.bloom_filter.read_only),
                "size_in_bytes": int(parsed_config.bloom_filter.size_in_bytes),
                "estimated_doc_count": int(parsed_config.bloom_filter.estimated_doc_count),
                "desired_false_positive_rate": float(parsed_config.bloom_filter.desired_false_positive_rate),
            }

            if dict_config["bloom_filter"]["size_in_bytes"] <= 0 and (
                dict_config["bloom_filter"]["estimated_doc_count"] <= 0
                or dict_config["bloom_filter"]["desired_false_positive_rate"] <= 0
            ):
                raise ValueError(
                    "Either bloom_filter.size_in_bytes or bloom_filter.estimated_doc_count and "
                    "bloom_filter.desired_false_positive_rate must be specified"
                )

            dict_config["work_dir"] = {"input": str(work_dirs.input), "output": str(work_dirs.output)}
            dict_config["processes"] = int(parsed_config.processes)

            if len(dict_config["documents"]) == 0:
                raise ValueError("At least one document must be specified")

            print_config(dict_config)
            if parsed_config.dryrun:
                logger.info("Exiting due to dryrun.")
                return

            # run the deduper
            deduper(dict_config)

            # upload to remote file if necessary
            if not parsed_config.bloom_filter.read_only and not path_is_local:
                print(f"Pushing Bloom filter to {parsed_config.bloom_filter.file}")
                local = stack.enter_context(smart_open.open(local_bloom_file, "rb"))
                remote = stack.enter_context(smart_open.open(parsed_config.bloom_filter.file, "wb"))
                remote.write(local.read())
