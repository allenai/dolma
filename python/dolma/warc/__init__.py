import os
import tempfile
from contextlib import ExitStack
from typing import List, Optional, Union

from ..core.paths import glob_path
from ..core.runtime import _make_paths_from_prefix

# we need to import these to make sure they are registered
from . import linearizers  # noqa: F401  # pyright: ignore
from .processor import WarcProcessor


def create_and_run_warc_pipeline(
    documents: Union[str, List[str]],
    destination: Union[str, List[str]],
    source_name: str,
    metadata: Union[None, str, List[str]] = None,
    debug: bool = False,
    seed: int = 0,
    ignore_existing: bool = False,
    skip_on_failure: bool = False,
    retries_on_error: int = 0,
    num_processes: int = 1,
    skip_if_empty_heuristics: bool = False,
    store_html_in_metadata: bool = False,
    linearizer_name: str = "resiliparse",
    extractors_name: Optional[List[str]] = None,
):
    with ExitStack() as stack:
        if metadata is None:
            if isinstance(destination, str):
                metadata = stack.enter_context(tempfile.TemporaryDirectory())
            else:
                metadata = [stack.enter_context(tempfile.TemporaryDirectory()) for _ in range(len(destination))]

        all_src_paths = []
        all_dst_paths = []
        all_meta_paths = []

        if isinstance(destination, str) and isinstance(metadata, str):
            for src_pattern in [documents] if isinstance(documents, str) else documents:
                all_src_paths.extend(list(glob_path(src_pattern)))
            all_dst_paths.extend(_make_paths_from_prefix(paths=all_src_paths, prefix=destination))
            all_meta_paths.extend(_make_paths_from_prefix(paths=all_src_paths, prefix=metadata))

        elif isinstance(destination, list) and isinstance(metadata, list):
            if not isinstance(documents, list):
                raise ValueError("documents must be a list of strings")
            if not isinstance(metadata, list):
                raise ValueError("metadata must be a list of strings")
            if len(documents) != len(destination):
                raise ValueError("documents and destination must have the same length")
            if len(metadata) != len(destination):
                raise ValueError("metadata and destination must have the same length")

            for src_pattern, dst_pattern, meta_pattern in zip(documents, destination, metadata):
                src_paths = list(glob_path(src_pattern))
                all_src_paths.extend(src_paths)
                all_dst_paths.extend(_make_paths_from_prefix(paths=src_paths, prefix=dst_pattern))
                all_meta_paths.extend(_make_paths_from_prefix(paths=src_paths, prefix=meta_pattern))
        else:
            raise ValueError("destination must be a string or a list of strings")

        processor = WarcProcessor(
            source_prefix=all_src_paths,
            destination_prefix=all_dst_paths,
            metadata_prefix=all_meta_paths,
            debug=debug,
            seed=seed,
            ignore_existing=ignore_existing,
            retries_on_error=retries_on_error,
            num_processes=num_processes,
        )
        processor(
            skip_on_failure=skip_on_failure,
            skip_if_empty_heuristics=skip_if_empty_heuristics,
            store_html_in_metadata=store_html_in_metadata,
            linearizer=linearizer_name,
            extractors=extractors_name,
            source_name=source_name,
        )
