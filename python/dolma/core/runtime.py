import argparse
import gzip
import logging
import multiprocessing
import os
import tempfile
from contextlib import ExitStack
from queue import Queue
from typing import Dict, List, Optional, Union

import msgspec
import smart_open

from .data_types import InputSpec, OutputSpec
from .errors import DolmaFatalError, DolmaRetryableFailure, DolmaShardError
from .parallel import BaseParallelProcessor
from .paths import join_path, make_relative, split_glob, split_path
from .registry import TaggerRegistry
from .utils import make_variable_name


def _make_paths_from_substitution(paths: List[str], find: str, replace: str) -> List[str]:
    """
    Utility function to make paths using a find/replace substitution. This is useful if you want to
    create a destination path from a source path by replacing part of the source path with something else.

    For example, if you have a source path of `current_paths = ["s3://bucket/data/documents/**.json.gz"]` and
    you want to replace `documents` with `attributes`, you can use this function to do that. by calling
    `_make_paths_from_substitution(current_paths, "documents", "attribute")`. This will return the following
    list `["s3://bucket/data/attributes"]`. Note how glob patterns are removed from the paths.
    """
    new_paths: List[str] = []
    for curr in paths:
        curr_pre_glob, _ = split_glob(curr)
        curr_prot, curr_parts = split_path(curr_pre_glob)
        find_dir_index = curr_parts.index(find)

        if not curr_pre_glob.strip():
            raise RuntimeError(f"Path '{curr}' contains a wildcard at the beginning. ")
        elif find_dir_index < 0:
            raise RuntimeError(f"Path '{curr}' does not contain a '{find}' component.")

        dst_parts = [p if i != find_dir_index else replace for i, p in enumerate(curr_parts)]
        new_paths.append(join_path(curr_prot, dst_parts))

    return new_paths


def _make_paths_from_prefix(paths: List[str], prefix: str) -> List[str]:
    new_paths: List[str] = []
    prefix_prot, prefix_path = split_path(prefix)
    _, relative_paths = make_relative(
        paths,
    )

    for curr_path in relative_paths:
        base_curr_path, _ = split_glob(curr_path)
        new_paths.append(join_path(prefix_prot, prefix_path, base_curr_path))

    return new_paths


class TaggerProcessor(BaseParallelProcessor):
    @classmethod
    def get_logger(cls) -> logging.Logger:
        return logging.getLogger(cls.__name__)

    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue,  # queue must be the first argument, and it should be a positional-only argument
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        """We override this method to specify which units we want to keep track of in a progress bar.
        Specifically, we keep track of files and documents in this example. Their default value must be zero."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: "Queue",
        **kwargs,
    ):
        """Lets count run the taggers! We will use the destination path to save each tagger output."""

        # get names of taggers
        taggers_names = kwargs.get("taggers_names", None)
        if taggers_names is None:
            raise RuntimeError("Taggers not in kwargs, this is a bug! Please report it.")
        elif not isinstance(taggers_names, list) or not all(isinstance(t, str) for t in taggers_names):
            raise RuntimeError("Taggers are in the wrong format, this is a bug! Please report it.")
        taggers = {make_variable_name(t): TaggerRegistry.get(t)() for t in taggers_names}

        # get name of experiment
        experiment_name = kwargs.get("experiment_name", None)
        if experiment_name is None:
            raise RuntimeError("Experiment name not in kwargs, this is a bug! Please report it.")
        experiment_name = make_variable_name(experiment_name)

        # skip on failure
        skip_on_failure = kwargs.get("skip_on_failure", False)

        # local read cache
        local_read_cache = kwargs.get("local_read_cache", None)

        # interval at which to update the progress bar; will double if it gets
        # too full
        update_interval = 1

        # running document count; gets reset every time we update the progress
        # bar
        docs_cnt = 0

        # creating dedicated encoders/decoders speeds up the process
        encoder = msgspec.json.Encoder()
        decoder = msgspec.json.Decoder(InputSpec)

        # this will be used to cache the file locally if needed
        caching_path = source_path

        with ExitStack() as stack:
            try:
                # open each file for reading and writing. We use open_file_for_read to handle s3 paths and
                # download the file locally if needed, while gzip.open is used to read and write gzipped files.
                in_stream = stack.enter_context(smart_open.open(source_path, "rt", encoding="utf-8"))
                out_stream = stack.enter_context(smart_open.open(destination_path, "wt", encoding="utf-8"))

                for raw in in_stream:
                    # row = json.loads(raw)
                    row = decoder.decode(raw)

                    # running the taggers and merging them flat
                    attributes = {}
                    for tagger_name, tagger in taggers.items():
                        for key_name, key_value in tagger.tag(row).items():
                            key_name = f"{experiment_name}__{tagger_name}__{make_variable_name(key_name)}"
                            attributes[key_name] = key_value

                    # make output file
                    output = OutputSpec(source=row.source, id=row.id, attributes=attributes)

                    # write the output to the output file
                    out_stream.write(encoder.encode(output).decode("utf-8") + "\n")  # pyright: ignore

                    # increment the number of documents processed so far
                    docs_cnt += 1

                    if docs_cnt % update_interval == 0:
                        # update the progress bar every 1000 documents to prevent
                        # buffering
                        cls.increment_progressbar(queue, documents=docs_cnt)
                        docs_cnt = 0

                        if queue.qsize() >= multiprocessing.cpu_count():
                            # double the update interval if the queue is full
                            update_interval *= 2

            except Exception as e:
                # handle any exception that might have occurred
                msg = f"Failed to process {source_path} due to {e.__class__.__name__}: {' '.join(e.args)}"
                if e.__class__.__name__ == "IncompleteReadError":
                    # Intermittent error that occurs when reading from S3
                    raise DolmaRetryableFailure(msg) from e
                else:
                    if skip_on_failure:
                        raise DolmaShardError(msg) from e
                    else:
                        raise DolmaFatalError(msg) from e
            finally:
                if caching_path != source_path and os.path.exists(caching_path):
                    os.remove(caching_path)

        # increment the files progress bar
        cls.increment_progressbar(queue, files=1, documents=docs_cnt)

    @classmethod
    def main(
        cls,
        documents: List[str],
        destination: Union[None, List[str]] = None,
        metadata: Union[None, List[str]] = None,
    ):
        if destination is None:
            try:
                destination = _make_paths_from_substitution(documents, "documents", "attributes")
            except Exception as e:
                raise RuntimeError("Could not make destination paths from documents paths") from e

        if metadata is None:
            _, rel_docs = make_relative(documents)

        # use a local read cache if we are in safe mode or if a local read cache is provided
        local_read_cache = opts.local_read_cache or (tempfile.gettempdir() if opts.safe_mode else None)

        with tempfile.TemporaryDirectory() as tempdir:
            metadata_workdir = opts.reuse_existing or tempdir
            ignore_existing = opts.reuse_existing is None
            manually_included_paths = opts.manually_included_paths
            if (
                manually_included_paths
                and len(manually_included_paths) == 1
                and os.path.exists(manually_included_paths[0])
            ):
                manually_included_paths = [lp.strip() for lp in open(manually_included_paths[0])]
            manually_excluded_paths = opts.manually_excluded_paths
            if (
                manually_excluded_paths
                and len(manually_excluded_paths) == 1
                and os.path.exists(manually_excluded_paths[0])
            ):
                manually_excluded_paths = [lp.strip() for lp in open(manually_excluded_paths[0])]

            msg = (
                "----- TaggerProcessor -----\n"
                f"source:       {source_prefix}\n"
                f"destination:  {destination_prefix}\n"
                f"scratch:      {tempdir}\n"
                f"taggers:      {', '.join(opts.taggers)}\n"
                f"parallel:     {opts.parallel}\n"
                f"debug:        {opts.debug}\n"
                f"skip on fail: {opts.skip_on_failure}\n"
                f"reuse prev:   {not ignore_existing}\n"
                f"workdir:      {metadata_workdir}\n"
                f"safe mode:    {opts.safe_mode}\n"
                f"local cache:  {local_read_cache}\n"
                f"file regex:   {opts.files_regex_pattern}\n"
                "---------------------------\n"
            )
            print(msg)

            parallel_compute = cls(
                source_prefix=source_prefix,
                destination_prefix=destination_prefix,
                metadata_prefix=metadata_workdir,
                num_processes=opts.parallel,
                ignore_existing=ignore_existing,
                debug=opts.debug,
                include_paths=opts.manually_included_paths,
                exclude_paths=opts.manually_excluded_paths,
                files_regex_pattern=opts.files_regex_pattern,
            )
            parallel_compute(
                taggers_names=opts.taggers,
                experiment_name=opts.experiment_name,
                skip_on_failure=opts.skip_on_failure,
                local_read_cache=local_read_cache,
                retry_on_read_error=opts.retry_on_read_error,
            )
