import logging
import multiprocessing
import tempfile
from contextlib import ExitStack, contextmanager
from queue import Queue
from typing import (
    IO,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Union,
)

import msgspec
import smart_open

from .data_types import InputSpec, OutputSpec, TaggerOutputDictType
from .errors import DolmaFatalError, DolmaRetryableFailure, DolmaShardError
from .parallel import BaseParallelProcessor
from .paths import join_path, make_relative, mkdir_p, split_glob, split_path
from .registry import TaggerRegistry
from .utils import make_variable_name

# this placeholder gets used when a user has provided no experiment name, and we want to use taggers'
# names as experiment names.
EXPERIMENT_PLACEHOLDER_NAME = "_______EXPERIMENT_PLACEHOLDER_NAME_______"


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
    """
    Utility function to make paths using a prefix. This is useful if you want to create a destination path
    from a source path by prepending a prefix to the source path.

    To create destination paths, we first find the longest common prefix among all source paths. Then, we
    we remove the prefix from each source path and prepend the new prefix to each source path. For example,
    if you have a source path of
    ```
    current_paths = [
        "s3://bucket/data/documentsA/**.json.gz",
        "s3://bucket/data/documentsB/**.json.gz",
    ]
    ```
    and you want to replace `s3://bucket/data/` with `s3://bucket/attributes/`, you can use this function
    to do that. by calling `_make_paths_from_prefix(current_paths, "s3://bucket/attributes/")`. This will
    return the following list

    ```
    [
        "s3://bucket/attributes/documentsA/",
        "s3://bucket/attributes/documentsB/",
    ]
    ```

    Note how glob patterns are removed from the paths.
    """

    new_paths: List[str] = []
    prefix_prot, prefix_path = split_path(prefix)
    _, relative_paths = make_relative(paths)

    for curr_path in relative_paths:
        base_curr_path, _ = split_glob(curr_path)
        new_paths.append(join_path(prefix_prot, prefix_path, base_curr_path))

    return new_paths


class TaggerOutputLocation(NamedTuple):
    exp: str
    name: str
    path: str


class TaggerOutputIO(NamedTuple):
    exp: str
    taggers: Set[str]
    path: str
    io: IO
    encoder: msgspec.json.Encoder

    def write(self, d: OutputSpec) -> None:
        enc = self.encoder.encode(d)
        self.io.write(enc.decode("utf-8") + "\n")


def _determine_output_paths_for_taggers(
    experiment_name: str, destination: str, taggers: Iterable[str]
) -> Dict[str, TaggerOutputLocation]:
    """Utility function to derive the paths to which taggers output should be written.

    If experiment_name is the placeholder name, then the name of each tagger will be used as part of the
    destination path. Otherwise, the destination path will be used for all taggers."""

    if experiment_name == EXPERIMENT_PLACEHOLDER_NAME:
        return {
            tagger_name: TaggerOutputLocation(
                exp=make_variable_name(tagger_name),
                name=make_variable_name(tagger_name),
                path=destination.replace(EXPERIMENT_PLACEHOLDER_NAME, tagger_name),
            )
            for tagger_name in taggers
        }
    else:
        return {
            tagger_name: TaggerOutputLocation(
                exp=make_variable_name(experiment_name), name=make_variable_name(tagger_name), path=destination
            )
            for tagger_name in taggers
        }


@contextmanager
def _make_output_streams(
    taggers_paths: Dict[str, TaggerOutputLocation], **open_kwargs: Any
) -> Generator[Dict[str, TaggerOutputIO], None, None]:
    """Utility function to open paths for taggers.

    It is designed NOT to open duplicate paths if multiple taggers are writing to the same file.
    """
    # keep track of the paths that have been opened
    opened: Dict[str, TaggerOutputIO] = {}

    with ExitStack() as stack:
        for key, loc in taggers_paths.items():
            if loc.path not in opened:
                # make sure the parent directory exists
                prot, path = split_path(loc.path)
                parent = join_path(prot, path[:-1])
                mkdir_p(parent)

                # open a new file and create a new encoder
                io = stack.enter_context(smart_open.open(loc.path, **open_kwargs))
                encoder = msgspec.json.Encoder()
                opened[loc.path] = TaggerOutputIO(
                    exp=loc.exp, taggers=set(), path=loc.path, io=io, encoder=encoder
                )

            # keep track of which taggers are writing to this paths
            opened[loc.path].taggers.add(key)

        yield opened


@contextmanager
def _write_sample_to_streams(
    taggers_paths: Dict[str, TaggerOutputLocation],
    output_streams: Dict[str, TaggerOutputIO],
    row: InputSpec,
) -> Generator[Dict[str, TaggerOutputDictType], None, None]:
    """Utility function to write a sample to the output streams; yields a dictionary that should be used
    to collect the output of each tagger."""

    samples_collectors: Dict[str, TaggerOutputDictType] = {}
    yield samples_collectors

    attributes_by_stream: Dict[str, TaggerOutputDictType] = {}
    for tagger_name, tagger_data in samples_collectors.items():
        tagger_output = taggers_paths[tagger_name]
        for tagger_key, tagger_value in tagger_data.items():
            tagger_key = f"{tagger_output.exp}__{tagger_output.name}__{make_variable_name(tagger_key)}"
            attributes_by_stream.setdefault(tagger_output.path, {})[tagger_key] = tagger_value

    for stream_path, attributes in attributes_by_stream.items():
        output = OutputSpec(source=row.source, id=row.id, attributes=attributes)
        output_streams[stream_path].write(output)


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
        if (experiment_name := kwargs.get("experiment_name", None)) is None:
            raise RuntimeError("Experiment name not in kwargs, this is a bug! Please report it.")

        # this is the dictionary that will hold the output of each tagger
        taggers_paths = _determine_output_paths_for_taggers(
            experiment_name=experiment_name, destination=destination_path, taggers=taggers
        )

        # skip on failure
        skip_on_failure = kwargs.get("skip_on_failure", False)

        # interval at which to update the progress bar; will double if it gets
        # too full
        update_interval = 1

        # running document count; gets reset every time we update the progress
        # bar
        docs_cnt = 0

        # creating dedicated decoder speeds up the process
        decoder = msgspec.json.Decoder(InputSpec)

        with smart_open.open(source_path, "rt", encoding="utf-8") as in_stream, _make_output_streams(
            taggers_paths=taggers_paths, mode="wt", encoding="utf-8"
        ) as output_streams:
            try:
                for raw in in_stream:
                    row = decoder.decode(raw)

                    with _write_sample_to_streams(
                        taggers_paths=taggers_paths,
                        output_streams=output_streams,
                        row=row,
                    ) as samples_collectors:
                        # we run the taggers; the context manager will write the output to the output streams
                        for tagger_name, tagger in taggers.items():
                            samples_collectors[tagger_name] = tagger.tag(row)

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

        # increment the files progress bar
        cls.increment_progressbar(queue, files=1, documents=docs_cnt)


def create_and_run_tagger(
    documents: List[str],
    taggers: List[str],
    experiment: Optional[str] = None,
    destination: Union[None, str, List[str]] = None,
    metadata: Union[None, str, List[str]] = None,
    debug: bool = False,
    seed: int = 0,
    ignore_existing: bool = False,
    skip_on_failure: bool = False,
    retries_on_error: int = 0,
    num_processes: int = 1,
):
    """This function creates a tagger and runs it on a list of documents.

    Args:
        documents (List[str]): List of documents to run the taggers on. Each element of the list is a path to
            a file containing documents in json lines format, or a glob pattern that matches such files.
        taggers (List[str]): List of taggers to run. Each element of the list is the name of a tagger.
        experiment (str, optional): The name of the experiment. This will be used to prefix the names of the
            attributes, as well as the name of the directory where the outputs will be saved in `destination`.
            If not provided, the name of each tagger will be used as the experiment name.
        destination (Union[None, str, List[str]], optional): The path where the outputs will be saved. If
            `None`, the outputs will be saved in a directory parallel to the directory containing the
            documents, with the same name as `experiment`. If a string, paths corresponding to each element
            of `documents` will be created by determining a relative path from the directory containing the
            documents.
        metadata (Union[None, str, List[str]], optional): Location where to save metadata that keeps track of
            which documents have been processed. If `None`, the metadata will be saved in a temporary directory.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        seed (int, optional): The seed to use for the random number generator. Defaults to 0.
        ignore_existing (bool, optional): Whether to ignore existing outputs and re-run the taggers.
            Defaults to False.
        skip_on_failure (bool, optional): Whether to skip a document if it fails to process. Defaults to False.
        retries_on_error (int, optional): Number of times to retry processing a document if it fails.
            Defaults to 0 (fail immediately)
        num_processes (int, optional): Number of processes to use. Defaults to 1.
    """

    # use placeholder experiment name if none is provided; raise an error if the placeholder name is used
    if experiment == EXPERIMENT_PLACEHOLDER_NAME:
        raise RuntimeError(f"Experiment name cannot be {EXPERIMENT_PLACEHOLDER_NAME}; reserved for internal use.")
    elif experiment is None:
        experiment = EXPERIMENT_PLACEHOLDER_NAME

    if destination is None:
        try:
            destination = _make_paths_from_substitution(documents, "documents", f"attributes/{experiment}")
        except Exception as e:
            raise RuntimeError("Could not make destination paths from documents paths") from e
    elif isinstance(destination, str):
        try:
            destination = _make_paths_from_prefix(documents, join_path(None, destination, experiment))
        except Exception as e:
            raise RuntimeError(f"Could not make destination paths from prefix {destination}") from e

    metadata = metadata or tempfile.mkdtemp()
    if isinstance(metadata, str):
        try:
            metadata = _make_paths_from_prefix(documents, metadata)
        except Exception as e:
            raise RuntimeError(f"Could not make metadata paths from prefix {metadata}") from e

        tagger = TaggerProcessor(
            source_prefix=documents,
            destination_prefix=destination,
            metadata_prefix=metadata,
            debug=debug,
            seed=seed,
            ignore_existing=ignore_existing,
            retries_on_error=retries_on_error,
            num_processes=num_processes,
        )
        tagger(
            experiment_name=experiment,
            taggers_names=taggers,
            skip_on_failure=skip_on_failure,
        )
