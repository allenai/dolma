import itertools
import logging
import multiprocessing
import pickle
import random
import re
from datetime import datetime
from functools import partial
from queue import Queue
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union

import backoff
import smart_open
from backoff.types import Details
from typing_extensions import TypeAlias

from .errors import DolmaError, DolmaRetryableFailure
from .loggers import get_logger
from .mp_tools import PoolWithDebug, get_manager
from .paths import (
    add_suffix,
    exists,
    glob_path,
    join_path,
    make_relative,
    mkdir_p,
    parent,
    split_path,
)
from .progressbar import BaseProgressBar
from .utils import batch_iterator

METADATA_SUFFIX = ".done.txt"

# we need to quote the type alias because we want to support Python 3.8
QueueType: TypeAlias = "Queue[Union[None, Tuple[int, ...]]]"
KwargsType: TypeAlias = Dict[str, Any]
BPP = TypeVar("BPP", bound="BaseParallelProcessor")


class AllPathsTuple(NamedTuple):
    src: List[str]
    dst: List[str]
    meta: List[str]
    kwargs: List[KwargsType]

    @classmethod
    def new(cls) -> "AllPathsTuple":
        return AllPathsTuple([], [], [], [])

    def __len__(self) -> int:
        return len(self.src)

    @property
    def empty(self) -> bool:
        return len(self.src) == 0

    def partition(self, k: int = 1) -> List["AllPathsTuple"]:
        """Partition the paths into k / n slices containing k files each."""
        return [
            AllPathsTuple(
                src=self.src[i : i + k],
                dst=self.dst[i : i + k],
                meta=self.meta[i : i + k],
                kwargs=self.kwargs[i : i + k],
            )
            for i in range(0, len(self.src), k)
        ]


class BaseParallelProcessor:
    """A base parallel processor that supports applying the same process_single method to a list of files.

    This class is meant to be subclassed. The subclass must implement:
        - `process_single` method, which takes a source path file to transform, and a destination path where
           to save the transformed file.
        - `increment_progressbar` method, which defines which units to keep track of in the progress bar.

    See documentation of both methods for more details on how to implement them correctly.
    """

    PROGRESS_BAR_CLS: Type[BaseProgressBar]

    def __init__(
        self,
        source_prefix: Union[str, List[str]],
        destination_prefix: Union[str, List[str]],
        metadata_prefix: Union[str, List[str]],
        num_processes: int = 1,
        debug: bool = False,
        seed: int = 0,
        pbar_timeout: float = 1e-3,
        ignore_existing: bool = False,
        skip_source_glob: bool = False,
        shuffle_src_paths: bool = True,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        files_regex_pattern: Optional[str] = None,
        batch_size: int = 1,
        process_single_kwargs: Union[None, KwargsType, List[KwargsType]] = None,
        backoff_max_time: Optional[float] = None,
        backoff_max_tries: int = 1,
        retries_on_error: Optional[int] = None,
        backoff_exceptions: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
    ):
        """Initialize the parallel processor.

        Args:
            source_prefix (str): The location where source files are stored. This can be a local directory or a
                prefix to an S3 location.
            destination_prefix (str): The location where to save the transformed files. This can be a local
                directory or a prefix to an S3 location. Local directories will be created if they do not exist.
                The directory structure from the source prefix will be replicated in the destination prefix;
                file names will also be the same.
            metadata_prefix (str): The prefix of the metadata files to save. This can be a local path or an
                S3 path. Metadata output will be created for each file after it is processed. Filenames are
                checked to verify if a file has been processed and can be skipped unless `ignore_existing` is
                set to true.
            num_processes (int, optional): The number of processes to use. Defaults to 1.
            debug (bool, optional): Whether to run in debug mode; if true, no multiprocessing will be used.
                Defaults to False.
            seed (int, optional): The random seed to use when shuffling input files. Defaults to 0.
            pbar_timeout (float, optional): How often to update progress bars in seconds.
                Defaults to 0.01 seconds.
            skip_source_glob (bool, optional): Do not glob source files. Off by default.
            ignore_existing (bool, optional): Whether to ignore files that have been already processed and
                re-run the processor on all files from scratch. Defaults to False.
            shuffle_src_paths (bool, optional): Whether to shuffle the source paths before processing them.
                Defaults to True.
            include_paths (List[str], optional): A list of paths to include. If provided, only files
                that match one of the paths will be processed. Defaults to None.
            exclude_paths (List[str], optional): A list of paths to exclude. If provided, files that
                match one of the paths will be skipped. Defaults to None.
            files_regex_pattern (str, optional): A regex pattern to match files. If provided, only
                files that match the pattern will be processed. Defaults to None.
            batch_size: (int, optional): number of files to group in a single bat
            process_single_kwargs (Union[None, KwargsType, List[KwargsType], optional): Additional kwargs to
                pass to the process_single method. If a single dict is provided, it will be used for all source
                prefixes. If a list of dicts is provided, each dict will be used for the corresponding source.
                By default, no additional kwargs are passed.
            backoff_max_time (float, optional): The maximum time to backoff. Defaults to None.
            backoff_max_tries (int, optional): The maximum number of tries to backoff. Defaults to 1.
            backoff_exceptions (Union[Type[Exception], Tuple[Type[Exception], ...]], optional): The
                exceptions to backoff on. Defaults to `dolma.core.errors.DolmaRetryableFailure`.
            retries_on_error (int, optional): Deprecated. The number of retries to attempt on error.
                Defaults to None.
        """
        self.src_prefixes = [source_prefix] if isinstance(source_prefix, str) else source_prefix
        self.dst_prefixes = [destination_prefix] if isinstance(destination_prefix, str) else destination_prefix
        self.meta_prefixes = [metadata_prefix] if isinstance(metadata_prefix, str) else metadata_prefix
        self.num_processes = num_processes
        self.debug = debug
        self.seed = seed
        self.pbar_timeout = pbar_timeout
        self.ignore_existing = ignore_existing

        self.logger = self.get_logger()

        self.include_paths = set(include_paths) if include_paths is not None else None
        self.exclude_paths = set(exclude_paths) if exclude_paths is not None else None
        self.files_regex_pattern = re.compile(files_regex_pattern) if files_regex_pattern else None
        self.shuffle_src_paths = shuffle_src_paths

        # this manages how many files to pass to a single processor
        self.batch_size = batch_size

        if retries_on_error is not None:
            self.logger.warning(
                "The `retries_on_error` parameter is deprecated and will be removed in a future release. "
                "Please use `backoff_max_tries` instead."
            )
            backoff_max_tries = retries_on_error + 1

        # this controls backoff
        self.backoff_max_time: float = float(backoff_max_time or "inf")
        self.backoff_max_tries: int = int(backoff_max_tries)
        self.backoff_exceptions: Tuple[Type[Exception], ...] = (
            (backoff_exceptions,)
            if isinstance(backoff_exceptions, type)
            else backoff_exceptions or (DolmaRetryableFailure,)
        )

        # this are additional kwargs to pass to the process_single method
        process_single_kwargs = process_single_kwargs or {}
        if isinstance(process_single_kwargs, dict):
            self.process_single_kwargs = [process_single_kwargs] * len(self.src_prefixes)
        else:
            self.process_single_kwargs = process_single_kwargs

        if not hasattr(self, "PROGRESS_BAR_CLS"):
            self.PROGRESS_BAR_CLS = BaseProgressBar.from_increment_function(self)

        if len(self.src_prefixes) != len(self.dst_prefixes):
            raise ValueError(
                "The number of source and destination prefixes must be the same "
                f"(got {len(self.src_prefixes)} and {len(self.dst_prefixes)})"
            )
        elif len(self.src_prefixes) != len(self.meta_prefixes):
            raise ValueError(
                "The number of source and metadata prefixes must be the same."
                f"(got {len(self.src_prefixes)} and {len(self.meta_prefixes)})"
            )
        elif len(self.src_prefixes) != len(self.process_single_kwargs):
            raise ValueError(
                "The number of source prefixes and process_single_kwargs must be the same."
                f"(got {len(self.src_prefixes)} and {len(self.process_single_kwargs)})"
            )

        if len(self.src_prefixes) == 0:
            raise ValueError("At least one source prefix must be provided.")

        self.skip_source_glob = skip_source_glob

        if any("*" in p for p in itertools.chain(self.dst_prefixes, self.meta_prefixes)):
            raise ValueError("Destination and metadata prefixes cannot contain wildcards.")

        if not hasattr(self, "PROGRESS_BAR_CLS"):
            raise AttributeError("BaseParallelProcessor subclasses must define the PROGRESS_BAR_CLS attribute.")

    def __add__(self: BPP, other: BPP) -> BPP:
        """Combine two parallel processors into one."""
        if not type(self) is type(other):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

        # we try combining the two list of include paths; if they are both None, then set the combo back to none
        include_paths: Union[List[str], None] = [*(self.include_paths or []), *(other.include_paths or [])]
        include_paths = sorted(set(include_paths or [])) if len(include_paths or []) else None

        # do the same for exclude paths
        exclude_paths: Union[List[str], None] = [*(self.exclude_paths or []), *(other.exclude_paths or [])]
        exclude_paths = sorted(set(exclude_paths or [])) if len(exclude_paths or []) else None

        # for the regex, do a simple or if both are set
        regex_pattern: Union[str, None] = None
        if self.files_regex_pattern and other.files_regex_pattern:
            regex_pattern = "(" + self.files_regex_pattern.pattern + "|" + other.files_regex_pattern.pattern + ")"
        elif self.files_regex_pattern:
            regex_pattern = self.files_regex_pattern.pattern
        elif other.files_regex_pattern:
            regex_pattern = other.files_regex_pattern.pattern

        return type(self)(
            source_prefix=[*self.src_prefixes, *other.src_prefixes],
            destination_prefix=[*self.dst_prefixes, *other.dst_prefixes],
            metadata_prefix=[*self.meta_prefixes, *other.meta_prefixes],
            num_processes=max(self.num_processes, other.num_processes),
            debug=self.debug or other.debug,
            seed=self.seed,
            pbar_timeout=max(self.pbar_timeout, other.pbar_timeout),
            ignore_existing=self.ignore_existing or other.ignore_existing,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            files_regex_pattern=regex_pattern,
            batch_size=max(self.batch_size, other.batch_size),
            process_single_kwargs=[*self.process_single_kwargs, *other.process_single_kwargs],
            backoff_max_time=min(self.backoff_max_time, other.backoff_max_time),
            backoff_max_tries=min(self.backoff_max_tries, other.backoff_max_tries),
            backoff_exceptions=tuple(set(self.backoff_exceptions + other.backoff_exceptions)),
        )

    def __radd__(self: BPP, other: BPP) -> BPP:
        """Combine two parallel processors into one."""
        return other.__add__(self)

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger for the class."""
        return get_logger(cls.__name__, "info")

    @classmethod
    def process_batch(
        cls,
        source_paths: List[str],
        destination_paths: List[str],
        queue: QueueType,
        kwargs: List[Dict[str, Any]],
    ):
        """Process multiple files. Naively calls process_single for each file, but can be overridden."""
        for src_path, dst_path, single_kwargs in zip(source_paths, destination_paths, kwargs):
            cls.process_single(source_path=src_path, destination_path=dst_path, queue=queue, **single_kwargs)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs: Any,
    ):
        """Process a single file.

        This method must be implemented by the subclass. It takes a source path file to transform, and a
        destination path where to save the transformed file. It also takes a queue to increment the progress
        bars. The queue should be passed to the `increment_progressbar` method.

        Args:
            source_path (str): The path to the source file to transform. Can be an S3 path or a local path.
            destination_path (str): The path to the destination file to save. Can be an S3 path or a local path.
            queue (QueueType): The queue to increment the progress bars.
        """
        raise NotImplementedError()

    @classmethod
    def _log_backoff(cls, details: Details):
        """Log backoff details."""
        message = (
            f"Backing off `{details['target'].__name__}` "
            f"after {details['tries']:,} "
            f"tries (wait: {details.get('wait', 0.0):.2f}s)"
        )
        if ex := details.get("exception"):
            # add details about the exception to the message
            import traceback  # pylint: disable=import-outside-toplevel

            message += " due to " + "\n".join(traceback.format_exception_only(ex)).strip()  # type: ignore

        cls.get_logger().warning(message)

    @classmethod
    def _process_batch_and_save_status(
        cls,
        source_paths: List[str],
        destination_paths: List[str],
        metadata_paths: List[str],
        queue: QueueType,
        serialized_kwargs: List[bytes],
        backoff_max_time: float,
        backoff_max_tries: int,
        backoff_exceptions: Tuple[Type[Exception], ...],
    ):
        """A wrapper around process single that saves a metadata file if processing is successful."""

        # make destination directory if it doesn't exist for the destination and metadata paths
        for path in itertools.chain(destination_paths, metadata_paths):
            mkdir_p(parent(path))

        # we unpickle the serialized kwargs
        deserialized_kwargs = [pickle.loads(kw) for kw in serialized_kwargs]

        # use backoff library to retry on failure; function _log_backoff is called on backoff
        # to inform the user of the backoff details.
        fn_with_backoff = backoff.on_exception(
            backoff.expo,
            exception=backoff_exceptions,
            max_tries=backoff_max_tries,
            max_time=backoff_max_time,
            on_backoff=cls._log_backoff,
        )(cls.process_batch)

        # start processing the file here
        fn_with_backoff(
            source_paths=source_paths, destination_paths=destination_paths, queue=queue, kwargs=deserialized_kwargs
        )

        # write the metadata files
        for path in metadata_paths:
            with smart_open.open(path, "wt") as f:
                f.write(datetime.now().isoformat())

    @classmethod
    def increment_progressbar(cls, queue: QueueType, /, **kwargs: int) -> Dict[str, int]:
        """Increment the progress bar by putting a tuple in the queue.

        When subclassing, we recommend defining which units to keep track of in the progress bar by
        defining keyword arguments. Then you can call the base class via `super()` and pass the keyword.
        Example:

        ```python
        class MyProcessor(BaseParallelProcessor):
            def increment_progressbar(self, queue, /, files = 0, documents = 0):   # we use two progress bars
                return super().increment_progressbar(queue, files=files, documents=documents)
        ```
        """
        queue.put(tuple(kwargs.get(k, 0) for k in kwargs))
        return kwargs

    def _run_all(
        self,
        all_source_paths: List[str],
        all_destination_paths: List[str],
        all_metadata_paths: List[str],
        all_process_kwargs: Union[List[KwargsType], None] = None,
        **process_single_kwargs: Any,
    ):
        """Run files in parallel using multiprocessing.

        Args:
            all_source_paths (List[MultiPath]): The list of source paths to process.
            all_destination_paths (List[MultiPath]): The list of destination paths to save.
            all_metadata_paths (List[MultiPath]): The locations where to save metadata.
            all_process_kwargs (Union[List[KwargsType], None]): Additional kwargs to pass to the process_single
        """
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            assert multiprocessing.get_start_method() == "spawn", "Multiprocessing start method must be spawn"

        all_process_kwargs = all_process_kwargs or [{} for _ in all_source_paths]

        batches = list(
            batch_iterator(
                # source paths
                all_source_paths,
                # destination paths
                all_destination_paths,
                # this is where we save the metadata to keep track of which files have been processed
                all_metadata_paths,
                # additional kwargs to pass to the process_single; if not provided, we use an empty dict
                # will be merged with the process_single_kwargs
                all_process_kwargs,
                # batch size is equal to 1 by default
                batch_size=self.batch_size,
            )
        )
        self.logger.info("Processing in %s batches", len(batches))

        # no need to be wasteful with processes: we only need as many cores a the number of batches
        num_processes = min(self.num_processes, len(batches))
        self.logger.info("Using %s processes", num_processes)

        with PoolWithDebug(processes=num_processes, debug=self.debug) as pool:
            pbar_queue: QueueType = (manager := get_manager(pool)).Queue()
            (pbar := self.PROGRESS_BAR_CLS(pbar_queue)).start()

            process_single_fn = partial(self.process_single, queue=pbar_queue)
            results = []

            for source_paths, destination_paths, metadata_paths, process_kwargs in batches:
                # we need to merge the process_single_kwargs with the additional kwargs
                # mypy is confused by the type of process_kwargs; we need to ignore the error
                serialized_kwargs = [
                    pickle.dumps({**kw, **process_single_kwargs}) for kw in process_kwargs  # type: ignore
                ]

                process_single_fn = partial(
                    self._process_batch_and_save_status,
                    queue=pbar_queue,
                    source_paths=source_paths,  # pyright: ignore
                    destination_paths=destination_paths,  # pyright: ignore
                    metadata_paths=metadata_paths,  # pyright: ignore
                    serialized_kwargs=serialized_kwargs,
                    backoff_max_time=self.backoff_max_time,
                    backoff_max_tries=self.backoff_max_tries,
                    backoff_exceptions=self.backoff_exceptions,
                )
                result = pool.apply_async(process_single_fn)
                results.append(result)

            for result in results:
                result.get()

            pool.close()
            pool.join()
            pbar.stop()
            manager.shutdown()

    def _valid_path(self, path: str) -> bool:
        if self.include_paths is not None and path not in self.include_paths:
            return False
        if self.exclude_paths is not None and path in self.exclude_paths:
            return False
        if self.files_regex_pattern is not None and not self.files_regex_pattern.search(path):
            return False
        return True

    def _get_all_paths(self) -> Tuple[AllPathsTuple, bool]:
        """Get all paths to process using prefixes provided"""
        all_paths = AllPathsTuple.new()

        for src_prefix, dst_prefix, meta_prefix, kwargs_prefix in zip(
            self.src_prefixes, self.dst_prefixes, self.meta_prefixes, self.process_single_kwargs
        ):
            current_source_prefixes = sorted([src_prefix] if self.skip_source_glob else glob_path(src_prefix))

            if len(current_source_prefixes) > 1:
                # make relative only makes sense if there is more than one path; otherwise, it's unclear
                # what a relative path would be.
                prefix, rel_paths = make_relative(current_source_prefixes)
            elif len(current_source_prefixes) == 1:
                # in case we have a single path, we can just use the path minus the file as the shared prefix,
                # and the file as the relative path
                prot, parts = split_path(current_source_prefixes[0])
                prefix, rel_paths = join_path(prot, *parts[:-1]), [parts[-1]]
            else:
                raise ValueError(f"Could not find any files matching {src_prefix}")

            if self.shuffle_src_paths:
                # shuffle the order of the files so time estimation in progress bars is more accurate
                random.shuffle(rel_paths)

            # # get a list of which metadata files already exist
            some_already_processed = False

            for path in rel_paths:
                metadata_path = add_suffix(meta_prefix, path) + METADATA_SUFFIX

                if not self._valid_path(path):
                    # invalid path; skip
                    continue

                if not self.ignore_existing and exists(metadata_path):
                    # metadata file exists, which indicates that the file has already been processed
                    some_already_processed = True
                    continue

                # create new paths to pass to taggers
                all_paths.src.append(add_suffix(prefix, path))
                all_paths.dst.append(add_suffix(dst_prefix, path))
                all_paths.meta.append(metadata_path)
                all_paths.kwargs.append(kwargs_prefix or {})

        return all_paths, some_already_processed

    def __call__(self, **process_single_kwargs: Any):
        """Run the processor."""

        random.seed(self.seed)

        all_paths, some_already_processed = self._get_all_paths()
        self.logger.info("Found %s files to process", len(all_paths.src))

        if all_paths.empty:
            if some_already_processed:
                self.logger.info("All files already processed; skipping.")
                return
            else:
                raise DolmaError("No files found to process.")

        self._run_all(
            all_source_paths=all_paths.src,
            all_destination_paths=all_paths.dst,
            all_metadata_paths=all_paths.meta,
            all_process_kwargs=all_paths.kwargs,
            **process_single_kwargs,
        )
