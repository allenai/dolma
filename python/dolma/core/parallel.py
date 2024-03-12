import inspect
import itertools
import logging
import multiprocessing
import pickle
import random
import re
import time
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union

import smart_open
import tqdm
from typing_extensions import TypeAlias

from .errors import DolmaError, DolmaRetryableFailure
from .loggers import get_logger
from .paths import (
    add_suffix,
    glob_path,
    join_path,
    make_relative,
    mkdir_p,
    parent,
    split_path,
    sub_prefix,
)

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
    def empty(cls) -> "AllPathsTuple":
        return AllPathsTuple([], [], [], [])


class BaseParallelProcessor:
    """A base parallel processor that supports applying the same process_single method to a list of files.

    This class is meant to be subclassed. The subclass must implement:
        - `process_single` method, which takes a source path file to transform, and a destination path where
           to save the transformed file.
        - `increment_progressbar` method, which defines which units to keep track of in the progress bar.

    See documentation of both methods for more details on how to implement them correctly.
    """

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
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        files_regex_pattern: Optional[str] = None,
        retries_on_error: int = 0,
        process_single_kwargs: Union[None, KwargsType, List[KwargsType]] = None,
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
            ignore_existing (bool, optional): Whether to ignore files that have been already processed and
                re-run the processor on all files from scratch. Defaults to False.
            include_paths (Optional[List[str]], optional): A list of paths to include. If provided, only files
                that match one of the paths will be processed. Defaults to None.
            exclude_paths (Optional[List[str]], optional): A list of paths to exclude. If provided, files that
                match one of the paths will be skipped. Defaults to None.
            files_regex_pattern (Optional[str], optional): A regex pattern to match files. If provided, only
                files that match the pattern will be processed. Defaults to None.
            retries_on_error (int, optional): The number of retries to attempt if an error occurs.
                Defaults to 0.
            process_single_kwargs (Union[None, KwargsType, List[KwargsType]], optional): Additional kwargs to
                pass to the process_single method. If a single dict is provided, it will be used for all source
                prefixes. If a list of dicts is provided, each dict will be used for the corresponding source.
                By default, no additional kwargs are passed.
        """

        self.src_prefixes = [source_prefix] if isinstance(source_prefix, str) else source_prefix
        self.dst_prefixes = [destination_prefix] if isinstance(destination_prefix, str) else destination_prefix
        self.meta_prefixes = [metadata_prefix] if isinstance(metadata_prefix, str) else metadata_prefix
        self.num_processes = num_processes
        self.debug = debug
        self.seed = seed
        self.pbar_timeout = pbar_timeout
        self.ignore_existing = ignore_existing

        self.include_paths = set(include_paths) if include_paths is not None else None
        self.exclude_paths = set(exclude_paths) if exclude_paths is not None else None
        self.files_regex_pattern = re.compile(files_regex_pattern) if files_regex_pattern else None
        self.retries_on_error = retries_on_error

        # this are additional kwargs to pass to the process_single method
        process_single_kwargs = process_single_kwargs or {}
        if isinstance(process_single_kwargs, dict):
            self.process_single_kwargs = [process_single_kwargs] * len(self.src_prefixes)
        else:
            self.process_single_kwargs = process_single_kwargs

        # checking that the increment_progressbar method is subclassed correctly
        sig = inspect.signature(self.increment_progressbar)
        if "queue" not in sig.parameters or sig.parameters["queue"].kind != inspect.Parameter.POSITIONAL_ONLY:
            raise AttributeError(
                "increment_progressbar must have a positional-only argument named 'queue'; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        if "kwargs" in sig.parameters and sig.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD:
            raise AttributeError(
                "increment_progressbar must not have a **kwargs argument; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        if any(p.name != "queue" and p.default != 0 for p in sig.parameters.values()):
            raise AttributeError(
                "increment_progressbar must have a default value of 0 for all arguments except 'queue'; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )

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

        if any("*" in p for p in itertools.chain(self.dst_prefixes, self.meta_prefixes)):
            raise ValueError("Destination and metadata prefixes cannot contain wildcards.")

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger for the class."""
        return get_logger(cls.__name__)

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
    def _process_single_and_save_status(
        cls,
        source_path: str,
        destination_path: str,
        metadata_path: str,
        queue: QueueType,
        serialized_kwargs: bytes,
    ):
        """A wrapper around process single that saves a metadata file if processing is successful."""

        # make destination directory if it doesn't exist for the destination and metadata paths
        mkdir_p(parent(destination_path))
        mkdir_p(parent(metadata_path))

        kwargs = pickle.loads(serialized_kwargs)
        retries_on_error = kwargs.get("retries_on_error", 0) + 1
        while True:
            try:
                cls.process_single(
                    source_path=source_path, destination_path=destination_path, queue=queue, **kwargs
                )
                break
            except DolmaRetryableFailure as exception:
                retries_on_error -= 1
                if retries_on_error == 0:
                    raise DolmaError from exception

        # write the metadata file
        with smart_open.open(metadata_path, "wt") as f:
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

    @classmethod
    def _run_threaded_progressbar(
        cls,
        queue: QueueType,
        timeout: float,
    ):
        """Run a progress bar in a separate thread.

        Args:
            queue (QueueType): The queue to increment the progress bars.
            timeout (float): How often to update the progress bars in seconds.
        """

        sample_queue_output = cls.increment_progressbar(queue)

        with ExitStack() as stack:
            pbars = [
                stack.enter_context(
                    tqdm.tqdm(desc=str(k), unit=str(k)[:1], position=i, unit_scale=True)  # pyright: ignore
                )
                for i, k in enumerate(sample_queue_output)
            ]

            while True:
                item = queue.get()
                if item is None:
                    break

                for pbar, value in zip(pbars, item):
                    pbar.update(value)

                time.sleep(timeout)

    def _debug_run_all(
        self,
        all_source_paths: List[str],
        all_destination_paths: List[str],
        all_metadata_paths: List[str],
        all_process_kwargs: Union[List[KwargsType], None] = None,
        **process_single_kwargs: Any,
    ):
        """Run files one by one on the main process

        Args:
            all_source_paths (List[MultiPath]): The list of source paths to process.
            all_destination_paths (List[MultiPath]): The list of destination paths to save.
            all_metadata_paths (List[MultiPath]): The locations where to save metadata.
            all_process_kwargs (Union[List[KwargsType], None]): Additional kwargs to pass to the process_single
        """

        arguments_iterator = zip(
            # source paths
            all_source_paths,
            # destination paths
            all_destination_paths,
            # this is where we save the metadata to keep track of which files have been processed
            all_metadata_paths,
            # additional kwargs to pass to the process_single; if not provided, we use an empty dict
            # will be merged with the process_single_kwargs
            all_process_kwargs or [{} for _ in all_source_paths],
        )
        pbar_queue: QueueType = Queue()
        thread = Thread(target=self._run_threaded_progressbar, args=(pbar_queue, self.pbar_timeout), daemon=True)
        thread.start()

        for source_path, destination_path, metadata_path, process_kwargs in arguments_iterator:
            self._process_single_and_save_status(
                source_path=source_path,
                destination_path=destination_path,
                metadata_path=metadata_path,
                queue=pbar_queue,
                serialized_kwargs=pickle.dumps({**process_kwargs, **process_single_kwargs}),
            )

        pbar_queue.put(None)
        thread.join()

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
            retries_on_error=max(self.retries_on_error, other.retries_on_error),
            process_single_kwargs=[*self.process_single_kwargs, *other.process_single_kwargs],
        )

    def __radd__(self: BPP, other: BPP) -> BPP:
        """Combine two parallel processors into one."""
        return other.__add__(self)

    def _multiprocessing_run_all(
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

        arguments_iterator = zip(
            # source paths
            all_source_paths,
            # destination paths
            all_destination_paths,
            # this is where we save the metadata to keep track of which files have been processed
            all_metadata_paths,
            # additional kwargs to pass to the process_single; if not provided, we use an empty dict
            # will be merged with the process_single_kwargs
            all_process_kwargs,
        )

        # no need to be wasteful with processes: we only need as many cores a the minimum of the number of
        # source paths, destination paths, metadata paths, and process kwargs.
        num_processes = min(
            self.num_processes,
            len(all_source_paths),
            len(all_destination_paths),
            len(all_metadata_paths),
            len(all_process_kwargs),
        )

        with multiprocessing.Pool(processes=num_processes) as pool:
            pbar_queue: QueueType = (manager := multiprocessing.Manager()).Queue()
            thread = Thread(
                target=self._run_threaded_progressbar, args=(pbar_queue, self.pbar_timeout), daemon=True
            )
            thread.start()

            process_single_fn = partial(self.process_single, queue=pbar_queue)
            results = []

            for source_path, destination_path, metadata_path, process_kwargs in arguments_iterator:
                process_single_fn = partial(
                    self._process_single_and_save_status,
                    queue=pbar_queue,
                    source_path=source_path,
                    destination_path=destination_path,
                    metadata_path=metadata_path,
                    # we need to merge the process_single_kwargs with the additional kwargs
                    serialized_kwargs=pickle.dumps({**process_kwargs, **process_single_kwargs}),
                )
                result = pool.apply_async(process_single_fn)
                results.append(result)

            for result in results:
                result.get()

            pool.close()
            pool.join()

            pbar_queue.put(None)
            thread.join()
            manager.shutdown()

    def _valid_path(self, path: str) -> bool:
        if self.include_paths is not None and path not in self.include_paths:
            return False
        if self.exclude_paths is not None and path in self.exclude_paths:
            return False
        if self.files_regex_pattern is not None and not self.files_regex_pattern.search(path):
            return False
        return True

    def _get_all_paths(self) -> AllPathsTuple:
        """Get all paths to process using prefixes provided"""
        all_paths = AllPathsTuple.empty()

        for src_prefix, dst_prefix, meta_prefix, kwargs_prefix in zip(
            self.src_prefixes, self.dst_prefixes, self.meta_prefixes, self.process_single_kwargs
        ):
            current_source_prefixes = sorted(glob_path(src_prefix))

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

            # shuffle the order of the files so time estimation in progress bars is more accurate
            random.shuffle(rel_paths)

            # get a list of which metadata files already exist
            existing_metadata_names = set(
                re.sub(rf"{METADATA_SUFFIX}$", "", sub_prefix(path, meta_prefix))
                for path in glob_path(meta_prefix)
            )

            for path in rel_paths:
                if not self.ignore_existing and path in existing_metadata_names:
                    continue

                if not self._valid_path(path):
                    continue

                # create new paths to pass to taggers
                all_paths.src.append(add_suffix(prefix, path))
                all_paths.dst.append(add_suffix(dst_prefix, path))
                all_paths.meta.append(add_suffix(meta_prefix, path) + METADATA_SUFFIX)
                all_paths.kwargs.append(kwargs_prefix or {})

        return all_paths

    def __call__(self, **process_single_kwargs: Any):
        """Run the processor."""

        random.seed(self.seed)

        # in case the user wants to override the default kwargs for retries
        process_single_kwargs.setdefault("retries_on_error", self.retries_on_error)

        all_paths = self._get_all_paths()

        print(f"Found {len(all_paths.src):,} files to process")

        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all

        fn(
            all_source_paths=all_paths.src,
            all_destination_paths=all_paths.dst,
            all_metadata_paths=all_paths.meta,
            all_process_kwargs=all_paths.kwargs,
            **process_single_kwargs,
        )
