import multiprocessing
import random
import re
from contextlib import ExitStack
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import msgspec
import smart_open

from ..core.data_types import InputSpecWithMetadata
from ..core.loggers import get_logger
from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.paths import (
    glob_path,
    join_path,
    make_relative,
    mkdir_p,
    parent,
    split_basename_and_extension,
)
from .word_tokenizers import TokenizerRegistry

LOGGER = get_logger(__name__)


def make_selector(jsonpath: str) -> Callable:
    """
    Returns a callable object that can be used to extract data from a
    JSON-like structure using JSONPath.

    Args:
        jsonpath (str): The JSONPath expression used to select the desired data.

    Returns:
        Callable: A callable object that takes a JSON-like structure as input and
            returns the selected data.

    Raises:
        ValueError: If the provided JSONPath expression is invalid.
    """
    # Split the JSONPath into parts
    jsonpath = jsonpath.lstrip("$")
    if m := re.match(r"^\[([0-9]+)\]", jsonpath):
        # This is an array index
        index = int(m.group(1))
        prev = make_selector(jsonpath[m.end() :])
        return lambda x: prev(x[index])

    elif m := re.match(r"^\.([a-zA-Z_][a-zA-Z0-9_]*)", jsonpath):
        # This is a key
        key = m.group(1)
        prev = make_selector(jsonpath[m.end() :])
        return lambda x: prev(x[key] if isinstance(x, dict) else getattr(x, key))
    elif not jsonpath.strip():
        return lambda x: x

    raise ValueError(f"Invalid JSONPath: {jsonpath}")


def combine_splits(sources: List[str], destination: str, splits: Optional[Tuple[str, ...]] = None):
    """Combine the splits generated for each source path into a single file for each split."""

    # if no splits are provided, we default to the standard train/dev/test splits
    splits = splits or ("train", "dev", "test")

    # we need to unique the sources because they are directories, not files.
    unique_sources = set(sources)

    for split in splits:
        with smart_open.open(join_path("", destination, f"{split}.txt"), "wt") as wf:
            # the paths are obtained by globbing the source directories for files containing the
            # name of the split.
            for path in chain.from_iterable(glob_path(f"{d}/*{split}*") for d in unique_sources):
                with smart_open.open(path, "rt") as rf:
                    wf.write(rf.read())


class BaseDataConverter(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: QueueType,
        /,
        train: int = 0,
        dev: int = 0,
        test: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, train=train, dev=dev, test=test)

    @classmethod
    def _make_text_fn(cls, text_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """
        Returns a function that extracts the text from the input document and preprocesses it.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

    @classmethod
    def _make_label_fn(cls, label_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """Create a function that extracts the label from the input document."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        """Script to perform extraction on a single file"""

        breakpoint()

        # get the probabilities for each split
        train_sample_rate = float(kwargs.get("train_sample_rate", 0.0))
        dev_sample_rate = train_sample_rate + float(kwargs.get("dev_sample_rate", 0.0))
        test_sample_rate = dev_sample_rate + float(kwargs.get("test_sample_rate", 0.0))

        # check if the sample sizes are valid
        if not 0 <= train_sample_rate <= 1:
            raise ValueError(f"train_sample_rate must be between 0 and 1, not {train_sample_rate}")
        if not train_sample_rate <= dev_sample_rate <= 1:
            raise ValueError(f"dev_sample_rate must be between 0 and 1, not {dev_sample_rate - train_sample_rate}")
        if not dev_sample_rate <= test_sample_rate <= 1:
            raise ValueError(f"test_sample_rate must be between 0 and 1, not {test_sample_rate - dev_sample_rate}")

        # these two functions are used to extract the text and label from each document
        text_fn = cls._make_text_fn(text_selector=kwargs.get("text_selector"))
        label_fn = cls._make_label_fn(label_selector=kwargs.get("label_selector"))

        # we create the tokenizer that is used to preprocess the text
        word_tokenizer = TokenizerRegistry.get(kwargs.get("tokenizer_name") or "noop")()

        decoder = msgspec.json.Decoder(InputSpecWithMetadata)
        train_cnt = dev_cnt = test_cnt = 0
        base_name, ext = split_basename_and_extension(destination_path)
        update_interval = 1

        with ExitStack() as stack:
            readfile = stack.enter_context(smart_open.open(source_path, "rt"))
            train_writefile = stack.enter_context(smart_open.open(f"{base_name}-train{ext}", "wt"))
            dev_writefile = stack.enter_context(smart_open.open(f"{base_name}-dev{ext}", "wt"))
            test_writefile = stack.enter_context(smart_open.open(f"{base_name}-test{ext}", "wt"))

            for i, line in enumerate(readfile):
                if (p := random.random()) > test_sample_rate:
                    continue

                data = decoder.decode(line)
                text = word_tokenizer(text_fn(data))
                label = label_fn(data)

                if p <= train_sample_rate:
                    train_writefile.write(label + text + "\n")
                    train_cnt += 1
                elif p <= dev_sample_rate:
                    dev_writefile.write(label + text + "\n")
                    dev_cnt += 1
                else:
                    test_writefile.write(label + text + "\n")
                    test_cnt += 1

                if i % update_interval == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(queue, train=train_cnt, dev=dev_cnt, test=test_cnt)
                    train_cnt = dev_cnt = test_cnt = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

    @classmethod
    def make_stream(
        cls,
        documents: List[str],
        output: str,
        text_selector: Optional[str] = None,
        label_selector: Optional[str] = None,
        word_tokenizer: str = "noop",
        train_sample_rate: float = 0.0,
        dev_sample_rate: float = 0.0,
        test_sample_rate: float = 0.0,
        num_processes: int = 1,
        debug: bool = False,
        **kwargs: Any,
    ):
        # duck-typing to ensure that the tokenizer is valid; raises a KeyError if it is not
        TokenizerRegistry.get(word_tokenizer)

        with TemporaryDirectory() as tmpdir:
            all_paths = [p for p in chain.from_iterable(glob_path(p) for p in documents)]
            _, rel_paths = make_relative(all_paths)

            # we make temporary directories for the destination and metadata files;
            # we will combine data from the temporary directories into the final destination
            # after processing.
            dest_paths = [
                parent(join_path("", tmpdir, "destination", p))
                if p != "."
                else join_path("", tmpdir, "destination")
                for p in rel_paths
            ]
            meta_paths = [
                parent(join_path("", tmpdir, "metadata", p)) if p != "." else join_path("", tmpdir, "metadata")
                for p in rel_paths
            ]

            for p in dest_paths + meta_paths:
                mkdir_p(parent(p))

            # make the parallel processor here and immediately call it with the the options
            # for selecting text/labels, sample sizes, etc.
            # note that the number of processes used is capped at the number of files to process.
            cls(
                source_prefix=all_paths,
                destination_prefix=dest_paths,
                metadata_prefix=meta_paths,
                num_processes=min(num_processes, len(all_paths)),
                debug=debug,
            )(
                text_selector=text_selector,
                label_selector=label_selector,
                train_sample_rate=train_sample_rate,
                dev_sample_rate=dev_sample_rate,
                test_sample_rate=test_sample_rate,
                tokenizer_name=word_tokenizer,
                **kwargs,
            )

            # this is where we combine the splits into a single file for each split
            mkdir_p(output)
            combine_splits(sources=dest_paths, destination=output)


class FastTextDataConverter(BaseDataConverter):
    """Generate fasttext-compatible data from Dolma-style JSONL files. Uses functions to
    select which fields to use as text and labels. The text and labels are then preprocessed
    and written to separate files for training, development, and testing. The files are
    written with format `__label__<label> <text>`."""

    @classmethod
    def _make_text_fn(cls, text_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """
        Create a function that extracts text from the input document and preprocesses it (replace all
        whitespace with a single space and strip leading/trailing whitespace; optionally lowercase).
        """
        return make_selector(text_selector or "$.text")

    @classmethod
    def _make_label_fn(cls, label_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """
        Creates a label function that extracts labels from the input document and normalizes them
        to be fasttext-compatible. The labels are then written to a file with format `__label__<label_1>
        __label__<label_2> `.
        """

        # create a function to format the labels
        def _format_fn(raw_label: str) -> str:
            normalized_label = re.sub(r"[^a-z,]+", "_", raw_label.lower())
            return " ".join([f"__label__{lb.strip('_')}" for lb in normalized_label.split(",")]) + " "

        # if the label is provided directly, we just use it as-is
        if label_selector is not None and not label_selector.startswith("$"):
            # really just a closure to return the formatted label
            formatted_label = _format_fn(label_selector)
            return lambda _: formatted_label

        # otherwise, we have to use a selector to extract the label from each document
        _sel_fn = make_selector(label_selector or "$.source")

        # this is the label function that will be returned by this factory to select each label
        def _label_fn(
            doc: InputSpecWithMetadata, sel_fn: Callable = _sel_fn, format_fn: Callable = _format_fn
        ) -> str:
            raw_label = sel_fn(doc)
            return format_fn(raw_label)

        return _label_fn


class FastTextUnsupervisedDataConverter(FastTextDataConverter):
    """Converts Dolma-style JSONL files to fasttext-compatible data for unsupervised learning.
    Ignores labels and writes only the text to a single file."""

    @classmethod
    def _make_label_fn(cls, label_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """
        No-op since KenLM does not require labels.
        """
        label_fn = lambda _: ""  # type: ignore # noqa: E731
        return label_fn


class KenLMDataConverter(BaseDataConverter):
    @classmethod
    def _make_text_fn(cls, text_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """
        Create a function that extracts text from the input document;
        KenLM does not require any preprocessing of the text.
        """
        return make_selector(text_selector or "$.text")

    @classmethod
    def _make_label_fn(cls, label_selector: Union[str, None] = None) -> Callable[[InputSpecWithMetadata], str]:
        """
        No-op since KenLM does not require labels.
        """
        label_fn = lambda _: ""  # type: ignore # noqa: E731
        return label_fn
