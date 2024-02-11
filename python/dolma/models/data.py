import multiprocessing
import random
import re
from contextlib import ExitStack
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    split_path,
)

LOGGER = get_logger(__name__)


def _make_selector(jsonpath: str) -> Callable:
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
        prev = _make_selector(jsonpath[m.end() :])
        return lambda x: prev(x[index])

    elif m := re.match(r"^\.([a-zA-Z_][a-zA-Z0-9_]*)", jsonpath):
        # This is a key
        key = m.group(1)
        prev = _make_selector(jsonpath[m.end() :])
        return lambda x: prev(x[key] if isinstance(x, dict) else getattr(x, key))
    elif not jsonpath.strip():
        return lambda x: x

    raise ValueError(f"Invalid JSONPath: {jsonpath}")


def _get_path_and_extension(path: str) -> Tuple[str, str]:
    prot, (*parts, filename) = split_path(path)
    base, *ext_parts = filename.split(".")
    ext = ("." + ".".join(ext_parts)) if ext_parts else ""
    return join_path(prot, *parts, base), ext


class FastTextDataWithSelector(BaseParallelProcessor):
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
    def _make_text_fn(cls, kwargs) -> Callable[[InputSpecWithMetadata], str]:
        _text_fn = _make_selector(kwargs.get("text_selector") or "$.text")
        _lowercase = kwargs.get("lowercase", False)

        def preprocess_text(
            doc: InputSpecWithMetadata, text_fn: Callable = _text_fn, lowercase: bool = _lowercase
        ) -> str:
            # fasttext expects just whitespace separated words, so we replace all
            # non-whitespace characters with whitespace. Read more at
            # https://github.com/facebookresearch/fastText/blob/main/python/README.md
            # under "IMPORTANT: Preprocessing data / encoding conventions"
            text = re.sub(r"\s+", " ", text_fn(doc)).strip()
            if lowercase:
                return text.lower()
            return text

        return preprocess_text

    @classmethod
    def _make_label_fn(cls, kwargs) -> Callable[[InputSpecWithMetadata], str]:
        label_fn = _make_selector(kwargs.get("label_selector") or "$.source")

        def _preprocess_label(doc: InputSpecWithMetadata, label_fn: Callable = label_fn) -> str:
            # we need to normalize labels to lowercase and replace non-alpha characters
            label = label_fn(doc)
            label = re.sub(r"[^a-z,]+", "_", label.lower())
            return " ".join([f"__label__{lb.strip('_')}" for lb in label.split(",")]) + " "

        return _preprocess_label

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        train_sample = kwargs.get("train_sample", 1.0)
        dev_sample = train_sample + kwargs.get("dev_sample", 0.0)
        test_sample = dev_sample + kwargs.get("test_sample", 0.0)

        # check if the sample sizes are valid
        assert 0 <= train_sample <= 1, "train_sample must be between 0 and 1"
        assert train_sample <= dev_sample <= 1, "dev_sample must be between 0 and 1"
        assert dev_sample <= test_sample <= 1, "test_sample must be between 0 and 1"

        text_fn = cls._make_text_fn(kwargs)
        label_fn = cls._make_label_fn(kwargs)

        decoder = msgspec.json.Decoder(InputSpecWithMetadata)
        train_cnt = dev_cnt = test_cnt = 0
        base_name, ext = _get_path_and_extension(destination_path)
        update_interval = 1

        with ExitStack() as stack:
            readfile = stack.enter_context(smart_open.open(source_path, "rt"))
            train_writefile = stack.enter_context(smart_open.open(f"{base_name}-train{ext}", "wt"))
            dev_writefile = stack.enter_context(smart_open.open(f"{base_name}-dev{ext}", "wt"))
            test_writefile = stack.enter_context(smart_open.open(f"{base_name}-test{ext}", "wt"))

            for i, line in enumerate(readfile):
                if (p := random.random()) > test_sample:
                    continue

                data = decoder.decode(line)
                text = text_fn(data)
                label = label_fn(data)

                if p <= train_sample:
                    train_writefile.write(label + text + "\n")
                    train_cnt += 1
                elif p <= dev_sample:
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
    def combine_splits(cls, sources: List[str], destination: str):
        for split in ("train", "dev", "test"):
            with smart_open.open(join_path("", destination, f"{split}.txt"), "wt") as wf:
                for path in chain.from_iterable(glob_path(f"{d}/*{split}*") for d in set(sources)):
                    with smart_open.open(path, "rt") as rf:
                        wf.write(rf.read())

    @classmethod
    def make(
        cls,
        paths: List[str],
        dest: str,
        text_selector: Optional[str] = None,
        label_selector: Optional[str] = None,
        train_sample: float = 1.0,
        dev_sample: float = 0.0,
        test_sample: float = 0.0,
        num_processes: int = 1,
        lowercase: bool = False,
        debug: bool = False,
    ):
        with TemporaryDirectory() as tmpdir:
            all_paths = [p for p in chain.from_iterable(glob_path(p) for p in paths)]
            _, rel_paths = make_relative(all_paths)
            dest_paths = [parent(join_path("", tmpdir, "destination", p)) for p in rel_paths]
            meta_paths = [parent(join_path("", tmpdir, "metadata", p)) for p in rel_paths]
            for p in dest_paths + meta_paths:
                mkdir_p(parent(p))

            cls(
                source_prefix=all_paths,
                destination_prefix=dest_paths,
                metadata_prefix=meta_paths,
                num_processes=min(num_processes, len(all_paths)),
                debug=debug,
            )(
                text_selector=text_selector,
                label_selector=label_selector,
                train_sample=train_sample,
                dev_sample=dev_sample,
                test_sample=test_sample,
                lowercase=lowercase,
            )
            mkdir_p(dest)
            cls.combine_splits(sources=dest_paths, destination=dest)


class FastTextDataFromDict(FastTextDataWithSelector):
    @classmethod
    def _make_label_fn(cls, kwargs) -> Callable[[InputSpecWithMetadata], str]:
        label = kwargs.get("label_selector")
        assert label is not None, "label_selector must be provided"

        label = re.sub(r"[^a-z,]+", "_", label.lower())
        label = " ".join([f"__label__{lb.strip('_')}" for lb in label.split(",")]) + " "
        return lambda _: label  # type: ignore  # noqa: E731

    @classmethod
    def make(  # type: ignore
        cls,
        paths: Dict[str, List[str]],
        dest: str,
        train_sample: float = 1.0,
        dev_sample: float = 0.0,
        test_sample: float = 0.0,
        text_selector: Optional[str] = None,
        lowercase: bool = False,
        num_processes: int = 1,
        debug: bool = False,
    ):
        with TemporaryDirectory() as tmpdir:
            for label, source_paths in paths.items():
                super().make(
                    paths=source_paths,
                    dest=f"{tmpdir}/{label}",
                    train_sample=train_sample,
                    dev_sample=dev_sample,
                    test_sample=test_sample,
                    text_selector=text_selector,
                    label_selector=label,
                    lowercase=lowercase,
                    num_processes=num_processes,
                    debug=debug,
                )
            cls.combine_splits(sources=[f"{tmpdir}/{label}" for label in paths], destination=dest)
