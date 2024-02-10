import random
import re
from contextlib import ExitStack
from io import TextIOWrapper
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Union

import msgspec
import smart_open
import tqdm

from ..core.data_types import InputSpecWithMetadata
from ..core.loggers import get_logger
from ..core.paths import glob_path, join_path, mkdir_p, split_path

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


def _make_fasttext_data_from_dict(
    paths: Dict[str, List[str]],
    text_selector: str,
) -> Generator[str, None, None]:
    """
    Generate fastText data from a dictionary of paths.

    Args:
        paths (Dict[str, List[str]]): A dictionary mapping labels to lists of file paths.
        text_selector (str, optional): JSONPath expression to select the text from the data. Defaults to "$.text".

    Yields:
        str: A string representing a single data instance in fastText format.

    """
    decoder = msgspec.json.Decoder(InputSpecWithMetadata)
    fn = _make_selector(text_selector)

    for label, label_paths in paths.items():
        label_formatted = " ".join([f"__label__{lb}" for lb in label.split(",")])
        for path in chain.from_iterable(glob_path(p) for p in label_paths):
            with smart_open.open(path, "rt") as f:
                cnt = 0
                for line in f:
                    data = decoder.decode(line)
                    text = fn(data)
                    yield f"{label_formatted} {text}"
                    cnt += 1
            LOGGER.info(f"Processed {cnt} lines from {path} with label {label}")


def _make_fasttext_data_from_label(
    paths: List[str],
    label_selector: str,
    text_selector: str,
) -> Generator[str, None, None]:
    """
    Generate formatted data from labeled input files.

    Args:
        paths (List[str]): List of file paths to read the labeled data from.
        label_selector (str, optional): JSONPath expression to select the label from the data. Defaults to "$.source".
        text_selector (str, optional): JSONPath expression to select the text from the data. Defaults to "$.text".

    Yields:
        str: Formatted data generated from the labeled input files.
    """
    decoder = msgspec.json.Decoder(InputSpecWithMetadata)
    text_fn = _make_selector(text_selector)
    label_fn = _make_selector(label_selector)

    for path in chain.from_iterable(glob_path(p) for p in paths):
        cnt = 0
        with smart_open.open(path, "rt") as f:
            for line in f:
                data = decoder.decode(line)
                text = text_fn(data)
                label = label_fn(data)
                label_formatted = " ".join([f"__label__{lb}" for lb in label.split(",")])
                yield f"{label_formatted} {text}"
                cnt += 1

        LOGGER.info(f"Processed {cnt} lines from {path} with label from {label_selector}")


class _PartitionedFileWriter:
    file_: Optional[TextIOWrapper]
    MAX_COUNT = 100000

    def __init__(self, path: str, max_size: Optional[int] = None, mode="wt", encoding="utf-8", **open_kwargs):
        self.prot, (*self.parts, fn) = split_path(path)
        self.base_fn, *_exts = fn.split(".")
        self.ext = ("." + ".".join(_exts)) if _exts else ""
        self.max_size = max_size
        self.file_ = None
        self.file_cnt = 0
        self.open_kwargs = {**open_kwargs, "mode": mode, "encoding": encoding}

    def _new_file(self):
        if self.file_:
            LOGGER.info(f"Closing file {self.file_.name}")
            self.file_.close()

        new_path = join_path(self.prot, *self.parts, f"{self.base_fn}-{self.file_cnt:05d}{self.ext}")
        self.file_ = smart_open.open(new_path, **self.open_kwargs)
        LOGGER.info(f"Created {self.file_.name}")
        self.file_cnt += 1

    def __enter__(self):
        self._new_file()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file_ and self.file_.close()  # pyright: ignore
        self.file_ = None

    def write(self, line: str) -> bool:
        if self.file_ is None:
            raise RuntimeError("File is not open")

        if self.max_size and self.file_.tell() >= self.max_size:
            self._new_file()

        self.file_.write(line)
        return True


def _write_data(
    data_it: Generator[str, None, None],
    dest: str,
    train_sample: float = 1.0,
    dev_sample: float = 0.0,
    test_sample: float = 0.0,
    max_size: Optional[int] = None,
    seed: int = 0,
):
    # setup: set the seed for future random calls and create the destination directory
    random.seed(seed)
    mkdir_p(dest)

    # Check if all sample rates are valid
    sample_sum = 0.0
    for sample in [train_sample, dev_sample, test_sample]:
        if sample < 0 or sample > 1:
            raise ValueError("Sample sizes must be between 0 and 1")
        sample_sum += sample
    if sample_sum == 0:
        raise ValueError("At least one sample size must be greater than 0")
    elif sample_sum > 1:
        raise ValueError("The sum of sample sizes must be less than or equal to 1")

    with ExitStack() as stack:
        train_file = (
            stack.enter_context(
                _PartitionedFileWriter(path=f"{dest}/train.txt", max_size=max_size, mode="wt", encoding="utf-8")
            )
            if train_sample > 0
            else None
        )
        dev_file = (
            stack.enter_context(
                _PartitionedFileWriter(path=f"{dest}/dev.txt", max_size=max_size, mode="wt", encoding="utf-8")
            )
            if dev_sample > 0
            else None
        )
        test_file = (
            stack.enter_context(
                _PartitionedFileWriter(path=f"{dest}/test.txt", max_size=max_size, mode="wt", encoding="utf-8")
            )
            if test_sample > 0
            else None
        )

        # keep a bunch of progress bars to track the progress of the data writing
        train_pbar = stack.enter_context(tqdm.tqdm(desc="Train data", unit=" samples", unit_scale=True))
        dev_pbar = stack.enter_context(tqdm.tqdm(desc="Dev data", unit=" samples", unit_scale=True))
        test_pbar = stack.enter_context(tqdm.tqdm(desc="Test data", unit=" samples", unit_scale=True))

        # when writing to the files, we need to check if the writer exists before writing
        for line in data_it:
            if ((r := random.random()) < train_sample) and train_file and train_file.write(line):
                train_pbar.update(1)
            elif (r < train_sample + dev_sample) and dev_file and dev_file.write(line):
                dev_pbar.update(1)
            elif test_file and test_file.write(line):
                test_pbar.update(1)


def make_fasttext_data(
    paths: Union[List[str], Dict[str, List[str]]],
    dest: str,
    text_selector: Optional[str] = None,
    label_selector: Optional[str] = None,
    train_sample: float = 1.0,
    dev_sample: float = 0.0,
    test_sample: float = 0.0,
    max_size: Optional[int] = None,
    seed: int = 0,
):
    """
    Generate fastText data from labeled input files or a dictionary of paths.

    Args:
        paths (Union[List[str], Dict[str, List[str]]]): Either a list of file paths to read the labeled data from, or a dictionary mapping labels to lists of file paths.
        dest (str): The destination directory to write the generated data to.
        text_selector (str, optional): JSONPath expression to select the text from the data. Defaults to "$.text".
        label_selector (str, optional): JSONPath expression to select the label from the data. Defaults to "$.source".
        train_sample (float, optional): The proportion of the data to use for training. Defaults to 1.0.
        dev_sample (float, optional): The proportion of the data to use for development. Defaults to 0.0.
        test_sample (float, optional): The proportion of the data to use for testing. Defaults to 0.0.
        max_size (int, optional): The maximum size of each partitioned file. Defaults to None.
        seed (int, optional): The seed for the random number generator. Defaults to 0.

    """
    # Create the data iterator; depending on the type of `paths`, this will be a different function
    if isinstance(paths, dict):
        text_selector = text_selector or "$.text"
        data_it = _make_fasttext_data_from_dict(paths=paths, text_selector=text_selector)
    elif isinstance(paths, list):
        label_selector = label_selector or "$.source"
        text_selector = text_selector or "$.text"
        data_it = _make_fasttext_data_from_label(paths, label_selector=label_selector, text_selector=text_selector)
    else:
        raise TypeError("`paths` must be a List[str] or a Dict[str, List[str]]")

    _write_data(
        data_it,
        dest,
        train_sample=train_sample,
        dev_sample=dev_sample,
        test_sample=test_sample,
        max_size=max_size,
        seed=seed,
    )
