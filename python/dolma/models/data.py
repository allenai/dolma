from contextlib import ExitStack
import re
from typing import Generator, List, Tuple, Union, Dict, overload, Callable, TypeVar
import msgspec

import smart_open
from ..core.data_types import InputSpecWithMetadata


def selector(jsonpath: str) -> Callable:
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
        prev = selector(jsonpath[m.end():])
        return lambda x: prev(x[index])

    elif m := re.match(r"^\.([a-zA-Z_][a-zA-Z0-9_]*)", jsonpath):
        # This is a key
        key = m.group(1)
        prev = selector(jsonpath[m.end():])
        return lambda x: prev(x[key] if isinstance(x, dict) else getattr(x, key))
    elif not jsonpath.strip():
        return lambda x: x

    raise ValueError(f"Invalid JSONPath: {jsonpath}")


def make_ft_data_from_dict(
    paths: Dict[str, List[str]],
    text_selector: str = "$.text",
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
    fn = selector(text_selector)

    for label, label_paths in paths.items():
        label_formatted = " ".join([f"__label__{lb}" for lb in label.split(",")])
        for path in label_paths:

            with smart_open.open(path, "rt") as f:
                for line in f:
                    data = decoder.decode(line)
                    text = fn(data)
                    yield f"{label_formatted} {text}\n"


def make_fn_data_from_label(
    paths: List[str],
    label_selector: str = "$.source",
    text_selector: str = "$.text",
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
    text_fn = selector(text_selector)
    label_fn = selector(label_selector)

    for path in paths:
        with smart_open.open(path, "rt") as f:
            for line in f:
                data = decoder.decode(line)
                text = text_fn(data)
                label = label_fn(data)
                label_formatted = " ".join([f"__label__{lb}" for lb in label.split(",")])
                yield f"{label_formatted} {text}\n"


if __name__ == "__main__":
    # d = {'a': [{'b': 1}, {'c': [2, {'d': 3}], 'e': 4}, {'f': 5}], 'g': 6}

    # assert selector("$.a")(d) == d['a']
    # assert selector("$.a[1].c")(d) == d['a'][1]['c']
    # assert selector("$.a[1].c[1]")(d) == d['a'][1]['c'][1]
    # assert selector("$.a[1].c[1].d")(d) == d['a'][1]['c'][1]['d']
    # assert selector("$.a[1].e")(d) == d['a'][1]['e']
    # assert selector("$.g")(d) == d['g']

    make_ft_data_from_dict(
        paths={
            "positive": ["tests/data/multiple_files/cc_en_head-0091.jsonl.gz"],
            "negative": ["tests/data/multiple_files/cc_en_head-0174.jsonl.gz"],
        },
        # destination="/tmp/fasttext.txt",
    )
