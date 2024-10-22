from typing import Generator, Callable
import jq
import smart_open
import fsspec
from msgspec.json import Decoder
from msgspec import Struct


class Document(Struct):
    text: str
    label: str


def read_file(path: str, label: str | None = None, selector: str | None = None) -> Generator[Document, None, None]:
    if selector is not None:
        compiled_selector = jq.compile(selector)
        label_fn = lambda row: str(compiled_selector.input(row).first())
    elif label is not None:
        label_fn = lambda row: str(label)
    else:
        raise ValueError("Either `label` or `selector` must be provided")

    decoder = Decoder()

    with smart_open.open(path) as f:
        for line in f:
            row = decoder.decode(line)
            label = label_fn(row)
            yield Document(text=row["text"], label=label)
