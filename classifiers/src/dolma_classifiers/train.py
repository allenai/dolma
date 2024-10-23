import multiprocessing
from dataclasses import dataclass
from typing import Callable, Generator, NamedTuple
from urllib.parse import urlparse

import fsspec
import jq
import smart_open
import torch
from msgspec.json import Decoder
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass(frozen=True)
class Document:
    text: str
    label: str


def read_file(path: str, label: str | None = None, selector: str | None = None) -> list[Document]:
    if selector is not None:
        compiled_selector = jq.compile(selector)
        label_fn = lambda row: str(compiled_selector.input(row).first())
    elif label is not None:
        label_fn = lambda row: str(label)
    else:
        raise ValueError("Either `label` or `selector` must be provided")

    decoder = Decoder()
    documents = []

    with smart_open.open(path) as f:
        for line in f:
            row = decoder.decode(line)
            label = label_fn(row)
            documents.append(Document(text=row["text"], label=label))

    return documents


@dataclass(frozen=True)
class DataConfig:
    path: str
    label: str | None = None
    selector: str | None = None

    def expand(self, fs: fsspec.AbstractFileSystem | None = None) -> list["DataConfig"]:
        fs = fs or fsspec.get_filesystem_class(urlparse(self.path).scheme)()
        paths = [str(p) for p in fs.glob(self.path)] if "*" in self.path else [self.path]
        return [DataConfig(path=path, label=self.label, selector=self.selector) for path in paths]


class ClassifierDataset(Dataset):
    def __init__(
        self,
        configs: list[DataConfig],
        workers: int = 1,
    ):
        with multiprocessing.Pool(workers) as pool:
            expanded_configs = list(
                tqdm(
                    pool.imap_unordered(lambda c: c.expand(), configs),
                    total=len(configs),
                    desc="Expanding configs",
                )
            )

        with multiprocessing.Pool(workers) as pool:
            self.documents = list(
                tqdm(
                    pool.imap_unordered(
                        lambda c: read_file(path=c.path, label=c.label, selector=c.selector), expanded_configs
                    ),
                    total=len(expanded_configs),
                    desc="Reading files",
                )
            )

        print(f"Read {len(self.documents)} documents from {len(expanded_configs)} configs")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]
