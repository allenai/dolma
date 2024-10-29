import multiprocessing
from dataclasses import dataclass
from functools import partial
from typing import Callable
from urllib.parse import urlparse

import fsspec
import jq
import smart_open
from msgspec.json import Decoder
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass(frozen=True)
class Document:
    text: str
    label: str


def _label_selector_fn(row: dict, selector: Callable | None, label: str | None) -> str:
    if selector is not None:
        return str(selector(row).first())
    elif label is not None:
        return str(label)
    else:
        raise ValueError("Either `label` or `selector` must be provided")


def read_file(path: str, label: str | None = None, selector: str | None = None) -> list[Document]:
    label_fn = partial(_label_selector_fn, label=label, selector=(jq.compile(selector) if selector else None))

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

    @staticmethod
    def expand(data_config: "DataConfig", fs: fsspec.AbstractFileSystem | None = None) -> list["DataConfig"]:
        fs = fs or fsspec.get_filesystem_class(urlparse(data_config.path).scheme)()
        assert fs is not None, f"Could not determine filesystem for {data_config.path}"
        paths = [str(p) for p in fs.glob(data_config.path)] if "*" in data_config.path else [data_config.path]
        return [DataConfig(path=path, label=data_config.label, selector=data_config.selector) for path in paths]


class ClassifierDataset(Dataset):
    def __init__(
        self,
        configs: list[DataConfig],
        workers: int = 1,
    ):
        with multiprocessing.Pool(workers) as pool:
            expanded_configs: list[DataConfig] = [
                data_config
                for data_configs in tqdm(
                    pool.imap_unordered(DataConfig.expand, configs),
                    total=len(configs),
                    desc="Expanding configs",
                )
                for data_config in data_configs
            ]

        with multiprocessing.Pool(workers) as pool:
            self.documents = list(
                tqdm(
                    pool.imap_unordered(
                        lambda c: read_file(path=c.path, label=c.label, selector=c.selector),
                        expanded_configs
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
