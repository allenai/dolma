from typing import Iterable, Any
from msgspec import Struct
from .utils import JsonObjHasher
import jq
from datasets import load_dataset, Dataset as HFDataset
from collections.abc import Sequence


DATASET_SEPARATOR = ":"
DOCUMENT_SEPARATOR = ";"


def flatten(element: Any) -> Iterable[Any]:
    if isinstance(element, Sequence) and not isinstance(element, (str, bytes, bytearray)):
        for item in element:
            yield from flatten(item)
    else:
        yield element


class Row(Struct, frozen=True):
    row_id: str
    dataset_label: str
    content: dict

class Dataset(Struct, dict=True, omit_defaults=True):
    path: str
    split: str
    data_files: str | None = None
    name: str | None = None
    id_selector: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.id_selector is not None:
            self._compiled_id_selector = lambda row: str(jq.compile(self.id_selector).input(row).first())
        else:
            self._compiled_id_selector = JsonObjHasher()

    def label(self) -> str:
        return DATASET_SEPARATOR.join([self.path, self.name or "", self.split])

    @classmethod
    def from_label(cls, label: str) -> "Dataset":
        assert label.count(DATASET_SEPARATOR) == 3, f"Invalid dataset label: {label}"
        path, name, split, id_selector = label.split(DATASET_SEPARATOR)
        return cls(path=path, name=name or None, split=split, id_selector=id_selector or None)


    def load(self) -> Iterable[dict]:
        return load_dataset(
            path=self.path,
            name=self.name,
            split=self.split,
            data_files=self.data_files,
            trust_remote_code=self.trust_remote_code
        )

    def _row_id(self, row: dict) -> str:
        if (row_id := self._compiled_id_selector(row)) is None:
            raise ValueError(f"Row ID is None for row with keys: {', '.join(row.keys())}")
        return row_id

    def rows(self) -> Iterable[Row]:
        dataset_label = self.label()
        for row in self.load():
            yield Row(row_id=self._row_id(row), dataset_label=dataset_label, content=row)

    def __repr__(self) -> str:
        return self.label()

    def __str__(self) -> str:
        return self.label()


class TargetOutput(Struct, frozen=True):
    target_id: str
    text: str
    label: str


class Target(Struct, dict=True, omit_defaults=True):
    expression: str
    label: str | None = None

    def __post_init__(self):
        assert self.label is None or ":" not in self.label, f"Label cannot contain ':'"
        self._compiled_text_selector = jq.compile(self.expression)

    @staticmethod
    def join_labels(left: str | None, right: str | None) -> str | None:
        if right is None:
            return left
        elif left is None:
            return right
        else:
            return f"{left}:{right}"

    def __or__(self, other: "Target") -> "Target":
        new_expression = f"({self.expression}) | ({other.expression})"
        new_label = self.join_labels(self.label, other.label)
        return type(self)(expression=new_expression, label=new_label)

    def __ror__(self, other: "Target") -> "Target":
        return other | self

    def __add__(self, other: "Target") -> "Target":
        new_expression = f"({self.expression}) + ({other.expression})"
        new_label = self.join_labels(self.label, other.label)
        return type(self)(expression=new_expression, label=new_label)

    def __radd__(self, other: "Target") -> "Target":
        return self + other

    def __str__(self) -> str:
        return self.expression

    def __repr__(self) -> str:
        return f'{type(self).__name__}(expression="{self.expression!r}", label={self.label})'

    def select(self, row: Row) -> Iterable[TargetOutput]:
        assert self.label is not None, "Label is required for select() function"

        matches = {str(e) for e in flatten(self._compiled_text_selector.input(row.content).all())}
        for i, element in enumerate(matches):
            yield TargetOutput(target_id=f"{i}", text=element, label=self.label)



class DocMeta(Struct, frozen=True):
    dataset_id: str
    target_id: str
    dataset_label: str
    target_label: str

    def label(self) -> str:
        return DOCUMENT_SEPARATOR.join([self.dataset_id, self.target_id, self.target_label, self.dataset_label])


class Doc(Struct, frozen=True):
    text: str
    meta: DocMeta

    @classmethod
    def make(cls, target_output: TargetOutput, dataset_row: Row) -> "Doc":
        meta = DocMeta(
            dataset_id=dataset_row.row_id,
            target_id=target_output.target_id,
            dataset_label=dataset_row.dataset_label,
            target_label=target_output.label,
        )
        return cls(text=target_output.text, meta=meta)


class Task(Struct, frozen=True):
    name: str
    datasets: list[Dataset]
    targets: list[Target]

    def docs(self) -> Iterable[Doc]:
        for dataset in self.datasets:
            for row in dataset.rows():
                for target in self.targets:
                    for target_output in target.select(row):
                        yield Doc.make(target_output=target_output, dataset_row=row)
