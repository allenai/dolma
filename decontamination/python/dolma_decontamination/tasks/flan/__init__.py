"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Commonsense reasoning tasks.

Author: Luca Soldaini (@soldni)
Email: luca@soldaini.net
"""

from ..base import Task, Dataset, Target, Row, TargetOutput
from ..registry import register_task
from . import commonsense, nli
import string
from typing import Iterable


__all__ = [
    "commonsense",
    "nli",
    "flan",
]


class FlanTarget(Target):
    expression: str = '.'
    label: str = 'flan'

    def select(self, row: Row) -> Iterable[TargetOutput]:
        last_char = row.content["inputs"][-1]
        if last_char in string.whitespace:
            formatted = f'{row.content["inputs"]}{row.content["targets"]}'
        elif last_char in string.punctuation:
            formatted = f'{row.content["inputs"]} {row.content["targets"]}'
        else:
            formatted = f'{row.content["inputs"]}\n{row.content["targets"]}'

        yield TargetOutput(target_id="0", text=formatted, label=self.label)


@register_task()
def flan() -> Task:
    datasets = [
        Dataset(
            "json",
            split="train",
            data_files="s3://ai2-llm/pretraining-data/sources/Muennighoff_flan/raw/validation/*.jsonl"),
        Dataset(
            "json",
            split="train",
            data_files="s3://ai2-llm/pretraining-data/sources/Muennighoff_flan/raw/test/*.jsonl"
        ),
    ]
    return Task(name="flan", datasets=datasets, targets=[FlanTarget()])
