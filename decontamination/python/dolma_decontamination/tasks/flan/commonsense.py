"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Commonsense reasoning tasks.

Author: Luca Soldaini (@soldni)
Email: luca@soldaini.net
"""

from ..base import Task, Dataset, Target
from typing import Literal



def copa() -> Task:
    datasets = [
        Dataset("aps/super_glue", name="copa", split="validation"),
        Dataset("aps/super_glue", name="copa", split="test"),
    ]
