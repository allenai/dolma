"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Commonsense reasoning tasks.

Author: Luca Soldaini (@soldni)
Email: luca@soldaini.net
"""

from ..base import Task, Dataset, Target
from ..formats import (
    mc_full_upper,
    mc_full_number,
    mc_short_upper,
    mc_short_number,
    mc_short_upper_number,
    mc_full_lower_number,
)



def copa() -> Task:
    datasets = [
        Dataset("aps/super_glue", name="copa", split="validation"),
        Dataset("aps/super_glue", name="copa", split="test"),
    ]
    rename_target = Target(
        '{question: .premise, }'
    )
