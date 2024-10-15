"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Commonsense reasoning tasks.

Author: Luca Soldaini (@soldni)
Email: luca@soldaini.net
"""

from ..base import Task, Dataset, Target
from ..registry import register_task
from ..formats import (
    mc_full_upper,
    mc_full_lower,
    mc_full_number,
    mc_short_upper,
    mc_short_lower,
    mc_short_number,
    mc_no_upper_double,
    mc_no_lower_double,
    mc_no_number_double,
)


@register_task()
def copa() -> Task:
    datasets = [
        Dataset("aps/super_glue", name="copa", split="validation", id_selector=".idx"),
        Dataset("aps/super_glue", name="copa", split="test", id_selector=".idx"),
    ]
    input_map_target = Target('{question: .premise, choices: [.choice1, .choice2], label: .label}')
    targets = [
        input_map_target | mc_full_upper(),
        input_map_target | mc_full_lower(),
        input_map_target | mc_full_number(),
        input_map_target | mc_short_upper(),
        input_map_target | mc_short_lower(),
        input_map_target | mc_short_number(),
        input_map_target | mc_no_upper_double(),
        input_map_target | mc_no_lower_double(),
        input_map_target | mc_no_number_double(),
    ]
    return Task(name="copa", datasets=datasets, targets=targets)


@register_task()
def hellaswag() -> Task:
    datasets = [
        Dataset("rowan/hellaswag", split="validation", id_selector=".idx", trust_remote_code=True),
        Dataset("rowan/hellaswag", split="test", id_selector=".idx", trust_remote_code=True),
    ]
    correct_generations = Target(
        '.ctx_a + " " + (.ctx_b[0:1] | ascii_upcase) + .ctx_b[1:] + " " + .endings[.label | tonumber]',
        label="correct_generations"
    )
    return Task(name="hellaswag", datasets=datasets, targets=[correct_generations])


@register_task()
def piqa() -> Task:
    datasets = [
        Dataset("ybisk/piqa", split="validation", trust_remote_code=True, id_selector=".row_id"),
        Dataset("ybisk/piqa", split="test", trust_remote_code=True, id_selector=".row_id"),
    ]
    correct_generations = Target(
        '.goal + " " + [.sol1, .sol2][.label | tonumber]',
        label="correct_generations"
    )
    return Task(name="piqa", datasets=datasets, targets=[correct_generations])
