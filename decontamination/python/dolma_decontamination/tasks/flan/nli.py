"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Natural Language Inference (NLI) tasks.

Author: Luca Soldaini (@soldni)
Email: luca@soldaini.net
"""

from ..base import Task, Dataset, Target
from ..registry import register_task


ENTAILMENT_FORMATS = [
    Target('"Premise: " + .premise + "\nHypothesis: " + .hypothesis + "\n" + .label', "ent_part_newline"),
    Target('"Sentence 1: " + .premise + "\nSentence 2: " + .hypothesis + "\n" + .label', "ent_part_newline"),
    Target('.premise + "\n" + .hypothesis + "\n" + .label', "ent_noprompt_newline"),
    Target('.premise + " " + .hypothesis + " " + .label', "ent_noprompt_space"),
]


@register_task()
def anli() -> Task:
    datasets = [
        Dataset(path="facebook/anli", split="dev_r1", id_selector=".uid"),
        Dataset(path="facebook/anli", split="dev_r2", id_selector=".uid"),
        Dataset(path="facebook/anli", split="dev_r3", id_selector=".uid"),
        Dataset(path="facebook/anli", split="test_r1", id_selector=".uid"),
        Dataset(path="facebook/anli", split="test_r2", id_selector=".uid"),
        Dataset(path="facebook/anli", split="test_r3", id_selector=".uid"),
    ]
    label_map_target = Target('(if .label == 0 then "entailment" elif .label == 1 then "neutral" else "contradiction" end)')
    base_target = Target(f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}')
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]

    return Task(name="anli", datasets=datasets, targets=targets)


@register_task()
def cb() -> Task:
    datasets = [
        Dataset(path="aps/super_glue", name="cb", split="validation", id_selector=".idx", trust_remote_code=True),
        Dataset(path="aps/super_glue", name="cb", split="test", id_selector=".idx", trust_remote_code=True)
    ]
    label_map_target = Target('(if .label == 0 then "entailment" elif .label == 1 then "contradiction" else "neutral" end)')
    base_target = Target(f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}')
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]

    return Task(name="cb", datasets=datasets, targets=targets)


@register_task()
def mnli() -> Task:
    datasets: list[Dataset] = []
    for split in ["validation", "test"]:
        for name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            datasets.append(Dataset(path="nyu-mll/glue", name=name, split=split, id_selector=".idx"))

    label_map_target = Target('(if .label == 0 then "entailment" elif .label == 1 then "neutral" else "contradiction" end)')
    base_target = Target(f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}')
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]

    return Task(name="mnli", datasets=datasets, targets=targets)


@register_task()
def qnli() -> Task:
    datasets = [
        Dataset(path="nyu-mll/glue", name="qnli", split="validation"),
        Dataset(path="nyu-mll/glue", name="qnli", split="test"),
    ]
    label_map_target = Target(
        '{question: .question, sentence: .sentence, '
        'label: (if .label == 0 then "entailment" else "not_entailment" end)}'
    )
    base_target = Target(
        f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}'
    )
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]
    return Task(name="qnli", datasets=datasets, targets=targets)


@register_task()
def rte() -> Task:
    datasets = [
        Dataset(path="nyu-mll/glue", name="rte", split="validation"),
        Dataset(path="nyu-mll/glue", name="rte", split="test"),
    ]
    label_map_target = Target(
        '{premise: .sentence1, hypothesis: .sentence2, '
        'label: (if .label == 0 then "entailment" else "not_entailment" end)}'
    )
    base_target = Target(
        f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}'
    )
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]
    return Task(name="rte", datasets=datasets, targets=targets)


@register_task()
def snli() -> Task:
    datasets: list[Dataset] = [
        Dataset(path="stanfordnlp/snli", split="validation"),
        Dataset(path="stanfordnlp/snli", split="test"),
    ]
    label_map_target = Target(
        '{premise: .premise, hypothesis: .hypothesis, '
        'label: (if .label == 0 then "entailment" elif .label == 1 then "neutral" else "contradiction" end)}'
    )
    base_target = Target(
        f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}'
    )
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]
    return Task(name="snli", datasets=datasets, targets=targets)


@register_task()
def wnli() -> Task:
    datasets = [
        Dataset(path="nyu-mll/glue", name="wnli", split="validation"),
        Dataset(path="nyu-mll/glue", name="wnli", split="test"),
    ]
    label_map_target = Target(
        '{sentence1: .sentence1, sentence2: .sentence2, '
        'label: (if .label == 0 then "entailment" else "not_entailment" end)}'
    )
    base_target = Target(
        f'{{premise: .premise, hypothesis: .hypothesis, label: {label_map_target}}}'
    )
    targets: list[Target] = [base_target | fmt for fmt in ENTAILMENT_FORMATS]
    return Task(name="wnli", datasets=datasets, targets=targets)
