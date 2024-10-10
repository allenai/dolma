"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Natural Language Inference (NLI) tasks.

Author: Luca Soldaini (@soldni)
Email: luca@soldaini.net
"""

from ..base import Task, Dataset, Target


ENTAILMENT_FORMATS = [
    ('"Premise:\n" + .premise + "\nHypothesis:\n" + .hypothesis + "\n" + .label', "ent_full_newline"),
    ('"Premise: " + .premise + "\nHypothesis: " + .hypothesis + "\n" + .label', "ent_part_newline"),
    ('"Sentence 1:\n" + .premise + "\nSentence 2:\n" + .hypothesis + "\n" + .label', "ent_full_newline"),
    ('"Sentence 1: " + .premise + "\nSentence 2: " + .hypothesis + "\n" + .label', "ent_part_newline"),
    ('.premise + "\n" + .hypothesis + "\n" + .label', "ent_noprompt_newline"),
    ('.premise + " " + .hypothesis + " " + .label', "ent_noprompt_space"),
]


def anli() -> Task:
    datasets = [
        Dataset(path="facebook/anli", split="dev_r1", id_selector=".uid"),
        Dataset(path="facebook/anli", split="dev_r2", id_selector=".uid"),
        Dataset(path="facebook/anli", split="dev_r3", id_selector=".uid"),
        Dataset(path="facebook/anli", split="test_r1", id_selector=".uid"),
        Dataset(path="facebook/anli", split="test_r2", id_selector=".uid"),
        Dataset(path="facebook/anli", split="test_r3", id_selector=".uid"),
    ]
    base_selector = '{premise: .premise, hypothesis: .hypothesis, label: (if .label == 0 then "entailment" elif .label == 1 then "neutral" else "contradiction" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="anli", datasets=datasets, targets=targets)


def cb() -> Task:
    datasets = [
        Dataset(path="aps/super_glue", name="cb", split="validation", id_selector=".idx", trust_remote_code=True),
        Dataset(path="aps/super_glue", name="cb", split="test", id_selector=".idx", trust_remote_code=True)
    ]
    base_selector = '{premise: .premise, hypothesis: .hypothesis, label: (if .label == 0 then "entailment" elif .label == 1 then "contradiction" else "neutral" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="cb", datasets=datasets, targets=targets)


def mnli() -> Task:

    datasets: list[Dataset] = []
    for split in ["validation", "test"]:
        for name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            datasets.append(Dataset(path="nyu-mll/glue", name=name, split=split, id_selector=".idx"))

    base_selector = '{premise: .premise, hypothesis: .hypothesis, label: (if .label == 0 then "entailment" elif .label == 1 then "neutral" else "contradiction" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="mnli", datasets=datasets, targets=targets)


def qnli() -> Task:
    datasets = [
        Dataset(path="nyu-mll/glue", name="qnli", split="validation"),
        Dataset(path="nyu-mll/glue", name="qnli", split="test"),
    ]
    base_selector = '{question: .question, sentence: .sentence, label: (if .label == 0 then "entailment" else "not_entailment" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="qnli", datasets=datasets, targets=targets)


def rte() -> Task:
    datasets = [
        Dataset(path="nyu-mll/glue", name="rte", split="validation"),
        Dataset(path="nyu-mll/glue", name="rte", split="test"),
    ]
    base_selector = '{premise: .sentence1, hypothesis: .sentence2, label: (if .label == 0 then "entailment" else "not_entailment" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="rte", datasets=datasets, targets=targets)


def snli() -> Task:
    datasets: list[Dataset] = [
        Dataset(path="stanfordnlp/snli", split="validation"),
        Dataset(path="stanfordnlp/snli", split="test"),
    ]
    base_selector = '{premise: .premise, hypothesis: .hypothesis, label: (if .label == 0 then "entailment" elif .label == 1 then "neutral" else "contradiction" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="snli", datasets=datasets, targets=targets)


def wnli() -> Task:
    datasets = [
        Dataset(path="nyu-mll/glue", name="wnli", split="validation"),
        Dataset(path="nyu-mll/glue", name="wnli", split="test"),
    ]
    base_selector = '{sentence1: .sentence1, sentence2: .sentence2, label: (if .label == 0 then "entailment" else "not_entailment" end)}'

    targets: list[Target] = []

    for format_, label in ENTAILMENT_FORMATS:
        selector = base_selector + " | " + format_
        targets.append(Target(selector=selector, label=label))

    return Task(name="wnli", datasets=datasets, targets=targets)
