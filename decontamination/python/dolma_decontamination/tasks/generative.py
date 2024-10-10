from .base import Task, Dataset, Target

QA_FORMATS = [
    Target('"Question: " + .question + "\nAnswer: " + .answer', "qa_full_newline"),
    Target('"Q: " + .question + "\nA: " + .answer', "qa_short_newline"),
    Target('.question + "\n" + .answer', "qa_noprompt_newline"),
    Target('"Question: " + .question + "\nAnswer: " + .answer', "qa_full_space"),
    Target('"Q: " + .question + "\nA: " + .answer', "qa_short_space"),
    Target('.question + " " + .answer', "qa_noprompt_space"),
]


def squad() -> Task:
    datasets = [Dataset(path="rajpurkar/squad", split="validation", id_selector=".id")]

    targets = [Target(expression=".context", label="context")]
    base_target = Target("{question, answer: .answers.text[]}")
    targets.extend([base_target + qa_format for qa_format in QA_FORMATS])

    return Task(name="squad", datasets=datasets, targets=targets)


def coqa() -> Task:
    datasets: list[Dataset] = [
        Dataset(path="stanfordnlp/coqa", split="validation", id_selector=".id"),
        Dataset(path="stanfordnlp/coqa", split="test", id_selector=".id"),
    ]
    targets = [Target(expression=".story", label="story")]
    base_target = Target("{question, answer: .answers.input_text[]}")
    targets.extend([base_target + qa_format for qa_format in QA_FORMATS])


    return Task(name="coqa", datasets=datasets, targets=targets)


def jeopardy() -> Task:
    ...
