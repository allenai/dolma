from .base import Task, Dataset, Target

QA_FORMATS = [
    ('"Question: " + .question + "\nAnswer: " + .answer', "qa_full_newline"),
    ('"Q: " + .question + "\nA: " + .answer', "qa_short_newline"),
    ('.question + "\n" + .answer', "qa_noprompt_newline"),
    ('"Question: " + .question + "\nAnswer: " + .answer', "qa_full_space"),
    ('"Q: " + .question + "\nA: " + .answer', "qa_short_space"),
    ('.question + " " + .answer', "qa_noprompt_space"),
]

def squad() -> Task:
    datasets = [Dataset(path="rajpurkar/squad", split="validation")]
    targets = [Target(selector=".context", label="context")]

    for qa_format, qa_label in QA_FORMATS:
        selector = "{question, answer: .answers.text[]} | " + qa_format
        targets.append(Target(selector=selector, label=qa_label))

    return Task(name="squad", datasets=datasets, targets=targets)


def coqa() -> Task:
    datasets = []
    for split in ["train", "validation"]:
        datasets.append(Dataset(path="stanfordnlp/coqa", split=split, id_selector=None))
    targets = [Target(selector=".story", label="story"),]

    for qa_format, qa_label in QA_FORMATS:
        selector = "{question: .questions[], answer: .answers.input_text[]} | " + qa_format
        targets.append(Target(selector=selector, label=qa_label))

    return Task(name="coqa", datasets=datasets, targets=targets)


def jeopardy() -> Task:
    ...
