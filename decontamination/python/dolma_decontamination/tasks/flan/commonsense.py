"""Tasks from the FLAN paper: https://arxiv.org/pdf/2109.01652
Commonsense reasoning tasks.

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
