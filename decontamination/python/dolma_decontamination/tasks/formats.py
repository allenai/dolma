from .base import Target
from typing import Literal


__all__ = [
    "mc_full_upper",
    "mc_full_lower",
    "mc_full_number",
    "mc_short_upper",
    "mc_short_lower",
    "mc_short_number",
    "mc_no_upper_double",
    "mc_no_lower_double",
    "mc_no_number_double",
]


def _make_multiple_choice(
    label: str,
    question_prefix: str,
    answer_prefix: str,
    choices_prefix: str,
    choices_format: Literal["uppercase", "lowercase", "number"],
    choices_suffix: str,
    element_separator: str,
) -> Target:
    if choices_format == "uppercase":
        choices_format = "[.key + 65] | implode"
    elif choices_format == "lowercase":
        choices_format = "[.key + 97] | implode"
    elif choices_format == "number":
        choices_format = "(.key + 1) | tostring"
    else:
        raise ValueError(f"Invalid choices format: {choices_format}")

    fmt = (
        Target(f'"{question_prefix}" + .question + "{element_separator}"') +
        (
            Target(".choices") |
            Target("to_entries") |
            Target(f'map("{choices_prefix}\({choices_format}){choices_suffix}\( .value )")') |
            Target(f'join("{element_separator}")')
        ) +
        Target(f'"{element_separator}{answer_prefix}"') +
        Target(f'{{key:.label}} | {choices_format}', label=label)
    )
    return fmt


def mc_full_upper() -> Target:
    return _make_multiple_choice(
        label="mc_full_upper",
        question_prefix="Question: ",
        answer_prefix="Answer: ",
        choices_prefix="",
        choices_format="uppercase",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_full_lower() -> Target:
    return _make_multiple_choice(
        label="mc_full_lower",
        question_prefix="Question: ",
        answer_prefix="Answer: ",
        choices_prefix="",
        choices_format="lowercase",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_full_number() -> Target:
    return _make_multiple_choice(
        label="mc_full_number",
        question_prefix="Question: ",
        answer_prefix="Answer: ",
        choices_prefix="",
        choices_format="number",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_short_upper() -> Target:
    return _make_multiple_choice(
        label="mc_short_upper",
        question_prefix="Q: ",
        answer_prefix="A: ",
        choices_prefix="",
        choices_format="uppercase",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_short_lower() -> Target:
    return _make_multiple_choice(
        label="mc_short_lower",
        question_prefix="Q: ",
        answer_prefix="A: ",
        choices_prefix="",
        choices_format="lowercase",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_short_number() -> Target:
    return _make_multiple_choice(
        label="mc_short_number",
        question_prefix="Q: ",
        answer_prefix="A: ",
        choices_prefix="",
        choices_format="number",
        choices_suffix=") ",
        element_separator="\n"
    )

def mc_no_upper_double() -> Target:
    return _make_multiple_choice(
        label="mc_no_upper_double",
        question_prefix="",
        answer_prefix="",
        choices_prefix="(",
        choices_format="uppercase",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_no_lower_double() -> Target:
    return _make_multiple_choice(
        label="mc_no_lower_double",
        question_prefix="",
        answer_prefix="",
        choices_prefix="(",
        choices_format="lowercase",
        choices_suffix=") ",
        element_separator="\n"
    )


def mc_no_number_double() -> Target:
    return _make_multiple_choice(
        label="mc_no_number_double",
        question_prefix="",
        answer_prefix="",
        choices_prefix="[",
        choices_format="number",
        choices_suffix="] ",
        element_separator="\n"
    )
