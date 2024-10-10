import pytest

from dolma_decontamination.tasks.base import Row
from dolma_decontamination.tasks.formats import (
    mc_full_upper,
    mc_full_lower,
    mc_full_number,
    mc_short_upper,
    mc_short_lower,
    mc_short_number,
    mc_no_upper_double,
    mc_no_lower_double,
    mc_no_number_double
)


@pytest.fixture
def sample_data() -> Row:
    content ={
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "label": 1
    }
    return Row(row_id="1", dataset_label="test", content=content)


def test_mc_full_upper(sample_data):
    target = mc_full_upper()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "Question: What is the capital of France?\n"
        "A) London\n"
        "B) Paris\n"
        "C) Berlin\n"
        "D) Madrid\n"
        "Answer: B"
    )
    assert result[0].text == expected

def test_mc_full_lower(sample_data):
    target = mc_full_lower()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "Question: What is the capital of France?\n"
        "a) London\n"
        "b) Paris\n"
        "c) Berlin\n"
        "d) Madrid\n"
        "Answer: b"
    )
    assert result[0].text == expected

def test_mc_full_number(sample_data):
    target = mc_full_number()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "Question: What is the capital of France?\n"
        "1) London\n"
        "2) Paris\n"
        "3) Berlin\n"
        "4) Madrid\n"
        "Answer: 2"
    )
    assert result[0].text == expected

def test_mc_short_upper(sample_data):
    target = mc_short_upper()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "Q: What is the capital of France?\n"
        "A) London\n"
        "B) Paris\n"
        "C) Berlin\n"
        "D) Madrid\n"
        "A: B"
    )
    assert result[0].text == expected

def test_mc_short_lower(sample_data):
    target = mc_short_lower()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "Q: What is the capital of France?\n"
        "a) London\n"
        "b) Paris\n"
        "c) Berlin\n"
        "d) Madrid\n"
        "A: b"
    )
    assert result[0].text == expected

def test_mc_short_number(sample_data):
    target = mc_short_number()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "Q: What is the capital of France?\n"
        "1) London\n"
        "2) Paris\n"
        "3) Berlin\n"
        "4) Madrid\n"
        "A: 2"
    )
    assert result[0].text == expected

def test_mc_no_upper_double(sample_data):
    target = mc_no_upper_double()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "What is the capital of France?\n"
        "(A) London\n"
        "(B) Paris\n"
        "(C) Berlin\n"
        "(D) Madrid\n"
        "B"
    )
    assert result[0].text == expected

def test_mc_no_lower_double(sample_data):
    target = mc_no_lower_double()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "What is the capital of France?\n"
        "(a) London\n"
        "(b) Paris\n"
        "(c) Berlin\n"
        "(d) Madrid\n"
        "b"
    )
    assert result[0].text == expected

def test_mc_no_number_double(sample_data):
    target = mc_no_number_double()
    result = list(target.select(sample_data))
    assert len(result) == 1
    expected = (
        "What is the capital of France?\n"
        "[1] London\n"
        "[2] Paris\n"
        "[3] Berlin\n"
        "[4] Madrid\n"
        "2"
    )
    assert result[0].text == expected
