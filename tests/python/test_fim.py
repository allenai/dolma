import re
import unittest
from uuid import uuid4

from dolma.core.data_types import DocumentWithMetadata
from dolma.data_transforms.fim import FillInMiddle, FIMConfig

FILE_SEPARATOR = "<|file_sep|>"
FIM_MIDDLE_TOKEN = "<|fim_mid|>"
FIM_PREFIX_TOKEN = "<|fim_prefix|>"
FIM_SUFFIX_TOKEN = "<|fim_suffix|>"


def mk_document(text: str) -> DocumentWithMetadata:
    return DocumentWithMetadata(source="somesource", version="1234", id=str(uuid4()), text=text, metadata={})


def mk_text(num_files: int) -> str:
    files = [CODE_FILE_1 if i % 2 == 0 else CODE_FILE_2 for i in range(num_files)]

    return FILE_SEPARATOR.join(files)


CODE_FILE_1 = """
def add_two_integers(a: int, b: int) -> int:
    sum = a + b
    return sum


def multiply_two_integers(a: int, b: int) -> int:
    product = a + b
    return product


def sum_and_multiple(a: int, b: int, c: int) -> int:
    sum = add_two_integers(a, b)
    product = multiply_two_integers(sum, c)
    return product
"""


CODE_FILE_2 = """
from typing import Any

import requests


def call_api(url: str, token: str, json: Dict[str, Any]) -> Dict[str, Any]:
    result = requests.post(
        url,
        headers={"x-api-token": token},
        json=json
    )

    return result.json()
"""


class TestFillInMiddle(unittest.TestCase):
    def test__fim_reordering_works(self) -> None:
        # First, Prefix-Suffix-Middle
        psm_config = FIMConfig(
            fim_rate=1.0,
            psm_spm_split=1.0,
            file_separator_token=FILE_SEPARATOR,
            fim_prefix_token=FIM_PREFIX_TOKEN,
            fim_middle_token=FIM_MIDDLE_TOKEN,
            fim_suffix_token=FIM_SUFFIX_TOKEN,
        )
        psm_fim = FillInMiddle(psm_config)
        original_text = mk_text(1)
        psm_reordered = psm_fim.perform_on_document(mk_document(original_text))
        final_text = psm_reordered.text
        prefix_plus_prefix_token, rest = final_text.split(FIM_SUFFIX_TOKEN)
        _, prefix = prefix_plus_prefix_token.split(FIM_PREFIX_TOKEN)
        suffix, middle = rest.split(FIM_MIDDLE_TOKEN)

        self.assertEqual(prefix + middle + suffix, original_text)

        # Next, Suffix-Prefix-Middle
        spm_config = FIMConfig(
            fim_rate=1.0,
            psm_spm_split=0,
            file_separator_token=FILE_SEPARATOR,
            fim_prefix_token=FIM_PREFIX_TOKEN,
            fim_middle_token=FIM_MIDDLE_TOKEN,
            fim_suffix_token=FIM_SUFFIX_TOKEN,
        )
        spm_fim = FillInMiddle(spm_config)
        original_text = mk_text(1)
        spm_reordered = spm_fim.perform_on_document(mk_document(original_text))
        final_text = spm_reordered.text
        suffix_plus_suffix_token, rest = final_text.split(FIM_PREFIX_TOKEN)
        _, suffix = suffix_plus_suffix_token.split(FIM_SUFFIX_TOKEN)
        prefix, middle = rest.split(FIM_MIDDLE_TOKEN)

        self.assertEqual(prefix + middle + suffix, original_text)

    def test__fim_and_reordering_split_rates_work(self) -> None:
        config = FIMConfig(
            fim_rate=0.5,
            psm_spm_split=0.5,
            file_separator_token=FILE_SEPARATOR,
            fim_prefix_token=FIM_PREFIX_TOKEN,
            fim_middle_token=FIM_MIDDLE_TOKEN,
            fim_suffix_token=FIM_SUFFIX_TOKEN,
        )
        fim = FillInMiddle(config)
        reordered = fim.perform_on_document(mk_document(mk_text(100_000)))
        files = reordered.text.split(FILE_SEPARATOR)

        self.assertEqual(len(files), 100_000)

        psm_reordered = 0
        spm_reordered = 0

        psm_match = r"<\|fim_prefix\|>.+<\|fim_suffix\|>.+<\|fim_mid\|>.+"
        spm_match = r"<\|fim_suffix\|>.+<\|fim_prefix\|>.+<\|fim_mid\|>.+"

        for file in files:
            for _ in re.finditer(psm_match, file, re.DOTALL):
                psm_reordered += 1
            for _ in re.finditer(spm_match, file, re.DOTALL):
                spm_reordered += 1

        self.assertAlmostEqual((psm_reordered + spm_reordered) / 100_000, 0.5, 2)
        self.assertAlmostEqual(psm_reordered / (psm_reordered + spm_reordered), 0.5, 2)
