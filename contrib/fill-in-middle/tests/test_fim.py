import json
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

FILE_SEPARATOR = "<|file_sep|>"
FIM_MIDDLE_TOKEN = "<|fim_mid|>"
FIM_PREFIX_TOKEN = "<|fim_prefix|>"
FIM_SUFFIX_TOKEN = "<|fim_suffix|>"


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


def mk_command(
    input_dir: str,
    output_dir: str,
    fim_rate: float,
    psm_spm_split: float,
) -> str:
    return f"""
cargo run -- \
    --inputs '{input_dir}/*.jsonl' \
    --output {output_dir} \
    --fim-rate {fim_rate} \
    --psm-spm-split {psm_spm_split} \
    --file-separator-token '{FILE_SEPARATOR}' \
    --fim-prefix-token '{FIM_PREFIX_TOKEN}' \
    --fim-middle-token '{FIM_MIDDLE_TOKEN}' \
    --fim-suffix-token '{FIM_SUFFIX_TOKEN}'
""".strip()


def mk_partition_files(dir: str, num_partitions: int, num_rows_per_partition: int, row_text: str) -> None:
    for i in range(num_partitions):
        output_path = os.path.join(dir, f"{i}.jsonl")
        with open(output_path, "w") as f:
            row_json = json.dumps(
                dict(source="somesource", version="1234", id=str(uuid4()), text=row_text, metadata={})
            )
            lines = "\n".join([row_json] * num_rows_per_partition)
            f.write(lines)


def mk_text(num_source_files: int) -> str:
    source_files = [CODE_FILE_1 if j % 2 == 0 else CODE_FILE_2 for j in range(num_source_files)]
    return FILE_SEPARATOR.join(source_files)


def perform_rewrites(
    num_partitions: int,
    num_rows_per_partition: int,
    row_text: str,
    fim_rate: float,
    psm_spm_split: float,
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.mkdir(input_dir)
        os.mkdir(output_dir)
        mk_partition_files(input_dir, num_partitions, num_rows_per_partition, row_text)
        cmd = mk_command(input_dir, output_dir, fim_rate, psm_spm_split)

        root_dir = _find_rust_root()
        subprocess.run(shlex.split(cmd), check=True, cwd=root_dir)

        input_file_names = sorted(os.listdir(input_dir))
        output_file_names = sorted(os.listdir(output_dir))

        results = []

        for input_file_name, output_file_name in zip(input_file_names, output_file_names):
            with open(os.path.join(input_dir, input_file_name), "r") as input_file:
                input_dicts = [json.loads(line) for line in input_file.readlines()]
            with open(os.path.join(output_dir, output_file_name), "r") as output_file:
                output_dicts = [json.loads(line) for line in output_file.readlines()]
            results.append((input_dicts, output_dicts))

        return results


def perform_single_row_rewrite(
    row_text: str, fim_rate: float, psm_spm_split: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inputs, outputs = perform_rewrites(
        num_partitions=1,
        num_rows_per_partition=1,
        row_text=row_text,
        fim_rate=fim_rate,
        psm_spm_split=psm_spm_split,
    )[0]

    return inputs[0], outputs[0]


def _find_rust_root() -> Path:
    rust_root = Path(__file__)
    while True:
        if rust_root == Path("/"):
            raise FileNotFoundError("Could not find rust root")
        if (rust_root / "Cargo.toml").exists():
            return rust_root
        rust_root = rust_root.parent


class TestFillInMiddle(unittest.TestCase):
    def test__fim_reordering_works(self) -> None:
        # First, Prefix-Suffix-Middle
        psm_input_row, psm_output_row = perform_single_row_rewrite(
            row_text=mk_text(1),
            fim_rate=1.0,
            psm_spm_split=1.0,
        )

        psm_original_text = psm_input_row["text"]
        psm_final_text = psm_output_row["text"]

        prefix_plus_prefix_token, rest = psm_final_text.split(FIM_SUFFIX_TOKEN)
        _, prefix = prefix_plus_prefix_token.split(FIM_PREFIX_TOKEN)
        suffix, middle = rest.split(FIM_MIDDLE_TOKEN)

        self.assertEqual(prefix + middle + suffix, psm_original_text)

        # Next, Suffix-Prefix-Middle
        spm_input_row, spm_output_row = perform_single_row_rewrite(
            row_text=mk_text(1),
            fim_rate=1.0,
            psm_spm_split=0,
        )

        spm_original_text = spm_input_row["text"]
        spm_final_text = spm_output_row["text"]

        suffix_plus_suffix_token, rest = spm_final_text.split(FIM_PREFIX_TOKEN)
        _, suffix = suffix_plus_suffix_token.split(FIM_SUFFIX_TOKEN)
        prefix, middle = rest.split(FIM_MIDDLE_TOKEN)

        self.assertEqual(prefix + middle + suffix, spm_original_text)

    def test__fim_and_reordering_split_rates_work(self) -> None:
        _, output_row = perform_single_row_rewrite(
            row_text=mk_text(300_000),
            fim_rate=0.5,
            psm_spm_split=0.5,
        )

        final_text = output_row["text"]
        files = final_text.split(FILE_SEPARATOR)

        self.assertEqual(len(files), 300_000)

        psm_reordered = 0
        spm_reordered = 0

        psm_match = r"<\|fim_prefix\|>.+<\|fim_suffix\|>.+<\|fim_mid\|>.+"
        spm_match = r"<\|fim_suffix\|>.+<\|fim_prefix\|>.+<\|fim_mid\|>.+"

        for file in files:
            for _ in re.finditer(psm_match, file, re.DOTALL):
                psm_reordered += 1
            for _ in re.finditer(spm_match, file, re.DOTALL):
                spm_reordered += 1

        self.assertAlmostEqual((psm_reordered + spm_reordered) / 300_000, 0.5, 2)
        self.assertAlmostEqual(psm_reordered / (psm_reordered + spm_reordered), 0.5, 2)

    def test__fim_needs_at_least_five_characters_to_rearrange(self) -> None:
        for i in range(5):
            starting_string = "a" * i or ""
            final_string = perform_single_row_rewrite(
                row_text=starting_string,
                fim_rate=1,
                psm_spm_split=1,
            )[
                0
            ]["text"]

            if i < 5:
                self.assertEqual(final_string, starting_string)
            else:
                self.assertTrue(FIM_PREFIX_TOKEN in final_string)

    def test__fim_handles_lots_of_partitions_with_lots_of_rows(self) -> None:
        results = perform_rewrites(
            num_partitions=5, num_rows_per_partition=2, row_text=mk_text(10), fim_rate=1, psm_spm_split=1
        )

        self.assertEqual(len(results), 5)

        for inputs, outputs in results:
            self.assertEqual(len(inputs), 2)
            self.assertEqual(len(outputs), 2)

            psm_match = r"<\|fim_prefix\|>.+<\|fim_suffix\|>.+<\|fim_mid\|>.+(<\|file_sep\|>)?"

            for output in outputs:
                num_rewrites = 0
                files = output["text"].split(FILE_SEPARATOR)
                for file in files:
                    num_rewrites += len([re.finditer(psm_match, file, re.DOTALL)])
                self.assertEqual(num_rewrites, 10)
