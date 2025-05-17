import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

FILE_SEPARATOR = "<|file_sep|>"
REPO_FIELD_NAME = "repo_name"
PL_FIELD_NAME = "language"


JAVASCRIPT_FILE_CONTENTS = [
    "function add(a, b) { return a + b; };",
    "function subtract(a, b) { return a - b; };",
]

PYTHON_FILE_CONTENTS = [
    "add = lambda a, b: a + b",
    "subtract = lambda a, b: a - b",
]


def mk_command(
    input_dir: str,
    output_dir: str,
    randomize_order: bool,
) -> str:
    return f"""
cargo run -- \
    --inputs '{input_dir}/*.jsonl' \
    --output {output_dir} \
    {'--randomize-order' if randomize_order else ''} \
    --file-separator-token '{FILE_SEPARATOR}' \
    --repo-field-name '{REPO_FIELD_NAME}' \
    --pl-field-name '{PL_FIELD_NAME}' 
""".strip()


def mk_partition_files(dir: str, num_partitions: int, num_repos_per_partition: int) -> None:
    row_jsons = []
    for i in range(num_repos_per_partition):

        # javascript
        for javascript in JAVASCRIPT_FILE_CONTENTS:
            row_json = json.dumps(
                dict(
                    source="somesource",
                    version="1234",
                    id=str(uuid4()),
                    text=javascript,
                    metadata={PL_FIELD_NAME: "Javascript", REPO_FIELD_NAME: f"repo-{i}"},
                )
            )
            row_jsons.append(row_json)

        # python
        for python in PYTHON_FILE_CONTENTS:
            row_json = json.dumps(
                dict(
                    source="somesource",
                    version="1234",
                    id=str(uuid4()),
                    text=python,
                    metadata={PL_FIELD_NAME: "Python", REPO_FIELD_NAME: f"repo-{i}"},
                )
            )
            row_jsons.append(row_json)

    lines = "\n".join(row_jsons)

    for i in range(num_partitions):
        output_path = os.path.join(dir, f"{i}.jsonl")
        with open(output_path, "w") as f:
            f.write(lines)


def perform_concatenation(
    num_partitions: int, num_repos_per_partition: int, randomize_order: float
) -> List[List[Dict[str, Any]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.mkdir(input_dir)
        os.mkdir(output_dir)
        mk_partition_files(input_dir, num_partitions, num_repos_per_partition)
        cmd = mk_command(input_dir, output_dir, randomize_order)

        root_dir = _find_rust_root()
        subprocess.run(shlex.split(cmd), check=True, cwd=root_dir)

        results = []

        for output_file_name in sorted(os.listdir(output_dir)):
            with open(os.path.join(output_dir, output_file_name), "r") as output_file:
                output_dicts = [json.loads(line) for line in output_file.readlines()]
            results.append(output_dicts)

        return results


def _find_rust_root() -> Path:
    rust_root = Path(__file__)
    while True:
        if rust_root == Path("/"):
            raise FileNotFoundError("Could not find rust root")
        if (rust_root / "Cargo.toml").exists():
            return rust_root
        rust_root = rust_root.parent


class TestCodeFileConcat(unittest.TestCase):
    def assert_partition_looks_good(self, repo_docs, expected_num_repos) -> None:
        print(repo_docs)
        self.assertEqual(len(repo_docs), expected_num_repos * 2)

        for i in range(0, expected_num_repos, 2):
            javascript_row, python_row = repo_docs[i], repo_docs[i + 1]

            self.assertEqual(FILE_SEPARATOR.join(JAVASCRIPT_FILE_CONTENTS), javascript_row["text"])
            self.assertEqual(FILE_SEPARATOR.join(PYTHON_FILE_CONTENTS), python_row["text"])

            self.assertEqual(javascript_row["metadata"][PL_FIELD_NAME], "Javascript")
            self.assertEqual(python_row["metadata"][PL_FIELD_NAME], "Python")

            self.assertEqual(javascript_row["metadata"][REPO_FIELD_NAME], f"repo-{i//2}")
            self.assertEqual(python_row["metadata"][REPO_FIELD_NAME], f"repo-{i//2}")

    def test__concatenation_works_in_simplest_case(self) -> None:
        output_rows = perform_concatenation(
            num_partitions=1,
            num_repos_per_partition=1,
            randomize_order=False,
        )

        self.assertEqual(len(output_rows), 1)
        self.assert_partition_looks_good(output_rows[0], expected_num_repos=1)

    def test__concatenation_works_over_many_repos_and_partitions(self) -> None:
        output_rows = perform_concatenation(num_partitions=10, num_repos_per_partition=4, randomize_order=False)

        self.assertEqual(len(output_rows), 10)

        for partition in output_rows:
            self.assert_partition_looks_good(partition, expected_num_repos=4)

    def test__randomized_order_works(self) -> None:
        output_rows = perform_concatenation(
            num_partitions=1,
            num_repos_per_partition=10_000,
            randomize_order=True,
        )

        self.assertEqual(len(output_rows[0]), 20_000)

        repo_docs_that_start_with_the_add_fn = [
            repo_doc
            for repo_doc in output_rows[0]
            if (
                repo_doc["text"].startswith(JAVASCRIPT_FILE_CONTENTS[0])
                and repo_doc["metadata"][PL_FIELD_NAME] == "Javascript"
            )
            or (
                repo_doc["text"].startswith(PYTHON_FILE_CONTENTS[0])
                and repo_doc["metadata"][PL_FIELD_NAME] == "Python"
            )
        ]

        self.assertAlmostEqual(len(repo_docs_that_start_with_the_add_fn) / 20_000, 0.5, 2)
