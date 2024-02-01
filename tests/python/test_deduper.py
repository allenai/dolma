import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import TestCase

from dolma.cli.__main__ import main

from .utils import (
    TestCasePipeline,
    clean_test_data,
    download_s3_prefix,
    get_test_prefix,
    load_jsonl,
    skip_aws_tests,
    upload_s3_prefix,
)

DEDUPE_BY_URL = Path(__file__).parent.parent / "config/dedupe-by-url.json"
DEDUPE_PARAGRAPHS = Path(__file__).parent.parent / "config/dedupe-paragraphs.json"


class TestDeduper(TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.local_temp_dir = self.stack.enter_context(TemporaryDirectory()).rstrip("/")

        if skip_aws_tests():
            self.remote_test_prefix = None
        else:
            self.remote_test_prefix = get_test_prefix()

            # upload test data
            upload_s3_prefix(
                s3_prefix=f"{self.remote_test_prefix}", local_prefix="tests/data/provided/documents/*.gz"
            )

        # copy provided config files to local temp dir
        shutil.copytree(
            "tests/data/provided/documents",
            f"{self.local_temp_dir}/tests/data/provided/documents",
            dirs_exist_ok=True,
        )

    def tearDown(self) -> None:
        if self.remote_test_prefix is not None:
            clean_test_data(self.remote_test_prefix)
        self.stack.close()

    def test_dedupe_by_url(self):
        with open(DEDUPE_BY_URL, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.local_temp_dir}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        expected = load_jsonl("tests/data/expected/dedupe-by-url.json.gz")
        computed = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/attributes/dedupe_by_url/000.json.gz")
        self.assertEqual(expected, computed)

    def test_dedupe_paragraphs(self):
        with open(DEDUPE_PARAGRAPHS, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.local_temp_dir}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        expected = load_jsonl("tests/data/expected/dedupe-paragraphs.json.gz")
        computed = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/attributes/dedupe_paragraphs/000.json.gz"
        )
        self.assertEqual(expected, computed)

    def test_dedupe_by_url_remote_input(self):
        if self.remote_test_prefix is None:
            return self.skipTest("Skipping AWS tests")

        with open(DEDUPE_BY_URL, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.remote_test_prefix}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        download_s3_prefix(self.remote_test_prefix, self.local_temp_dir)

        expected = load_jsonl("tests/data/expected/dedupe-by-url.json.gz")
        computed = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/attributes/dedupe_by_url/000.json.gz")
        self.assertEqual(expected, computed)


class TestDeduperPipeline(TestCasePipeline):
    def test_skip_empty(self):
        duplicate_text = "More text"
        documents = [
            self.combineIntoDoc("Short document", "", duplicate_text),
            self.combineIntoDoc("Short document #2", "", duplicate_text),
        ]

        docs_fp = self.writeDocs(documents)
        key_name = "dedupe_paragraphs"
        attribute_name = "bff_duplicate_paragraph_spans"

        config = {
            "documents": docs_fp,
            "dedupe": {
                "name": key_name,
                "paragraphs": {"attribute_name": attribute_name},
                "skip_empty": True,
            },
            "bloom_filter": {
                "file": self.makeUniquePath(),
                "read_only": False,
                "estimated_doc_count": 100,
                "desired_false_positive_rate": 1e-06,
            },
            "processes": 1,
        }

        config_path = self.writeConfig(config)

        main(argv=["-c", config_path, "dedupe"])

        expected = self.readUnits([p.replace("documents", f"attributes/{key_name}") for p in docs_fp])
        self.assertEqual(len(expected), 2)

        # no duplicate on first doc
        self.assertIn("attributes", expected[0])
        self.assertIn(attribute_name, expected[0]["attributes"])
        self.assertEqual(expected[0]["attributes"][attribute_name], [])

        # duplicate on second doc
        self.assertIn("attributes", expected[1])
        self.assertIn(attribute_name, expected[1]["attributes"])
        self.assertEqual(len(expected[1]["attributes"][attribute_name]), 1)
        (start, end, score), *_ = expected[1]["attributes"][attribute_name]
        self.assertEqual(documents[1][start:end], duplicate_text)
        self.assertEqual(score, 1.0)

        # now let's not skip empty docs
        config["dedupe"]["skip_empty"] = False
        config["bloom_filter"]["file"] = self.makeUniquePath()  # new filter
        config_path = self.writeConfig(config)

        main(argv=["-c", config_path, "dedupe"])

        expected = self.readUnits([p.replace("documents", f"attributes/{key_name}") for p in docs_fp])

        # two duplicates on second doc
        self.assertEqual(len(expected[1]["attributes"][attribute_name]), 2)
        (s1, e1, v1), (s2, e2, v2) = expected[1]["attributes"][attribute_name]
        self.assertEqual(documents[1][s1:e1], "\n")
        self.assertEqual(v1, 1.0)
        self.assertEqual(documents[1][s2:e2], duplicate_text)
        self.assertEqual(v2, 1.0)
