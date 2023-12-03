import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List
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

EMAIL_SPANS = Path(__file__).parent.parent / "config/email-spans.json"
FILTER_BY_SPANS = Path(__file__).parent.parent / "config/filter-by-spans.json"
MIXER = Path(__file__).parent.parent / "config/mixer.json"
PARAGRAPH_SPANS = Path(__file__).parent.parent / "config/paragraph-spans.json"


class TestMixer(TestCase):
    def setUp(self) -> None:
        if skip_aws_tests():
            self.remote_test_prefix = None
        else:
            self.remote_test_prefix = get_test_prefix()
            upload_s3_prefix(s3_prefix=f"{self.remote_test_prefix}", local_prefix="tests/data/provided/**/*.gz")
            upload_s3_prefix(
                s3_prefix=f"{self.remote_test_prefix}", local_prefix="tests/data/provided/attributes/**/*.gz"
            )

    def tearDown(self) -> None:
        if self.remote_test_prefix is not None:
            clean_test_data(self.remote_test_prefix)

    def checkAndRemoveProvenance(self, provided: List[dict]) -> List[dict]:
        prev_id = 0
        for row in provided:
            self.assertIn("metadata", row)
            self.assertIn("provenance", row["metadata"])
            provenance = row["metadata"].pop("provenance")
            path, lid = provenance.rsplit(":", 1)
            self.assertGreater(int(lid), prev_id)
            prev_id = int(lid)

            # remove metadata if empty
            len(row["metadata"]) == 0 and row.pop("metadata")

        return provided

    def test_email_spans(self):
        main(argv=["-c", str(EMAIL_SPANS), "mix"])

        expected = load_jsonl("tests/data/expected/email-spans.json.gz")
        provided = load_jsonl("tests/work/output/email-spans/email-spans-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)

    def test_filter_by_spans(self):
        main(argv=["-c", str(FILTER_BY_SPANS), "mix"])

        expected = load_jsonl("tests/data/expected/filter-by-spans.json.gz")
        provided = load_jsonl("tests/work/output/filter-by-spans/filter-by-spans-test-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)

    def test_mixer(self):
        main(argv=["-c", str(MIXER), "mix"])

        expected = load_jsonl("tests/data/expected/mixer.json.gz")
        provided = load_jsonl("tests/work/output/mixer/mixer-test-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)

    def test_paragraph_spans(self):
        main(argv=["-c", str(PARAGRAPH_SPANS), "mix"])

        expected = load_jsonl("tests/data/expected/remove-paragraphs.json.gz")
        provided = load_jsonl("tests/work/output/paragraph-spans/paragraph-spans-test-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)

    def test_local_input_remote_output(self):
        if self.remote_test_prefix is None:
            return self.skipTest("Skipping AWS tests")

        with open(MIXER, mode="r", encoding="utf8") as f:
            config = json.load(f)

        # keep track of local output path
        local_output = config["streams"][0]["output"]["path"]

        # replace results path with s3 path
        config["streams"][0]["output"]["path"] = f"{self.remote_test_prefix}/{local_output}"

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "mix"])

        download_s3_prefix(f"{self.remote_test_prefix}/tests/work", "tests/work/remote")

        expected = load_jsonl("tests/data/expected/mixer.json.gz")
        provided = load_jsonl("tests/work/remote/output/mixer/mixer-test-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)

    def test_remote_input_remote_output(self):
        if self.remote_test_prefix is None:
            return self.skipTest("Skipping AWS tests")

        with open(MIXER, mode="r", encoding="utf8") as f:
            config = json.load(f)

        # keep track of local output path
        local_input = config["streams"][0]["documents"][0]
        local_output = config["streams"][0]["output"]["path"]

        # replace results path with s3 path
        config["streams"][0]["output"]["path"] = f"{self.remote_test_prefix}/{local_output}"

        # upload local input to s3, replace local input with s3 path
        config["streams"][0]["documents"][0] = f"{self.remote_test_prefix}/{local_input}"

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "mix"])

        download_s3_prefix(f"{self.remote_test_prefix}/tests/work", "tests/work/remote")
        expected = load_jsonl("tests/data/expected/mixer.json.gz")
        provided = load_jsonl("tests/work/remote/output/mixer/mixer-test-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)

    def test_remote_input_local_output(self):
        if self.remote_test_prefix is None:
            return self.skipTest("Skipping AWS tests")

        with open(MIXER, mode="r", encoding="utf8") as f:
            config = json.load(f)

        # keep track of local output path
        local_input = config["streams"][0]["documents"][0]

        # upload local input to s3, replace local input with s3 path
        config["streams"][0]["documents"][0] = f"{self.remote_test_prefix}/{local_input}"

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "mix"])

        expected = load_jsonl("tests/data/expected/mixer.json.gz")
        provided = load_jsonl("tests/work/output/mixer/mixer-test-0000.json.gz")
        provided = self.checkAndRemoveProvenance(provided)
        self.assertEqual(expected, provided)


class TestMixerPipeline(TestCasePipeline):
    def test_min_length(self):
        source_dir = Path(self.makeUniquePath())
        output_dir = Path(self.makeUniquePath())

        to_remove = "remove second sentence"
        to_keep_head = "This is a test"
        to_keep_tail = "do not touch"
        documents = [
            "doc",
            self.combineIntoDoc(to_keep_head, to_remove),
            self.combineIntoDoc("A", to_remove),
            self.combineIntoDoc(to_keep_head, to_keep_tail),
            self.combineIntoDoc("", "", "", "p", "", "", ""),
        ]
        docs_path = self.writeDocs(docs=documents, ext_dir=source_dir)

        attributes = [
            [],
            [((start := documents[1].find(to_remove)), start + len(to_remove), 1)],
            [((start := documents[2].find(to_remove)), start + len(to_remove), 1)],
            [],
            [],
        ]
        self.writeAttributes(attributes=attributes, attribute_name="test", ext_dir=source_dir)

        config = {
            "streams": [
                {
                    "name": "test",
                    "documents": docs_path,
                    "attributes": ["test"],
                    "output": {"path": str(output_dir), "max_size_in_bytes": 10000000, "min_text_length": 4},
                    "span_replacement": [{"span": "$.attributes.test", "min_score": 0.5, "replacement": ""}],
                }
            ],
            "processes": 1,
        }

        config_path = self.writeConfig(config=config)

        main(argv=["-c", config_path, "mix"])

        new_docs = self.readUnits(list(output_dir.iterdir()))

        self.assertEqual(len(new_docs), 2)
        self.assertEqual(new_docs[0]["text"], self.combineIntoDoc(to_keep_head, ""))
        self.assertEqual(new_docs[1]["text"], self.combineIntoDoc(to_keep_head, to_keep_tail))
