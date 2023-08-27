import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

from dolma.cli.__main__ import main

from .utils import (
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

    def test_email_spans(self):
        main(argv=["-c", str(EMAIL_SPANS), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/email-spans.json.gz"),
            load_jsonl("tests/work/output/email-spans/email-spans-0000.json.gz"),
        )

    def test_filter_by_spans(self):
        main(argv=["-c", str(FILTER_BY_SPANS), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/filter-by-spans.json.gz"),
            load_jsonl("tests/work/output/filter-by-spans/filter-by-spans-test-0000.json.gz"),
        )

    def test_mixer(self):
        main(argv=["-c", str(MIXER), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/mixer.json.gz"),
            load_jsonl("tests/work/output/mixer/mixer-test-0000.json.gz"),
        )

    def test_paragraph_spans(self):
        main(argv=["-c", str(PARAGRAPH_SPANS), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/remove-paragraphs.json.gz"),
            load_jsonl("tests/work/output/paragraph-spans/paragraph-spans-test-0000.json.gz"),
        )

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

        self.assertEqual(
            load_jsonl("tests/data/expected/mixer.json.gz"),
            load_jsonl("tests/work/remote/output/mixer/mixer-test-0000.json.gz"),
        )

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

        self.assertEqual(
            load_jsonl("tests/data/expected/mixer.json.gz"),
            load_jsonl("tests/work/remote/output/mixer/mixer-test-0000.json.gz"),
        )

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

        self.assertEqual(
            load_jsonl("tests/data/expected/mixer.json.gz"),
            load_jsonl("tests/work/output/mixer/mixer-test-0000.json.gz"),
        )
