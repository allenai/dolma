import os
import json
from typing import List
from pathlib import Path
from unittest import TestCase

import smart_open

from dolma.cli.__main__ import main
from .utils import get_test_prefix, clean_test_data


EMAIL_SPANS = Path(__file__).parent.parent / "config/email-spans.json"
FILTER_BY_SPANS = Path(__file__).parent.parent / "config/filter-by-spans.json"
MIXER = Path(__file__).parent.parent / "config/mixer.json"
PARAGRAPH_SPANS = Path(__file__).parent.parent / "config/paragraph-spans.json"


def load_jsonl(fp: str) -> List[dict]:
    with smart_open.open(fp, "r") as f:
        return [json.loads(ln) for ln in f]


class TestMixer(TestCase):
    def setUp(self) -> None:
        self.test_prefix = get_test_prefix()
        self.to_delete: List[str] = []

    def tearDown(self) -> None:
        clean_test_data(self.test_prefix)
        for fp in self.to_delete:
            if not os.path.exists(fp):
                continue
            os.remove(fp)

    def test_email_spans(self):
        main(argv=["-c", str(EMAIL_SPANS), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/email-spans.json.gz"),
            load_jsonl("tests/work/output/email-spans/email-spans-0000.json.gz")
        )

    def test_filter_by_spans(self):
        main(argv=["-c", str(FILTER_BY_SPANS), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/filter-by-spans.json.gz"),
            load_jsonl("tests/work/output/filter-by-spans/filter-by-spans-test-0000.json.gz")
        )

    def test_mixer(self):
        main(argv=["-c", str(MIXER), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/mixer.json.gz"),
            load_jsonl("tests/work/output/mixer/mixer-test-0000.json.gz")
        )

    def test_paragraph_spans(self):
        main(argv=["-c", str(PARAGRAPH_SPANS), "mix"])

        self.assertEqual(
            load_jsonl("tests/data/expected/remove-paragraphs.json.gz"),
            load_jsonl("tests/work/output/paragraph-spans/paragraph-spans-test-0000.json.gz")
        )
