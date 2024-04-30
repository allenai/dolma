import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List

import smart_open

from dolma.warc import create_and_run_warc_pipeline

DATA_PATH = Path(__file__).parent.parent / "data/warc"


class TestWarcExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.tempdir = self.stack.enter_context(tempfile.TemporaryDirectory())

    def tearDown(self) -> None:
        self.stack.close()

    def _run_pipeline(self) -> Dict[str, List[dict]]:
        create_and_run_warc_pipeline(
            documents=[f"{DATA_PATH}/*.warc.gz"],
            destination=[self.tempdir],
            num_processes=1,
            ignore_existing=False,
            debug=True,
            source_name="test",
            skip_if_empty_heuristics=False,
            store_html_in_metadata=False,
            linearizer_name="resiliparse",
            extractors_name=["fasttext_en", "cc_re", "cc_re_fast"],
        )
        outputs: Dict[str, List[dict]] = {}
        for fn in os.listdir(self.tempdir):
            with smart_open.open(os.path.join(self.tempdir, fn), mode="rt", encoding="utf-8") as f:
                for ln in f:
                    outputs.setdefault(fn, []).append(json.loads(ln))

        return outputs

    def test_verify_extraction(self):
        outputs = self._run_pipeline()
        self.assertEqual(len(outputs), 2)
        self.assertIn("sample-0000.jsonl.gz", outputs)
        self.assertIn("sample-0001.jsonl.gz", outputs)

        sample0 = outputs["sample-0000.jsonl.gz"]
        sample1 = outputs["sample-0001.jsonl.gz"]

        self.assertEqual(len(sample0), 22)
        self.assertEqual(len(sample1), 15)

        breakpoint()
