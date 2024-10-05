import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from itertools import chain
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

    def _run_pipeline(self, html: bool = False, pretag: bool = False) -> Dict[str, List[dict]]:
        create_and_run_warc_pipeline(
            documents=[f"{DATA_PATH}/*.warc.gz"],
            destination=[self.tempdir],
            num_processes=1,
            ignore_existing=False,
            debug=True,
            source_name="test",
            skip_no_pre_taggers=pretag,
            skip_no_post_taggers=False,
            store_html_in_metadata=html,
            linearizer_name="resiliparse",
            pre_taggers=["cc_re"],
            post_taggers=["lingua_1e2"],
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

        for sample in chain(sample0, sample1):
            languages = [
                (k.rsplit("_", 1)[-1], s[0][-1])
                for k, s in sample["attributes"].items()
                if k.startswith("lingua_1e2")
            ]
            languages = sorted(languages, key=lambda x: -x[1])

            self.assertGreater(len(sample["text"]), 50)
            self.assertEqual(languages[0][0], "en")

            self.assertEqual(sample["version"], "v0")
            self.assertEqual(sample["source"], "test")
            self.assertIn("warc_url", sample["metadata"])
            self.assertIn("url", sample["metadata"])
            self.assertIn("warc_date", sample["metadata"])
            self.assertIn("warc_filename", sample["metadata"])
            self.assertIn("content_type", sample["metadata"])

    def test_pretag_html(self):
        outputs = self._run_pipeline(html=True, pretag=True)
        self.assertEqual(len(outputs), 2)
        self.assertIn("sample-0000.jsonl.gz", outputs)
        self.assertIn("sample-0001.jsonl.gz", outputs)

        sample0 = outputs["sample-0000.jsonl.gz"]
        sample1 = outputs["sample-0001.jsonl.gz"]

        self.assertEqual(len(sample0), 1)
        self.assertEqual(len(sample1), 3)

        self.assertEqual(sample0[0]["metadata"]["url"], "soldaini.net")
        self.assertTrue(sample1[0]["metadata"]["url"].startswith("creativecommons.org"))
        self.assertTrue(sample1[1]["metadata"]["url"].startswith("creativecommons.org"))
        self.assertTrue(sample1[2]["metadata"]["url"].startswith("creativecommons.org"))

        self.assertIn("cc_re__cc_re__cc_by_4_0", sample0[0]["attributes"])
        self.assertEqual(
            set(k.strip("cc_re__cc_re__") for k in sample1[0]["attributes"] if k.startswith("cc")),
            {"by_4_0", "publicdomain_mark_1_0", "by_2_0", "by_3_0_en", "by_nc_sa_4_0"},
        )
        self.assertEqual(
            set(k.strip("cc_re__cc_re__") for k in sample1[1]["attributes"] if k.startswith("cc")),
            {"by_4_0", "by_3_0"},
        )
        self.assertIn("cc_re__cc_re__cc_by_4_0", sample1[2]["attributes"])
