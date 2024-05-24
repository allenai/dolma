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
from dolma.warc.iterator import BackoffWarcIterator

DATA_PATH = Path(__file__).parent.parent / "data/warc"

URL_LIST = [
    "https://creativecommons.org/",
    "https://creativecommons.org/mission/",
    "https://creativecommons.org/2024/03/28/cc-joins-civil-society-letter-urging-u-s-to-support-openness-and-transparency-in-ai/",
    "https://creativecommons.org/2024/04/23/cc-at-wipo-slow-progress-on-copyright-exceptions-for-cultural-heritage-institutions/",
    "https://allenai.org/",
    "https://allenai.org/",
    "https://prior.allenai.org/",
    "https://www.semanticscholar.org/about",
    "https://allenai.org/reviz",
    "https://allenai.org/",
    "https://commoncrawl.org/",
    "https://commoncrawl.org/ccbot",
    "https://commoncrawl.org/blog/march-april-2024-newsletter",
    "https://commoncrawl.org/blog/host-and-domain-level-web-graphs-september-october-november-december-2023-and-february-march-2024",
    "https://commoncrawl.org/faq",
]


class TestWarcExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.tempdir = self.stack.enter_context(tempfile.TemporaryDirectory())

    def tearDown(self) -> None:
        self.stack.close()

    def _run_pipeline(
        self, html: bool = False, pretag: bool = False, skip_dup: bool = False
    ) -> Dict[str, List[dict]]:
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
            backoff_max_time=0,
            backoff_max_tries=1,
            skip_duplicate_urls=skip_dup,
            compression="gz",
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

    def test_warc_dedup(self):
        outputs = self._run_pipeline(skip_dup=True)
        self.assertEqual(len(outputs), 2)
        self.assertIn("sample-0000.jsonl.gz", outputs)
        self.assertIn("sample-0001.jsonl.gz", outputs)

        sample0 = outputs["sample-0000.jsonl.gz"]
        sample1 = outputs["sample-0001.jsonl.gz"]

        self.assertEqual(len(sample0), 22)
        self.assertEqual(len(sample1), 13)  # has 2 duplicates

    def test_pretag_html(self):
        outputs = self._run_pipeline(html=True, pretag=True)
        self.assertEqual(len(outputs), 2)
        self.assertIn("sample-0000.jsonl.gz", outputs)
        self.assertIn("sample-0001.jsonl.gz", outputs)

        sample0 = outputs["sample-0000.jsonl.gz"]
        sample1 = outputs["sample-0001.jsonl.gz"]

        self.assertEqual(len(sample0), 1)
        self.assertEqual(len(sample1), 3)

        self.assertTrue(sample0[0]["metadata"]["url"].startswith("soldaini.net"))
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


class TestBackoffWarcIterator(unittest.TestCase):
    def setUp(self):
        self.path = "tests/data/warc/sample-0001.warc.gz"
        self.response_cnt = 22
        self.info_cnt = 1

    def test_backoff(self):
        elements = []
        offset = 0

        with BackoffWarcIterator(path=self.path, max_tries=1) as it:
            for i, record in enumerate(it):
                elements.append(record)
                if i:
                    self.assertGreater(it._location, offset)
                offset = it._location

        self.assertGreater(len(elements), 0)
        self.assertGreater(os.path.getsize(self.path), offset)
        self.assertEqual(len(elements), self.response_cnt + self.info_cnt)

    def test_seek_mechanism(self):
        elements = []
        offset_fifth_elem = None

        LOC_A = 2
        LOC_B = 7
        expected_order = URL_LIST[: LOC_B + 1] + URL_LIST[LOC_A + 1 :]
        self.assertEqual(len(expected_order), 20)

        with BackoffWarcIterator(path=self.path, max_tries=2, record_types=["response"]) as it:
            for i, record in enumerate(it):
                url = record.headers.get("WARC-Target-URI").rstrip(">").lstrip("<")
                elements.append(url)
                print(i, url)
                self.assertEqual(url, expected_order[i])
                if i == LOC_A:
                    offset_fifth_elem = it._location
                if offset_fifth_elem and i == LOC_B:
                    it._location = offset_fifth_elem
                    it._file_object.close()  # this will trigger backoff  # pyright: ignore

        self.assertEqual(len(elements), len(expected_order))
