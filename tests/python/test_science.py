import itertools
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List

import smart_open

from dolma.warc import create_and_run_warc_pipeline

DATA_PATH = Path(__file__).parent.parent / "data/science"


class TestScienceWarcExtractor(unittest.TestCase):
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
            skip_no_pre_taggers=False,
            skip_no_post_taggers=False,
            store_html_in_metadata=True,
            linearizer_name="fast-p",
            pre_taggers=["owm_math_v2", "owm_latex_v2", "science_kw_v2"],
            post_taggers=["owmV2_FTsciV1_comb"],
            backoff_max_time=0,
            backoff_max_tries=1,
            compression="gz",
        )
        outputs: Dict[str, List[dict]] = {}
        for fn in os.listdir(self.tempdir):
            with smart_open.open(os.path.join(self.tempdir, fn), mode="rt", encoding="utf-8") as f:
                for ln in f:
                    outputs.setdefault(fn, []).append(json.loads(ln))
        return outputs

    def test_science_filter_pipeline(self):
        outputs = self._run_pipeline()
        self.assertEqual(len(outputs), 2)
        documents = {d["metadata"]["url"]: d for d in itertools.chain.from_iterable(outputs.values())}

        taylor = documents["localhost:8000/taylor.html"]
        science = documents["localhost:8000/science.html"]
        math = documents["localhost:8000/math.html"]
        games = documents["localhost:8000/games.html"]

        self.assertIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science", taylor["attributes"])
        self.assertLess(taylor["attributes"]["owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science"][0][-1], 0.5)
        self.assertNotIn("owm_math_v2__owm_math_v2__math", taylor["attributes"])
        self.assertNotIn("owm_latex_v2__owm_latex_v2__latex", taylor["attributes"])
        self.assertNotIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__math_latex", taylor["attributes"])
        self.assertNotIn("science_kw_v2__science_kw_v2__science", taylor["attributes"])

        self.assertIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science", science["attributes"])
        self.assertGreater(science["attributes"]["owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science"][0][-1], 0.5)
        self.assertNotIn("owm_math_v2__owm_math_v2__math", science["attributes"])
        self.assertNotIn("owm_latex_v2__owm_latex_v2__latex", science["attributes"])
        self.assertNotIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__math_latex", science["attributes"])
        self.assertIn("science_kw_v2__science_kw_v2__science", science["attributes"])

        self.assertNotIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science", math["attributes"])
        self.assertIn("owm_math_v2__owm_math_v2__math", math["attributes"])
        self.assertIn("owm_latex_v2__owm_latex_v2__latex", math["attributes"])
        self.assertIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__math_latex", math["attributes"])
        self.assertIn("science_kw_v2__science_kw_v2__science", math["attributes"])

        self.assertIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science", games["attributes"])
        self.assertLess(games["attributes"]["owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__science"][0][-1], 0.5)
        self.assertNotIn("owm_math_v2__owm_math_v2__math", games["attributes"])
        self.assertNotIn("owm_latex_v2__owm_latex_v2__latex", games["attributes"])
        self.assertNotIn("owmV2_FTsciV1_comb__owmV2_FTsciV1_comb__math_latex", games["attributes"])
        self.assertNotIn("science_kw_v2__science_kw_v2__science", games["attributes"])
