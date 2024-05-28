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


DATA_PATH = Path(__file__).parent.parent / "data/science"

class TestScienceWarcExtractor(unittest.TestCase):
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
