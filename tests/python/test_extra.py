import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import smart_open

from dolma.core.registry import TaggerRegistry
from dolma.core.runtime import create_and_run_tagger
from dolma.core.utils import import_modules

LOCAL_DATA = Path(__file__).parent.parent / "data"


class TestExtra(unittest.TestCase):
    def setUp(self) -> None:
        self.current_path = Path(__file__).parent.absolute()

    def test_import_from_module(self):
        sys.path.append(f"{self.current_path}/extras")
        import_modules(["extras_from_module"])
        self.assertTrue(TaggerRegistry.has("extra_v1"))

    def test_import_from_path(self):
        import_modules([f"{self.current_path}/extras/extras_from_path/extra_taggers.py"])
        self.assertTrue(TaggerRegistry.has("extra_v2"))

    def test_import_from_module_path(self):
        import_modules([f"{self.current_path}/extras/extras_from_module_path"])
        self.assertTrue(TaggerRegistry.has("extra_v3"))

    def test_tagging_with_extra(self):
        taggers_modules = [f"{self.current_path}/extras/useful_extra"]
        documents_path = f"{LOCAL_DATA}/provided/documents/000.json.gz"
        taggers = ["c4_v1", "extra_v4"]

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[documents_path],
                destination=temp_dir,
                taggers=taggers,
                experiment="test",
                taggers_modules=taggers_modules,
                debug=True,
            )
            destination_file = os.path.join(temp_dir, "test", "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(destination_file, "rt") as f:
                attributes_with_exp = [json.loads(ln) for ln in f]

            for row in attributes_with_exp:
                self.assertTrue("test__extra_v4__random" in row["attributes"])
