import sys
import unittest
from pathlib import Path

from dolma.core.registry import TaggerRegistry
from dolma.core.utils import import_modules


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
