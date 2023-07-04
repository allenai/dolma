import itertools
from pathlib import Path
from unittest import TestCase
import os

from dolma.core.paths import glob_path


LOCAL_DATA = Path(__file__).parent.parent / "data"


class TestPaths(TestCase):
    def test_local_glob_path(self):
        local_glob = str(LOCAL_DATA / "*.json.gz")
        paths = list(glob_path(local_glob))
        expected = [str(LOCAL_DATA / fn) for fn in os.listdir(LOCAL_DATA) if fn.endswith(".json.gz")]
        self.assertEqual(sorted(paths), sorted(expected))

    def test_remote_glob_path(self):
        prefix = "s3://ai2-llm/pretraining-data/tests/mixer/expected"
        paths = glob_path(f"{prefix}/*")
        expected = [f"{prefix}/{fn}" for fn in os.listdir(LOCAL_DATA / "expected") if fn.endswith(".json.gz")]
        self.assertEqual(sorted(paths), sorted(expected))

    def test_local_glob_with_recursive(self):
        local_glob = str(LOCAL_DATA / "**/*-paragraphs.json.gz")
        paths = list(glob_path(local_glob))
        expected = list(
            itertools.chain.from_iterable(
                (str(fp), ) if (fp := LOCAL_DATA / fn).is_file() and 'paragraphs' in fn else (
                    (str(fp / sn) for sn in os.listdir(fp) if 'paragraphs' in sn) if fp.is_dir() else ()
                )
                for fn in os.listdir(LOCAL_DATA)
            )
        )
        self.assertEqual(sorted(paths), sorted(expected))
