import itertools
import os
from pathlib import Path
from unittest import TestCase

from dolma.core.paths import glob_path, sub_path, add_path, _pathify

LOCAL_DATA = Path(__file__).parent.parent / "data"


class TestPaths(TestCase):
    def test_pathify(self):
        path = "s3://path/to/file"
        protocol, path = _pathify(path)
        self.assertEqual(protocol, "s3")
        self.assertEqual(path, Path("path/to/file"))

        path = "path/to/file"
        protocol, path = _pathify(path)
        self.assertEqual(protocol, "")
        self.assertEqual(path, Path("path/to/file"))

        path = "/path/to/file"
        protocol, path = _pathify(path)
        self.assertEqual(protocol, "")
        self.assertEqual(path, Path("/path/to/file"))

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
                (str(fp),)
                if (fp := LOCAL_DATA / fn).is_file() and "paragraphs" in fn
                else ((str(fp / sn) for sn in os.listdir(fp) if "paragraphs" in sn) if fp.is_dir() else ())
                for fn in os.listdir(LOCAL_DATA)
            )
        )
        self.assertEqual(sorted(paths), sorted(expected))

    def test_sub_path(self):
        path_a = "s3://path/to/b/and/more"
        path_b = "s3://path/to/b"

        self.assertEqual(sub_path(path_a, path_b), "and/more")
        self.assertEqual(sub_path(path_b, path_a), path_b)

        path_c = "/path/to/c"
        path_d = "/path/to/c/and/more"

        self.assertEqual(sub_path(path_d, path_c), "and/more")
        self.assertEqual(sub_path(path_c, path_d), path_c)

        with self.assertRaises(ValueError):
            sub_path(path_a, path_c)

    def test_add_path(self):
        path_a = "s3://path/to/b"
        path_b = "and/more"

        self.assertEqual(add_path(path_a, path_b), "s3://path/to/b/and/more")

        path_c = "/path/to/c"
        path_d = "and/more"

        self.assertEqual(add_path(path_c, path_d), "/path/to/c/and/more")

        with self.assertRaises(ValueError):
            add_path(path_a, path_c)
            add_path(path_c, path_a)
            add_path(path_a, path_a)
