import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import smart_open

from dolma.models.data import _make_selector, _PartitionedFileWriter, make_fasttext_data

DATA_DIR = Path(__file__).parent.parent / "data"


class TestSelector(TestCase):
    def test_selector(self):
        d = {"a": [{"b": 1}, {"c": [2, {"d": 3}], "e": 4}, {"f": 5}], "g": 6}
        self.assertEqual(_make_selector("$.a")(d), d["a"])
        self.assertEqual(_make_selector("$.a[1].c")(d), d["a"][1]["c"])
        self.assertEqual(_make_selector("$.a[1].c[1]")(d), d["a"][1]["c"][1])
        self.assertEqual(_make_selector("$.a[1].c[1].d")(d), d["a"][1]["c"][1]["d"])
        self.assertEqual(_make_selector("$.a[1].e")(d), d["a"][1]["e"])
        self.assertEqual(_make_selector("$.g")(d), d["g"])

        with self.assertRaises(TypeError):
            _make_selector("$.a[1].c[1].d[0]")(d)

        with self.assertRaises(KeyError):
            _make_selector("$.z")(d)


class PartitionedFileWriterTest(TestCase):
    num_rows = 100
    max_size = 100

    def test_single_file_written(self):
        with TemporaryDirectory() as tmpdir:
            # this will create a single file
            with _PartitionedFileWriter(path=f"{tmpdir}/file.txt.gz") as writer:
                for i in range(self.num_rows):
                    writer.write(f"{i}\n")

            # we check if the file name is as expected
            self.assertEqual(os.listdir(tmpdir), ["file-00000.txt.gz"])

            # we read the file and check if the content is as expected
            with smart_open.open(f"{tmpdir}/file-00000.txt.gz", "r") as f:
                self.assertEqual(f.read(), "".join(f"{i}\n" for i in range(self.num_rows)))

    def test_multiple_file_written(self):
        with TemporaryDirectory() as tmpdir:
            # this will create multiple files (should be 3 files in total)
            with _PartitionedFileWriter(path=f"{tmpdir}/file.txt.gz", max_size=self.max_size) as writer:
                bytes_written = 0
                for i in range(self.num_rows):
                    s = f"{i}\n"
                    writer.write(s)
                    bytes_written += len(s)

            # we first check if the number of files and their names are as expected
            expected_files = [f"file-{i:05d}.txt.gz" for i, _ in enumerate(range(0, bytes_written, self.max_size))]
            self.assertEqual(sorted(os.listdir(tmpdir)), expected_files)

            # we read everything in the files and check if the content is as expected
            all_read = ""
            for fn in expected_files:
                with smart_open.open(f"{tmpdir}/{fn}", "r") as f:
                    all_read += f.read()
            self.assertEqual(all_read, "".join(f"{i}\n" for i in range(self.num_rows)))

    def test_extension_options(self):
        with TemporaryDirectory() as tmpdir:
            # this will create a single file with one extension part
            with _PartitionedFileWriter(path=f"{tmpdir}/file.txt") as writer:
                for i in range(self.num_rows):
                    writer.write(f"{i}\n")

            # we check if the file name is as expected
            self.assertEqual(os.listdir(tmpdir), ["file-00000.txt"])

        with TemporaryDirectory() as tmpdir:
            # this will create a single file with no extension part
            with _PartitionedFileWriter(path=f"{tmpdir}/file") as writer:
                for i in range(self.num_rows):
                    writer.write(f"{i}\n")

            # we check if the file name is as expected
            self.assertEqual(os.listdir(tmpdir), ["file-00000"])


class TestFasttextData(TestCase):
    def test_make_fasttext_data_from_dict(self):
        paths = {"pos": [str(DATA_DIR / "mutiple_files")], "neg": [str(DATA_DIR / "provided/documents")]}

        with TemporaryDirectory() as tmpdir:
            make_fasttext_data(
                paths=paths,
                dest=tmpdir,
                train_sample=0.5,
                dev_sample=0.3,
                test_sample=0.2,
            )
            print(tmpdir)
            breakpoint()
