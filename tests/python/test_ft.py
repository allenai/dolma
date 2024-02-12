import glob
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from unittest import TestCase

import smart_open

from dolma.core.data_types import InputSpecWithMetadata
from dolma.models.data import FastTextDataConverter, make_selector

DATA_DIR = Path(__file__).parent.parent / "data"


class TestSelector(TestCase):
    def test_selector(self):
        d = {"a": [{"b": 1}, {"c": [2, {"d": 3}], "e": 4}, {"f": 5}], "g": 6}
        self.assertEqual(make_selector("$.a")(d), d["a"])
        self.assertEqual(make_selector("$.a[1].c")(d), d["a"][1]["c"])
        self.assertEqual(make_selector("$.a[1].c[1]")(d), d["a"][1]["c"][1])
        self.assertEqual(make_selector("$.a[1].c[1].d")(d), d["a"][1]["c"][1]["d"])
        self.assertEqual(make_selector("$.a[1].e")(d), d["a"][1]["e"])
        self.assertEqual(make_selector("$.g")(d), d["g"])

        with self.assertRaises(TypeError):
            make_selector("$.a[1].c[1].d[0]")(d)

        with self.assertRaises(KeyError):
            make_selector("$.z")(d)


class TestFasttextData(TestCase):
    def test_label_formatting(self):
        d = InputSpecWithMetadata(source="", text="", id="")  # noqa: E731
        fn = lambda x: FastTextDataConverter._make_label_fn(label_selector=x)  # noqa: E731
        self.assertEqual(fn("pos")(d), "__label__pos ")
        self.assertEqual(fn("pos,neg")(d), "__label__pos __label__neg ")
        self.assertEqual(fn("pos, neg")(d), "__label__pos __label__neg ")
        self.assertEqual(fn("POS__,NEG")(d), "__label__pos __label__neg ")

    def test_label_formatting_with_selector(self):
        d = lambda x: InputSpecWithMetadata(source=x, text="", id="")  # noqa: E731
        fn = lambda x: FastTextDataConverter._make_label_fn(label_selector=x)  # noqa: E731
        self.assertEqual(fn("$.source")(d("pos")), "__label__pos ")
        self.assertEqual(fn("$.source")(d("pos/neg")), "__label__pos_neg ")
        self.assertEqual(fn("$.source")(d("POS,NEG")), "__label__pos __label__neg ")
        self.assertEqual(fn("$.source")(d("___pos___")), "__label__pos ")

    def test_text_formatting(self):
        d = lambda x: InputSpecWithMetadata(source="", text=x, id="")  # noqa: E731
        fn = lambda x: FastTextDataConverter._make_text_fn(lowercase=x)  # noqa: E731
        self.assertEqual(fn(False)(d("hello world")), "hello world")
        self.assertEqual(fn(False)(d("Hello, World")), "Hello, World")
        self.assertEqual(fn(True)(d("Hello, World")), "hello, world")
        self.assertEqual(fn(False)(d("hello\nworld")), "hello world")
        self.assertEqual(fn(False)(d("hello\n\n\t world")), "hello world")

    def _load_expected(self, *paths: str, lowercase: bool = False) -> Tuple[List[str], List[str]]:
        expected_text = []
        expected_labels = []
        for path in paths:
            for fn in glob.glob(f"{path}/*"):
                with smart_open.open(fn) as f:
                    for ln in f:
                        data = json.loads(ln)
                        label = "__label__" + data["source"].lower().replace("-", "_")
                        text = re.sub(r"\s+", " ", data["text"]).strip()
                        if lowercase:
                            text = text.lower()
                        expected_labels.append(label)
                        expected_text.append(text)

        return expected_text, expected_labels

    def _load_output(self, dest: str, splits: Optional[Tuple[str, ...]] = None) -> Tuple[List[str], List[str]]:
        got_text: List[str] = []
        got_labels: List[str] = []
        splits = splits or ("train", "dev", "test")
        for split in splits:
            with open(os.path.join(dest, f"{split}.txt")) as f:
                for ln in f:
                    label, text = ln.strip().split(" ", 1)
                    got_text.append(text)
                    got_labels.append(label)
        return got_text, got_labels

    def test_fasttext_data_with_selector(self):
        # paths = {"pos": [str(DATA_DIR / "mutiple_files")], "neg": [str(DATA_DIR / "provided/documents")]}
        source_paths = str(DATA_DIR / "multiple_files")
        expected_text, _ = self._load_expected(source_paths, lowercase=False)

        with TemporaryDirectory() as tmpdir:
            FastTextDataConverter.make_stream(documents=[source_paths], output=tmpdir, debug=True, lowercase=False)

            texts, _ = self._load_output(tmpdir, splits=("train",))
            self.assertEqual(sorted(texts), sorted(expected_text))

            texts, _ = self._load_output(tmpdir, splits=("dev",))
            self.assertEqual(len(texts), 0)

            texts, _ = self._load_output(tmpdir, splits=("test",))
            self.assertEqual(len(texts), 0)

    def test_fasttext_data_with_selector_split(self):
        source_paths = str(DATA_DIR / "multiple_files")
        expected_text, expected_labels = self._load_expected(source_paths, lowercase=True)

        with TemporaryDirectory() as tmpdir:
            FastTextDataConverter.make_stream(
                documents=[source_paths],
                output=tmpdir,
                train_sample=0.5,
                dev_sample=0.2,
                test_sample=0.3,
                lowercase=True,
                debug=True,
            )
            got_text, got_labels = self._load_output(tmpdir)

            for i, (got, exp) in enumerate(zip(sorted(got_text), sorted(expected_text))):
                self.assertEqual(got, exp, f"got: {got[:40]}, expected: {exp[:40]}, index: {i}")
            for i, (got, exp) in enumerate(zip(sorted(got_labels), sorted(expected_labels))):
                self.assertEqual(got, exp, f"got: {got}, expected: {exp}, index: {i}")

    def test_fasttext_data_with_dict_split(self):
        documents = [str(DATA_DIR / "multiple_files"), str(DATA_DIR / "provided/documents")]
        expected_text, _ = self._load_expected(*documents, lowercase=True)

        with TemporaryDirectory() as tmpdir:
            FastTextDataConverter.make_stream(
                documents=documents,
                output=tmpdir,
                label_selector="pos",
                train_sample=0.5,
                dev_sample=0.2,
                test_sample=0.3,
                lowercase=True,
                num_processes=2,
            )

            got_text, got_labels = self._load_output(tmpdir)
            for i, (got, exp) in enumerate(zip(sorted(got_text), sorted(expected_text))):
                self.assertEqual(got, exp, f"got: {got[:40]}, expected: {exp[:40]}, index: {i}")
            self.assertEqual(set(got_labels), {"__label__pos"})
