import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, List, Tuple, TypeVar, Union
from unittest import TestCase

import smart_open
from typing_extensions import TypedDict

from dolma.cli.__main__ import main
from dolma.core.utils import split_words

from .utils import (
    TestCasePipeline,
    clean_test_data,
    download_s3_prefix,
    get_test_prefix,
    load_jsonl,
    skip_aws_tests,
    upload_s3_prefix,
)

TEST_DIR = Path(__file__).parent.parent
DEDUPE_BY_URL = TEST_DIR / "config/dedupe-by-url.json"
DEDUPE_PARAGRAPHS = TEST_DIR / "config/dedupe-paragraphs.json"
DEDUPE_PARAGRAPH_NGRAMS = TEST_DIR / "config/dedupe-paragraph-ngrams.json"


D = TypeVar("D", bound="DedupeAttributesDict")


class DedupeAttributesDict(TypedDict):
    id: str
    attributes: Dict[str, List[Tuple[int, int, Union[int, float]]]]


class TestDeduper(TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.local_temp_dir = self.stack.enter_context(TemporaryDirectory()).rstrip("/")

        if skip_aws_tests():
            self.remote_test_prefix = None
        else:
            self.remote_test_prefix = get_test_prefix()

            # upload test data
            upload_s3_prefix(
                s3_prefix=f"{self.remote_test_prefix}", local_prefix="tests/data/provided/deduper/documents/*.gz"
            )

        # copy provided config files to local temp dir
        shutil.copytree(
            "tests/data/provided/deduper/documents",
            f"{self.local_temp_dir}/tests/data/provided/deduper/documents",
            dirs_exist_ok=True,
        )

    def tearDown(self) -> None:
        if self.remote_test_prefix is not None:
            clean_test_data(self.remote_test_prefix)
        self.stack.close()

    def test_dedupe_by_url(self):
        with open(DEDUPE_BY_URL, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.local_temp_dir}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        expected = load_jsonl("tests/data/expected/dedupe-by-url.json.gz")
        computed = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_by_url/000.json.gz"
        )
        return self._compare_dedupe_output(expected, computed)  # pyright: ignore

    def test_dedupe_paragraphs(self):
        with open(DEDUPE_PARAGRAPHS, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.local_temp_dir}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        expected = load_jsonl("tests/data/expected/dedupe-paragraphs.json.gz")
        computed = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_paragraphs/000.json.gz"
        )
        return self._compare_dedupe_output(expected, computed)  # pyright: ignore

    def test_dedupe_paragraphs_change_splitter(self):
        with open(DEDUPE_PARAGRAPHS, "r") as f:
            config = json.load(f)

        config["documents"] = [f'{self.local_temp_dir}/{config["documents"][0]}']
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        split_seq = "tt"

        # separate on characters "tt" instead of "\n"
        config["dedupe"]["paragraphs"]["paragraph_separator"] = split_seq

        # this will ensure that the deduper will output something for each paragraph
        config["dedupe"]["paragraphs"]["by_ngram"] = {"ngram_length": 1, "stride": 1, "overlap_threshold": 0.0}

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        documents = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/deduper/documents/000.json.gz")
        attributes = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_paragraphs/000.json.gz"
        )
        for doc, attr in zip(documents, attributes):
            self.assertEqual(
                len(doc["text"].split(split_seq)), len(attr["attributes"]["bff_duplicate_paragraph_spans"])
            )

    def test_dedupe_paragraphs_stride_math(self):
        with open(DEDUPE_PARAGRAPHS, "r") as f:
            config = json.load(f)

        config["documents"] = [f'{self.local_temp_dir}/{config["documents"][0]}']
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        # this will ensure that the deduper will output something for each paragraph
        config["dedupe"]["paragraphs"]["by_ngram"] = {"ngram_length": 10, "stride": 5, "overlap_threshold": 0.0}

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        documents = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/deduper/documents/000.json.gz")
        attributes = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_paragraphs/000.json.gz"
        )
        for doc, attr in zip(documents, attributes):
            valid_paragraphs = []
            i = 0
            for para in doc["text"].split("\n"):
                j = min(i + len(para) + 1, len(doc["text"]))
                valid_paragraphs.append((i, j))
                i = j
            spans = attr["attributes"]["bff_duplicate_paragraph_spans"]

            self.assertEqual(len(valid_paragraphs), len(spans))
            for (start_para, end_para), (start_span, end_span, _) in zip(valid_paragraphs, spans):
                self.assertEqual(doc["text"][start_para:end_para], doc["text"][start_span:end_span])

    def test_dedupe_paragraphs_stride_math_skip_short(self):
        with open(DEDUPE_PARAGRAPHS, "r") as f:
            config = json.load(f)

        config["documents"] = [f'{self.local_temp_dir}/{config["documents"][0]}']
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        # this will ensure that the deduper will output something for each paragraph
        config["dedupe"]["paragraphs"]["by_ngram"] = (
            ng_cfg := {"ngram_length": 20, "stride": 5, "overlap_threshold": 0.0, "skip_short_paragraphs": True}
        )

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        documents = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/deduper/documents/000.json.gz")
        attributes = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_paragraphs/000.json.gz"
        )
        for doc, attr in zip(documents, attributes):
            valid_paragraphs = []
            i = 0
            for para in doc["text"].split("\n"):
                j = min(i + len(para) + 1, len(doc["text"]))
                if len(split_words(para)) >= ng_cfg["ngram_length"]:
                    valid_paragraphs.append((i, j))
                i = j
            spans = attr["attributes"]["bff_duplicate_paragraph_spans"]

            self.assertEqual(len(valid_paragraphs), len(spans))

            for (start_para, end_para), (start_span, end_span, _) in zip(valid_paragraphs, spans):
                self.assertEqual(doc["text"][start_para:end_para], doc["text"][start_span:end_span])

    def test_dedupe_paragraph_ngrams(self):
        with open(DEDUPE_PARAGRAPH_NGRAMS, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.local_temp_dir}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        expected = load_jsonl("tests/data/expected/dedupe-paragraph-ngrams.json.gz")
        print(
            f"Loading data from {self.local_temp_dir}/tests/data/provided/attributes/dedupe_paragraph_ngrams/000.json.gz"
        )
        computed = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_paragraph_ngrams/000.json.gz"
        )
        return self._compare_dedupe_output(expected, computed)  # pyright: ignore

    def _compare_dedupe_output(self, expected: List[D], computed: List[D]):
        self.assertEqual(len(expected), len(computed))
        for exp_row, comp_row in zip(expected, computed):
            self.assertEqual(exp_row["id"], comp_row["id"])
            self.assertEqual(exp_row["attributes"].keys(), comp_row["attributes"].keys())
            for attr in exp_row["attributes"].keys():
                for exp_span, comp_span in zip(exp_row["attributes"][attr], comp_row["attributes"][attr]):
                    self.assertEqual(exp_span, comp_span)

    def test_dedupe_by_url_remote_input(self):
        if self.remote_test_prefix is None:
            return self.skipTest("Skipping AWS tests")

        with open(DEDUPE_BY_URL, "r") as f:
            config = json.load(f)

        config["documents"][0] = f'{self.remote_test_prefix}/{config["documents"][0]}'
        config["bloom_filter"]["file"] = f'{self.local_temp_dir}/{config["bloom_filter"]["file"]}'

        with NamedTemporaryFile("w") as f:
            json.dump(config, f)
            f.flush()

            main(argv=["-c", f.name, "dedupe"])

        download_s3_prefix(self.remote_test_prefix, self.local_temp_dir)

        expected = load_jsonl("tests/data/expected/dedupe-by-url.json.gz")
        computed = load_jsonl(
            f"{self.local_temp_dir}/tests/data/provided/deduper/attributes/dedupe_by_url/000.json.gz"
        )
        return self._compare_dedupe_output(expected, computed)  # pyright: ignore


class TestDeduperPipeline(TestCasePipeline):
    def test_skip_empty(self):
        duplicate_text = "More text"
        documents = [
            self.combineIntoDoc("Short document", "", duplicate_text),
            self.combineIntoDoc("Short document #2", "", duplicate_text),
        ]

        docs_fp = self.writeDocs(documents)
        key_name = "dedupe_paragraphs"
        attribute_name = "bff_duplicate_paragraph_spans"

        config = {
            "documents": docs_fp,
            "dedupe": {
                "name": key_name,
                "paragraphs": {"attribute_name": attribute_name},
                "skip_empty": True,
            },
            "bloom_filter": {
                "file": self.makeUniquePath(),
                "read_only": False,
                "estimated_doc_count": 100,
                "desired_false_positive_rate": 1e-06,
            },
            "processes": 1,
        }

        config_path = self.writeConfig(config)

        main(argv=["-c", config_path, "dedupe"])

        expected = self.readUnits([p.replace("documents", f"attributes/{key_name}") for p in docs_fp])
        self.assertEqual(len(expected), 2)

        # no duplicate on first doc
        self.assertIn("attributes", expected[0])
        self.assertIn(attribute_name, expected[0]["attributes"])
        self.assertEqual(expected[0]["attributes"][attribute_name], [])

        # duplicate on second doc
        self.assertIn("attributes", expected[1])
        self.assertIn(attribute_name, expected[1]["attributes"])
        self.assertEqual(len(expected[1]["attributes"][attribute_name]), 1)
        (start, end, score), *_ = expected[1]["attributes"][attribute_name]
        self.assertEqual(documents[1][start:end], duplicate_text)
        self.assertEqual(score, 1.0)

        # now let's not skip empty docs
        config["dedupe"]["skip_empty"] = False
        config["bloom_filter"]["file"] = self.makeUniquePath()  # new filter
        config_path = self.writeConfig(config)

        main(argv=["-c", config_path, "dedupe"])

        expected = self.readUnits([p.replace("documents", f"attributes/{key_name}") for p in docs_fp])

        # two duplicates on second doc
        self.assertEqual(len(expected[1]["attributes"][attribute_name]), 2)
        (s1, e1, v1), (s2, e2, v2) = expected[1]["attributes"][attribute_name]
        self.assertEqual(documents[1][s1:e1], "\n")
        self.assertEqual(v1, 1.0)
        self.assertEqual(documents[1][s2:e2], duplicate_text)
        self.assertEqual(v2, 1.0)


class TestDeduperUnicode(TestCase):
    def test_unicode(self):
        paragraphs = [
            "This is a paragraph that has no duplication.",
            "This is a paragraph that will be duplicated.",
            "This is a paragraph that will be duplicated.",
            "This paragraph has emojis 😊",
            "This paragraph has emojis 🀜 and is duplicated",
            "This paragraph has emojis 🀜 and is duplicated",
            "Wrapping up with one more unique paragraph.",
        ]
        text = "\n".join(paragraphs)

        with TemporaryDirectory() as d:
            (documents_dir := Path(d) / "documents").mkdir(exist_ok=True, parents=True)
            with smart_open.open(f"{documents_dir}/test.jsonl.gz", "wt", encoding="utf-8") as f:
                f.write(json.dumps({"id": "1", "text": text, "source": __file__}) + "\n")

            (attributes_dir := Path(d) / "attributes").mkdir(exist_ok=True, parents=True)
            config = {
                "documents": [f"{documents_dir}/test.jsonl.gz"],
                "dedupe": {
                    "name": "dedupe_paragraphs",
                    "paragraphs": {"attribute_name": "bff_duplicate_paragraph_spans"},
                    "skip_empty": True,
                },
                "bloom_filter": {
                    "file": f"{d}/bloom_filter",
                    "read_only": False,
                    "estimated_doc_count": 100,
                    "desired_false_positive_rate": 1e-06,
                },
                "processes": 1,
            }

            with smart_open.open(f"{d}/config.json", "wt", encoding="utf-8") as f:
                f.write(json.dumps(config))

            main(argv=["-c", f"{d}/config.json", "dedupe"])

            attributes = load_jsonl(f"{attributes_dir}/dedupe_paragraphs/test.jsonl.gz")

            first_dup, second_dup = attributes[0]["attributes"]["bff_duplicate_paragraph_spans"]

            self.assertEqual(text[first_dup[0] : first_dup[1]], paragraphs[2] + "\n")
            self.assertEqual(text[second_dup[0] : second_dup[1]], paragraphs[5] + "\n")

    def test_unicode_fuzzy(self):
        paragraphs = [
            "This is a paragraph that has no duplication.",
            "This is a paragraph that will be duplicated.",
            "This is a paragraph that will be duplicated.",
            "This paragraph has emojis 😊",
            "This paragraph has emojis 🀜 and is duplicated",
            "This paragraph has emojis 🀜 and is duplicated",
            "Some more info here.",
            "Text for thought",
            "Beeeee 🃚",
            "Wrapping up with one more unique paragraph.",
            "Bang!",
        ]
        text = "\n".join(paragraphs)

        with TemporaryDirectory() as d:
            (documents_dir := Path(d) / "documents").mkdir(exist_ok=True, parents=True)
            with smart_open.open(f"{documents_dir}/test.jsonl.gz", "wt", encoding="utf-8") as f:
                f.write(json.dumps({"id": "1", "text": text, "source": __file__}) + "\n")

            (attributes_dir := Path(d) / "attributes").mkdir(exist_ok=True, parents=True)
            config = {
                "documents": [f"{documents_dir}/test.jsonl.gz"],
                "dedupe": {
                    "name": "dedupe_paragraphs",
                    "paragraphs": {
                        "attribute_name": "bff_duplicate_paragraph_spans",
                        "by_ngram": {"ngram_length": 5, "stride": 1, "overlap_threshold": 0.0},
                    },
                    "skip_empty": True,
                },
                "bloom_filter": {
                    "file": f"{d}/bloom_filter",
                    "read_only": False,
                    "estimated_doc_count": 100,
                    "desired_false_positive_rate": 1e-06,
                },
                "processes": 1,
            }

            with smart_open.open(f"{d}/config.json", "wt", encoding="utf-8") as f:
                f.write(json.dumps(config))

            main(argv=["-c", f"{d}/config.json", "dedupe"])

            attributes = load_jsonl(f"{attributes_dir}/dedupe_paragraphs/test.jsonl.gz")

            self.assertEqual(len(attributes[0]["attributes"]["bff_duplicate_paragraph_spans"]), len(paragraphs))

            for para, (start, end, _) in zip(
                paragraphs, attributes[0]["attributes"]["bff_duplicate_paragraph_spans"]
            ):
                if para == paragraphs[-1]:
                    # TODO: fix last paragraph not having a newline
                    self.assertEqual(text[start:end], para)
                else:
                    self.assertEqual(text[start:end], para + "\n")
