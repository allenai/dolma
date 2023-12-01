import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Optional
from unittest import TestCase
from uuid import uuid4

import smart_open

from dolma.cli.__main__ import main

from .utils import (
    clean_test_data,
    download_s3_prefix,
    get_test_prefix,
    load_jsonl,
    skip_aws_tests,
    upload_s3_prefix,
)

DEDUPE_BY_URL = Path(__file__).parent.parent / "config/dedupe-by-url.json"
DEDUPE_PARAGRAPHS = Path(__file__).parent.parent / "config/dedupe-paragraphs.json"


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
                s3_prefix=f"{self.remote_test_prefix}", local_prefix="tests/data/provided/documents/*.gz"
            )

        # copy provided config files to local temp dir
        shutil.copytree(
            "tests/data/provided/documents",
            f"{self.local_temp_dir}/tests/data/provided/documents",
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
        computed = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/attributes/dedupe_by_url/000.json.gz")
        self.assertEqual(expected, computed)

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
            f"{self.local_temp_dir}/tests/data/provided/attributes/dedupe_paragraphs/000.json.gz"
        )
        self.assertEqual(expected, computed)

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
        computed = load_jsonl(f"{self.local_temp_dir}/tests/data/provided/attributes/dedupe_by_url/000.json.gz")
        self.assertEqual(expected, computed)


class TestAdvancedDeduper(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_docs(self, docs: List[str], ext_dir: Optional[Path] = None) -> Path:
        encoded_docs = [{"id": str(i), "text": d, "source": __file__} for i, d in enumerate(docs)]
        dir_name = uuid4()
        file_name = uuid4()
        fp = Path(self.temp_dir.name) / f"{dir_name}/documents/{file_name}.jsonl.gz"
        with smart_open.open(fp, "w") as f:
            for doc in encoded_docs:
                f.write(json.dumps(doc) + "\n")
        return fp

    def read_docs(self, fp: Path) -> List[str]:
        ...

    def test_skip_empty(self):
        documents = ["Short document\n" + "\n" + "More text\n", "Short document #2\n" + "\n" + "More text\n"]

        docs_fp = self.write_docs(documents)

        config = {
            "documents": [str(docs_fp)],
            "attributes": [
                {
                    "name": "dedupe_paragraphs",
                    "type": "dedupe",
                    "params": {"dedupe_by": "paragraphs", "skip_empty": True},
                }
            ],
        }
