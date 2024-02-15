# mypy: disable-error-code="unused-ignore"

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

import smart_open

from dolma.core.parallel import BaseParallelProcessor, QueueType

LOCAL_DATA = Path(__file__).parent.parent / "data"


class MockProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(cls, queue, /, cnt: int = 0):  # type: ignore[override]
        return super().increment_progressbar(queue, cnt=cnt)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs: Any,
    ):
        with smart_open.open(source_path, "rb") as f, smart_open.open(destination_path, "wb") as g:
            g.write(f.read())
        cls.increment_progressbar(queue, cnt=1)


class MockProcessorKwargs(MockProcessor):
    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs: Any,
    ):
        with smart_open.open(source_path, "rt") as rf:
            count_rows, count_chars = 0, 0
            for line in rf:
                count_rows += 1
                count_chars += len(json.loads(line).get("text", ""))

        out = {
            "source_path": source_path,
            "destination_path": destination_path,
            "count_rows": count_rows,
            "count_chars": count_chars,
            **kwargs,
        }

        with smart_open.open(destination_path, "wt") as g:
            json.dump(out, g)

        cls.increment_progressbar(queue, cnt=1)


class TestParallel(TestCase):
    def _kwargs_base(self, debug: bool = False, use_files_kwargs: bool = False, use_call_kwargs: bool = False):
        with TemporaryDirectory() as tmp_dir:
            src_dir = LOCAL_DATA / "expected"
            dest_dir = os.path.join(tmp_dir, "output")
            meta_dir = os.path.join(tmp_dir, "metadata")

            src = [os.path.join(src_dir, p) for p in os.listdir(src_dir) if not p.startswith(".")]
            dst = [f"{dest_dir}/{os.path.basename(p)}" for p in src]

            if use_call_kwargs:
                call_kwargs = {"first_extra_kwarg": "first", "second_extra_kwarg": 1}
            else:
                call_kwargs = {}

            if use_files_kwargs:
                files_kwargs = [{"count": i} for i in range(len(src))]
            else:
                files_kwargs = None

            proc = MockProcessorKwargs(
                source_prefix=src,
                destination_prefix=[dest_dir for _ in src],
                metadata_prefix=[meta_dir for _ in src],
                process_single_kwargs=files_kwargs,
                ignore_existing=False,
                num_processes=2,
                debug=debug,
            )
            proc(**call_kwargs)

            for i in range(len(src)):
                src_path = src[i]
                dst_path = dst[i]

                expected_kwargs: dict = {**call_kwargs}
                if files_kwargs is not None:
                    expected_kwargs.update(files_kwargs[i])

                with smart_open.open(dst_path, "rt") as rf:
                    dst_data = json.load(rf)
                with smart_open.open(src_path, "rt") as rf:
                    src_data = [json.loads(ln) for ln in rf]

                self.assertEqual(dst_data["source_path"], src_path)
                self.assertEqual(dst_data["destination_path"], dst_path)
                self.assertEqual(dst_data["count_rows"], len(src_data))
                self.assertEqual(dst_data["count_chars"], sum(len(d.get("text", "")) for d in src_data))
                for k, v in expected_kwargs.items():
                    self.assertIn(k, dst_data)
                    self.assertEqual(dst_data[k], v)

    def test_base_parallel_processor_kwargs_file_kwargs(self, debug: bool = False):
        self._kwargs_base(use_files_kwargs=True, use_call_kwargs=False, debug=debug)

    def test_base_parallel_processor_kwargs_call_kwargs(self, debug: bool = False):
        self._kwargs_base(use_files_kwargs=False, use_call_kwargs=True, debug=debug)

    def test_base_parallel_processor_kwargs_file_call_kwargs(self, debug: bool = False):
        self._kwargs_base(use_files_kwargs=True, use_call_kwargs=True, debug=debug)

    def test_base_parallel_processor(self, debug: bool = False):
        with self.assertRaises(ValueError):
            MockProcessor(source_prefix=[], destination_prefix=[], metadata_prefix=[])

        with TemporaryDirectory() as d:
            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
                num_processes=2,
                debug=debug,
            )
            proc()
            src = [p for p in os.listdir(LOCAL_DATA / "expected") if not p.startswith(".")]
            meta = [p.rstrip(".done.txt") for p in os.listdir(f"{d}/metadata")]
            dest = [p for p in os.listdir(f"{d}/destination") if not p.startswith(".")]

            self.assertEqual(sorted(src), sorted(meta))
            self.assertEqual(sorted(src), sorted(dest))

            for p in src:
                with smart_open.open(LOCAL_DATA / "expected" / p, "rb") as f:
                    with smart_open.open(f"{d}/destination/{p}", "rb") as g:
                        self.assertEqual(f.read(), g.read())

    def test_base_parallel_processor_selector(self, debug: bool = False):
        with TemporaryDirectory() as d:
            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected" / "*-paragraphs.*"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
                num_processes=2,
                debug=debug,
            )
            proc()
            src = [p for p in os.listdir(LOCAL_DATA / "expected") if "paragraphs" in p]
            meta = [p.rstrip(".done.txt") for p in os.listdir(f"{d}/metadata")]
            dest = [p for p in os.listdir(f"{d}/destination")]
            self.assertEqual(sorted(src), sorted(meta))
            self.assertEqual(sorted(src), sorted(dest))

    def test_base_parallel_processor_kwargs_file_kwargs_debug(self):
        self.test_base_parallel_processor_kwargs_file_kwargs(debug=True)

    def test_base_parallel_processor_kwargs_call_kwargs_debug(self):
        self.test_base_parallel_processor_kwargs_call_kwargs(debug=True)

    def test_base_parallel_processor_kwargs_file_call_kwargs_debug(self):
        self.test_base_parallel_processor_kwargs_file_call_kwargs(debug=True)

    def test_base_parallel_processor_debug(self):
        self.test_base_parallel_processor(debug=True)

    def test_base_parallel_processor_selector_debug(self):
        self.test_base_parallel_processor_selector(debug=True)
