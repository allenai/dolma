# mypy: disable-error-code="unused-ignore"

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
        queue.put((1,))


class TestParallel(TestCase):
    def test_base_parallel_processor(self):
        with self.assertRaises(ValueError):
            MockProcessor(source_prefix=[], destination_prefix=[], metadata_prefix=[])

        with TemporaryDirectory() as d:
            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
            )
            proc()
            src = [p for p in os.listdir(LOCAL_DATA / "expected") if not p.startswith(".")]
            meta = [p.rstrip(".done.txt") for p in os.listdir(f"{d}/metadata")]
            dest = [p for p in os.listdir(f"{d}/destination") if not p.startswith(".")]
            self.assertEqual(sorted(src), sorted(meta))
            self.assertEqual(sorted(src), sorted(dest))

        with TemporaryDirectory() as d:
            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected" / "*-paragraphs.*"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
            )
            proc()
            src = [p for p in os.listdir(LOCAL_DATA / "expected") if "paragraphs" in p]
            meta = [p.rstrip(".done.txt") for p in os.listdir(f"{d}/metadata")]
            dest = [p for p in os.listdir(f"{d}/destination")]
            self.assertEqual(sorted(src), sorted(meta))
            self.assertEqual(sorted(src), sorted(dest))
