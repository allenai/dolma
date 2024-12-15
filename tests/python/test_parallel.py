# mypy: disable-error-code="unused-ignore"

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any
from unittest import TestCase

import smart_open

from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.progressbar import BaseProgressBar

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


class MockPbar(BaseProgressBar):
    a: int = 0
    b: int = 0


class NewStyleMockProcessor(BaseParallelProcessor):
    PROGRESS_BAR_CLS = MockPbar

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs: Any,
    ):
        with MockPbar(queue) as pbar:
            for _ in range(10):
                pbar.a += 1
                pbar.b += 5


class MockProcessorWithFail(MockProcessor):
    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs: Any,
    ):
        sleep(1)
        raise ValueError(f"Failed on {source_path}")


class TestParallel(TestCase):
    def _read(self, path):
        with smart_open.open(path, "rb") as f:
            return f.read()

    def test_new_style(self):
        with TemporaryDirectory() as d:
            proc = NewStyleMockProcessor(
                source_prefix=str(LOCAL_DATA / "expected"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
            )
            proc()

    def test_debug(self):
        with self.assertRaises(ValueError):
            MockProcessor(source_prefix=[], destination_prefix=[], metadata_prefix=[])

        with TemporaryDirectory() as d:
            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
                debug=True,
            )
            proc()
            src = [p for p in os.listdir(LOCAL_DATA / "expected") if not p.startswith(".")]
            meta = [p.rstrip(".done.txt") for p in os.listdir(f"{d}/metadata")]
            dest = [p for p in os.listdir(f"{d}/destination") if not p.startswith(".")]
            self.assertEqual(sorted(src), sorted(meta))
            self.assertEqual(sorted(src), sorted(dest))

            for s, e in zip(src, dest):
                s_ = LOCAL_DATA / "expected" / s
                e_ = f"{d}/destination/{e}"
                self.assertEqual(self._read(s_), self._read(e_))

    def test_base_parallel_processor(self):
        with self.assertRaises(ValueError):
            MockProcessor(source_prefix=[], destination_prefix=[], metadata_prefix=[])

        with TemporaryDirectory() as d:
            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
                num_processes=2,
            )
            proc()
            src = [p for p in os.listdir(LOCAL_DATA / "expected") if not p.startswith(".")]
            meta = [p.rstrip(".done.txt") for p in os.listdir(f"{d}/metadata")]
            dest = [p for p in os.listdir(f"{d}/destination") if not p.startswith(".")]
            self.assertEqual(sorted(src), sorted(meta))
            self.assertEqual(sorted(src), sorted(dest))

            for s, e in zip(src, dest):
                s_ = LOCAL_DATA / "expected" / s
                e_ = f"{d}/destination/{e}"
                self.assertEqual(self._read(s_), self._read(e_))

    def test_two_stages(self):
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

            proc = MockProcessor(
                source_prefix=str(LOCAL_DATA / "expected" / "*"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
            )
            proc()

            # the oldest two files are from the first stage
            dest2 = sorted(
                [p for p in os.listdir(f"{d}/destination")], key=lambda x: os.stat(f"{d}/destination/{x}").st_ctime
            )
            self.assertEqual(sorted(dest), sorted(dest2[:2]))

    def test_failure(self):
        with TemporaryDirectory() as d:
            proc = MockProcessorWithFail(
                source_prefix=str(LOCAL_DATA / "expected"),
                destination_prefix=f"{d}/destination",
                metadata_prefix=f"{d}/metadata",
                ignore_existing=False,
                backoff_exceptions=(ValueError,),
                backoff_max_time=3,
                backoff_max_tries=3,
                debug=True,
            )
            with self.assertRaises(ValueError):
                proc()
            self.assertEqual(len(os.listdir(f"{d}/destination")), 0)
            self.assertEqual(len(os.listdir(f"{d}/metadata")), 0)
