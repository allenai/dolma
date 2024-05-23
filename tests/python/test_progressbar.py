from queue import Queue
from time import sleep
from unittest import TestCase

from pytest import CaptureFixture

from dolma.core.parallel import BaseProgressBar, QueueType


class CustomProgressbar(BaseProgressBar):
    documents: int = 0
    files: int = 0


class TestProgressbar(TestCase):
    def test_values(self):
        queue: QueueType = Queue()
        updater = CustomProgressbar(queue, min_time=1)
        updater.documents += 1
        updater.files += 1

        # was pushed to the queue
        self.assertEqual(updater.documents, 0)

        # hasn't been pushed yet cuz delta time is small
        self.assertEqual(updater.files, 1)

        # pull from queue to verify that element was added
        element = updater.parse(queue.get())
        self.assertIn("documents", element)
        self.assertIn("files", element)
        self.assertEqual(element["documents"], 1)
        self.assertEqual(element["files"], 0)

        # do another update, these should go in queue right away
        sleep(updater._update_every_seconds)
        updater.files += 1
        element = updater.parse(queue.get())
        self.assertEqual(element["documents"], 0)
        self.assertEqual(element["files"], 2)

        # verify there is nothing is pbar queue
        self.assertTrue(queue.empty())
        self.assertEqual(updater.documents, 0)
        self.assertEqual(updater.files, 0)

    def test_error(self):
        queue: QueueType = Queue()
        with self.assertRaises(ValueError):
            _ = BaseProgressBar(queue)


def test_progressbar_in_thread(capsys: CaptureFixture):
    queue: QueueType = Queue()

    with CustomProgressbar(queue, thread=True) as pbar:
        for _ in range(5):
            pbar.documents += 1
            pbar.files += 1

    sleep(1.0)
    assert queue.empty(), "Queue should be empty"

    captured = capsys.readouterr()
    pbars_text = [ln.strip() for ln in captured.err.split("\n") if ln.strip()]
    assert len(pbars_text) >= 2, "At least 2 progress bars should have been printed"

    *_, last_files, last_docs = pbars_text
    if "files" in last_docs:
        last_files, last_docs = last_docs, last_files

    assert last_files.startswith("files"), "Second to last line should be about files"
    assert last_docs.startswith("documents"), "Last line should be about documents"

    assert "5.00f" in last_files, "Last line should have 5 files"
    assert "5.00d" in last_docs, "Last line should have 5 documents"
