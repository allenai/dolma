from pathlib import Path
from unittest import TestCase

from dolma.cli.__main__ import main

DEDUPE_BY_URL = Path(__file__).parent.parent / "config/dedupe-by-url.json"
DEDUPE_PARAGRAPHS = Path(__file__).parent.parent / "config/dedupe-paragraphs.json"


class TestDeduper(TestCase):
    def test_dedupe_by_url(self):
        main(argv=["-c", str(DEDUPE_BY_URL), "dedupe"])

    def test_dedupe_paragraphs(self):
        main(argv=["-c", str(DEDUPE_PARAGRAPHS), "dedupe"])
