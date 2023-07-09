from pathlib import Path
from unittest import TestCase

from dolma.cli.__main__ import main

C4_CLEANED = Path(__file__).parent.parent / "config/c4-cleaned.json"
EMAIL_SPANS = Path(__file__).parent.parent / "config/email-spans.json"
FILTER_BY_SPANS = Path(__file__).parent.parent / "config/filter-by-spans.json"
MIXER = Path(__file__).parent.parent / "config/mixer.json"
PARAGRAPH_SPANS = Path(__file__).parent.parent / "config/paragraph-spans.json"


class TestDeduper(TestCase):
    def test_email_spans(self):
        main(argv=["-c", str(EMAIL_SPANS), "mix"])

    def test_filter_by_spans(self):
        main(argv=["-c", str(FILTER_BY_SPANS), "mix"])

    def test_mixer(self):
        main(argv=["-c", str(MIXER), "mix"])

    def test_paragraph_spans(self):
        main(argv=["-c", str(PARAGRAPH_SPANS), "mix"])
