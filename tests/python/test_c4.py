"""

Unit tests for taggers/*.py

@kylel

"""

from unittest import TestCase

from dolma.core.data_types import Document
from dolma.taggers.c4 import C4Tagger, FasterC4Tagger


class TestC4Tagger(TestCase):
    def setUp(self):
        self.tagger = C4Tagger()

    def test_curly_braces_filter(self):
        doc = Document(source="", version="", id="", text="This is a test.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual([r.to_json() for r in result.spans if r.type == "has_curly_brace"], [])

        doc = Document(source="", version="", id="", text="This is a test {")
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "has_curly_brace"],
            [{"start": 0, "end": 16, "type": "has_curly_brace", "score": 1.0}],
        )

    def test_javascript_filter(self):
        doc = Document(source="", version="", id="", text="This is a test.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual([r.to_json() for r in result.spans if r.type == "has_javascript"], [])

        doc = Document(source="", version="", id="", text="This is a test javascript")
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "has_javascript"],
            [{"start": 0, "end": 25, "type": "has_javascript", "score": 1.0}],
        )

    def test_lorem_ipsum_filter(self):
        doc = Document(source="", version="", id="", text="This is a test.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual([r.to_json() for r in result.spans if r.type == "has_lorem_ipsum"], [])

        doc = Document(source="", version="", id="", text="This is a lorem ipsum test\nmore test.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "has_lorem_ipsum"],
            # the end should be 37 because lorem ipsum is detected at the document level
            [{"start": 0, "end": 37, "type": "has_lorem_ipsum", "score": 1.0}],
        )

    def test_line_ends_with_no_punctuation(self):
        doc = Document(
            source="",
            version="",
            id="",
            text="This is a test.\nIt has more;\nA trailing space! \nShould be good on this one.\nThis one is bad\n",
        )
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "lines_with_no_ending_punctuation"],
            [
                {"start": 16, "end": 29, "type": "lines_with_no_ending_punctuation", "score": 1.0},
                {"start": 76, "end": 92, "type": "lines_with_no_ending_punctuation", "score": 1.0},
                {"start": 92, "end": 92, "type": "lines_with_no_ending_punctuation", "score": 1.0},
            ],
        )

    def test_lines_with_too_few_words(self):
        doc = Document(
            source="",
            version="",
            id="",
            text="Short!\nThis short.\nBarely above the limit!\nthis is last frontier.",
        )

        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "lines_with_too_few_words"],
            [
                {"start": 0, "end": 7, "type": "lines_with_too_few_words", "score": 1.0},
                {"start": 7, "end": 19, "type": "lines_with_too_few_words", "score": 1.0},
            ],
        )

    def test_naughty_words(self):
        doc = Document(source="", version="", id="", text="This sentence has no bad words.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual([r.to_json() for r in result.spans if r.type == "has_naughty_word"], [])

        doc = Document(source="", version="", id="", text="This sentence mentions viagra as a bad word.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "has_naughty_word"],
            [{"start": 0, "end": len(doc.text), "type": "has_naughty_word", "score": 1.0}],
        )

        doc = Document(
            source="", version="", id="", text="This sentence has ass, but not a bad word because of comma."
        )
        result = self.tagger.predict(doc=doc)
        self.assertEqual([r.to_json() for r in result.spans if r.type == "has_naughty_word"], [])

        doc = Document(
            source="",
            version="",
            id="",
            text="If I say strap on because we are on a plane, it's still a bad word.",
        )
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [r.to_json() for r in result.spans if r.type == "has_naughty_word"],
            [{"start": 0, "end": len(doc.text), "type": "has_naughty_word", "score": 1.0}],
        )


class TestFasterC4Tagger(TestC4Tagger):
    def setUp(self):
        self.tagger = FasterC4Tagger()
