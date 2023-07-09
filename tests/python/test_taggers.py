"""

Unit tests for taggers/*.py

@kylel

"""

from unittest import TestCase

from dolma.core.data_types import Document
from dolma.taggers.gopher import GopherTagger


class TestGopherTagger(TestCase):
    def test_predict_short(self):
        tagger = GopherTagger()
        doc = Document(source="", version="", id="", text="This is a test.")
        doc_result = tagger.predict(doc=doc)
        d = doc_result.to_json()
        self.assertEqual(len(d["spans"]), 13)
        self.assertEqual(
            d["spans"][0],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_characters_in_most_common_2grams",
                "score": 0.5,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][1],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_characters_in_most_common_3grams",
                "score": 0.5833333333333334,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][2],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_characters_in_most_common_4grams",
                "score": 1.0,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][3],
            {"start": 0, "end": 15, "type": "character_count", "score": 15.0, "mention": "This is a test."},
        )
        self.assertEqual(
            d["spans"][4],
            {"start": 0, "end": 15, "type": "word_count", "score": 4.0, "mention": "This is a test."},
        )
        self.assertEqual(
            d["spans"][5],
            {"start": 0, "end": 15, "type": "median_word_length", "score": 3.0, "mention": "This is a test."},
        )
        self.assertEqual(
            d["spans"][6],
            {"start": 0, "end": 15, "type": "symbol_to_word_ratio", "score": 0.0, "mention": "This is a test."},
        )
        self.assertEqual(
            d["spans"][7],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_words_with_alpha_character",
                "score": 1.0,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][8],
            {"start": 0, "end": 15, "type": "required_word_count", "score": 0.0, "mention": "This is a test."},
        )
        self.assertEqual(
            d["spans"][9],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_lines_starting_with_bullet_point",
                "score": 0.0,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][10],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_lines_ending_with_ellipsis",
                "score": 0.0,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][11],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_duplicate_lines",
                "score": 0.0,
                "mention": "This is a test.",
            },
        )
        self.assertEqual(
            d["spans"][12],
            {
                "start": 0,
                "end": 15,
                "type": "fraction_of_characters_in_duplicate_lines",
                "score": 0.0,
                "mention": "This is a test.",
            },
        )

    def test_predict_multiline(self):
        tagger = GopherTagger()
        text = "This is a sentence. \n  \n This is another sentence.\n\n  This is a third sentence."
        doc = Document(source="", version="", id="", text=text)
        doc_result = tagger.predict(doc=doc)
        d = doc_result.to_json()
        self.assertEqual(len(d["spans"]), 19)
        self.assertEqual(
            d["spans"][0],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_most_common_2grams",
                "score": 0.3050847457627119,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][1],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_most_common_3grams",
                "score": 0.23728813559322035,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][2],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_most_common_4grams",
                "score": 0.2711864406779661,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][3],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_5grams",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][4],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_6grams",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][5],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_7grams",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][6],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_8grams",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][7],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_9grams",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][8],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_10grams",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][9], {"start": 0, "end": 79, "type": "character_count", "score": 79.0, "mention": text}
        )
        self.assertEqual(
            d["spans"][10], {"start": 0, "end": 79, "type": "word_count", "score": 13.0, "mention": text}
        )
        self.assertEqual(
            d["spans"][11], {"start": 0, "end": 79, "type": "median_word_length", "score": 4.0, "mention": text}
        )
        self.assertEqual(
            d["spans"][12], {"start": 0, "end": 79, "type": "symbol_to_word_ratio", "score": 0.0, "mention": text}
        )
        self.assertEqual(
            d["spans"][13],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_words_with_alpha_character",
                "score": 1.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][14], {"start": 0, "end": 79, "type": "required_word_count", "score": 0.0, "mention": text}
        )
        self.assertEqual(
            d["spans"][15],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_lines_starting_with_bullet_point",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][16],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_lines_ending_with_ellipsis",
                "score": 0.0,
                "mention": text,
            },
        )
        self.assertEqual(
            d["spans"][17],
            {"start": 0, "end": 79, "type": "fraction_of_duplicate_lines", "score": 0.0, "mention": text},
        )
        self.assertEqual(
            d["spans"][18],
            {
                "start": 0,
                "end": 79,
                "type": "fraction_of_characters_in_duplicate_lines",
                "score": 0.0,
                "mention": text,
            },
        )

    def test_word_count_is_whitespace_sep(self):
        tagger = GopherTagger()
        text = "T h i s \n    \n\n\n    isoneword !!!"
        doc = Document(source="", version="", id="", text=text)
        doc_result = tagger.predict(doc=doc)
        d = doc_result.to_json()
        self.assertEqual(d["spans"][6]["type"], "word_count")
        self.assertEqual(d["spans"][6]["score"], 6.0)

    def test_required_word_count(self):
        tagger = GopherTagger()
        text = "The.and.that"
        doc = Document(source="", version="", id="", text=text)
        doc_result = tagger.predict(doc=doc)
        d = doc_result.to_json()
        self.assertEqual(d["spans"][5]["type"], "required_word_count")
        self.assertEqual(d["spans"][5]["score"], 0.0)

        text = "The and that"
        doc = Document(source="", version="", id="", text=text)
        doc_result = tagger.predict(doc=doc)
        d = doc_result.to_json()
        self.assertEqual(d["spans"][7]["type"], "required_word_count")
        self.assertEqual(d["spans"][7]["score"], 2.0)
