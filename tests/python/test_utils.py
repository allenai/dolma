"""

Tests for the utils module.

@kylel

"""


from unittest import TestCase

from dolma.core.data_types import DocResult, Document, Span, TextSlice
from dolma.core.utils import make_variable_name, split_paragraphs, split_sentences


class TestUtils(TestCase):
    def test_make_variable_name(self):
        pass
    
    def test_split_paragraphs(self):
        text = "This is a paragraph.\nThis is another paragraph.\nThis is a third paragraph."
        paragraphs = split_paragraphs(text=text)
        self.assertIsInstance(paragraphs[0], TextSlice)
        self.assertEqual(len(paragraphs), 3)

        self.assertEqual(paragraphs[0].text, "This is a paragraph.")
        self.assertEqual(text[paragraphs[0].start : paragraphs[0].end], "This is a paragraph.")

        self.assertEqual(paragraphs[1].text, "This is another paragraph.")
        self.assertEqual(text[paragraphs[1].start : paragraphs[1].end], "This is another paragraph.")

    def test_split_paragraphs_empty(self):
        text = ""
        paragraphs = split_paragraphs(text=text)

        self.assertEqual(len(paragraphs), 0)

    def test_split_paragraphs_multi_newline(self):
        text = "This is a paragraph.\n\n\nThis is another paragraph.\n\nThis is a third paragraph."
        paragraphs = split_paragraphs(text=text)

        self.assertEqual(len(paragraphs), 3)
        self.assertIsInstance(paragraphs[0], TextSlice)
        self.assertEqual(len(paragraphs), 3)

        self.assertEqual(paragraphs[0].text, "This is a paragraph.")
        self.assertEqual(text[paragraphs[0].start : paragraphs[0].end], "This is a paragraph.")

        self.assertEqual(paragraphs[1].text, "This is another paragraph.")
        self.assertEqual(text[paragraphs[1].start : paragraphs[1].end], "This is another paragraph.")

    def test_split_sentences(self):
        text = "This is a sentence. This is another sentence. This is a third sentence."
        sentences = split_sentences(text=text)

        self.assertIsInstance(sentences[0], TextSlice)
        self.assertEqual(len(sentences), 3)

        self.assertEqual(sentences[0].text, "This is a sentence.")
        self.assertEqual(text[sentences[0].start : sentences[0].end], "This is a sentence.")

        self.assertEqual(sentences[1].text, "This is another sentence.")
        self.assertEqual(text[sentences[1].start : sentences[1].end], "This is another sentence.")

    def test_split_sentences_empty(self):
        text = ""
        sentences = split_sentences(text=text)

        self.assertEqual(len(sentences), 0)

    def test_split_sentences_with_newline(self):
        text = "This is a sentence. \nThis is another sentence.\nThis is a third sentence."
        sentences = split_sentences(text=text)

        self.assertEqual(len(sentences), 3)
        self.assertIsInstance(sentences[0], TextSlice)
        self.assertEqual(len(sentences), 3)

        self.assertEqual(sentences[0].text, "This is a sentence.")
        self.assertEqual(text[sentences[0].start : sentences[0].end], "This is a sentence.")

        self.assertEqual(sentences[1].text, "This is another sentence.")
        self.assertEqual(text[sentences[1].start : sentences[1].end], "This is another sentence.")

    