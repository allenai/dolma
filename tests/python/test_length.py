import unittest

from dolma.core.data_types import Document
from dolma.taggers.length import CharLengthStripWsV1, CharLengthV1


class TestLengthTaggers(unittest.TestCase):
    def setUp(self) -> None:
        self.doc_with_ws = Document(text="\n\tThis is a test.  ", id="0", source=__file__)
        self.doc_no_ws = Document(text="This is a test.", id="1", source=__file__)
        self.doc_has_inter_ws = Document(text="This is\n a test.", id="2", source=__file__)
        self.empty_doc = Document(text="", id="3", source=__file__)
        self.empty_doc_with_ws = Document(text="\n\t  \n", id="4", source=__file__)

        self.char_length_v1 = CharLengthV1()
        self.char_length_strip_ws_v1 = CharLengthStripWsV1()

    def test_doc_with_no_ws(self):
        result = self.char_length_v1.predict(self.doc_no_ws)
        self.assertEqual(result.spans[0].score, 15)

        result = self.char_length_strip_ws_v1.predict(self.doc_no_ws)
        self.assertEqual(result.spans[0].score, 15)

    def test_doc_with_ws(self):
        result = self.char_length_v1.predict(self.doc_with_ws)
        self.assertEqual(result.spans[0].score, 19)

        result = self.char_length_strip_ws_v1.predict(self.doc_with_ws)
        self.assertEqual(result.spans[0].score, 15)

    def test_doc_with_inter_ws(self):
        result = self.char_length_v1.predict(self.doc_has_inter_ws)
        self.assertEqual(result.spans[0].score, 16)

        result = self.char_length_strip_ws_v1.predict(self.doc_has_inter_ws)
        self.assertEqual(result.spans[0].score, 16)

    def test_empty_doc(self):
        result = self.char_length_v1.predict(self.empty_doc)
        self.assertEqual(result.spans[0].score, 0)

        result = self.char_length_strip_ws_v1.predict(self.empty_doc)
        self.assertEqual(result.spans[0].score, 0)

    def test_empty_doc_with_ws(self):
        result = self.char_length_v1.predict(self.empty_doc_with_ws)
        self.assertEqual(result.spans[0].score, 5)

        result = self.char_length_strip_ws_v1.predict(self.empty_doc_with_ws)
        self.assertEqual(result.spans[0].score, 0)
