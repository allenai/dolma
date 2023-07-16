"""

Unit tests for core/data_types.py

@kylel

"""

from unittest import TestCase

from dolma.core.data_types import DocResult, Document, InputSpec, Span, TextSlice


class TestDocument(TestCase):
    def test_document_to_from_json(self):
        doc = Document(source="source", version="version", id="id", text="text")
        doc_json = doc.to_json()
        doc_json2 = {
            "source": "source",
            "version": "version",
            "id": "id",
            "text": "text",
        }
        self.assertEqual(doc_json, doc_json2)
        doc2 = Document.from_json(doc_json2)
        self.assertEqual(doc_json, doc2.to_json())

    def test_document_to_from_spec(self):
        doc = Document(source="source", version="version", id="id", text="text")
        spec = doc.to_spec()
        spec2 = InputSpec(source="source", version="version", id="id", text="text")
        self.assertEqual(spec, spec2)
        doc2 = Document.from_spec(spec2)
        self.assertEqual(spec, doc2.to_spec())


class TestSpan(TestCase):
    def test_span_to_from_json(self):
        span = Span(start=0, end=1, type="type", score=1.0)
        span_json = span.to_json()
        span_json2 = {"start": 0, "end": 1, "type": "type", "score": 1.0}
        self.assertEqual(span_json, span_json2)
        span2 = Span.from_json(span_json2)
        self.assertEqual(span_json, span2.to_json())

    # TODO: add tests for to/from Spec
    def test_span_to_from_spec(self):
        span = Span(start=0, end=1, type="type", score=1.0)
        with self.assertRaises(AssertionError):
            span.to_spec()


class TestDocResult(TestCase):
    def test_doc_result_to_from_json(self):
        doc = Document(source="source", version="version", id="id", text="text")
        spans = [
            Span(start=0, end=2, type="xxx", score=1.0),
            Span(start=2, end=4, type="yyy", score=0.5),
        ]
        doc_result = DocResult(doc=doc, spans=spans)

        # to_json() doesnt return Document by default
        # also, it returns this extra field called `"mention"`
        doc_result_json = doc_result.to_json()
        doc_result_json2 = {
            "spans": [
                {"start": 0, "end": 2, "type": "xxx", "score": 1.0, "mention": "te"},
                {"start": 2, "end": 4, "type": "yyy", "score": 0.5, "mention": "xt"},
            ]
        }
        self.assertEqual(doc_result_json, doc_result_json2)

        # from_json() requires also providing the Document
        with self.assertRaises(KeyError):
            DocResult.from_json(doc_result_json2)
        doc_result_json3 = {
            "doc": {
                "source": "source",
                "version": "version",
                "id": "id",
                "text": "text",
            },
            "spans": [
                {"start": 0, "end": 2, "type": "xxx", "score": 1.0, "mention": "te"},
                {"start": 2, "end": 4, "type": "yyy", "score": 0.5, "mention": "xt"},
            ],
        }
        doc_result3 = DocResult.from_json(doc_result_json3)
        self.assertEqual(doc_result_json, doc_result3.to_json())


class TestTextSlice(TestCase):
    def test_text_slice_text(self):
        text = "This is a test"
        slice = TextSlice(doc=text, start=0, end=4)
        self.assertEqual(slice.text, "This")
