import unittest

from msgspec.json import Decoder

from dolma.tokenizer.tokenizer import make_retriever_for_field, make_spec_from_fields


class TestNestedStruct(unittest.TestCase):
    def test_simple_struct(self):
        spec = make_spec_from_fields("test", ("a", int), ("b", int))
        unit = spec(a=1, b=2)
        assert unit.a == 1
        assert unit.b == 2

    def test_nested_struct(self):
        spec = make_spec_from_fields("test", ("a.b", int), ("c", str))
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"b": 1}, "c": "test"}')

        assert unit.a.b == 1
        assert unit.c == "test"

    def test_nested_with_shared_prefix(self):
        spec = make_spec_from_fields("test", ("a.b", int), ("a.c", int), ("d", str))
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"b": 1, "c": 2}, "d": "test"}')

        assert unit.a.b == 1
        assert unit.a.c == 2
        assert unit.d == "test"

    def test_nested_struct_with_list(self):
        spec = make_spec_from_fields("test", ("a.b", int), ("c", str), None)
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"b": 1}, "c": "test"}')

        assert unit.a.b == 1
        assert unit.c == "test"

    def test_simple_field_retriever(self):
        spec = make_spec_from_fields("test", ("a", int))
        retriever = make_retriever_for_field("a", int)
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": 1}')
        assert retriever(unit) == 1

    def test_nested_field_retriever(self):
        spec = make_spec_from_fields("test", ("a.b", int))
        retriever = make_retriever_for_field("a.b", int)
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"b": 1}}')
        assert retriever(unit) == 1

    def test_failing_nested_field_retriever(self):
        spec = make_spec_from_fields("test", ("a.c", float))
        retriever = make_retriever_for_field("a.b", float)
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"c": 1.2}}')
        with self.assertRaises(AttributeError):
            retriever(unit)
