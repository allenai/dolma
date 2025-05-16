import unittest

from msgspec.json import Decoder
from dolma.tokenizer.tokenizer import make_spec_from_fields



class TestNestedStruct(unittest.TestCase):
    def test_simple_struct(self):
        spec = make_spec_from_fields("test", ("a", int), ("b", int))
        unit = spec(a=1, b=2)   # type: ignore
        assert unit.a == 1
        assert unit.b == 2

    def test_nested_struct(self):
        spec = make_spec_from_fields("test", ("a.b", int), ("c", str))
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"b": 1}, "c": "test"}')

        assert unit.a.b == 1
        assert unit.c == "test"


    def test_nested_struct_with_list(self):
        spec = make_spec_from_fields("test", ("a.b", int), ("c", str), None)
        decoder = Decoder(spec)
        unit = decoder.decode(b'{"a": {"b": 1}, "c": "test"}')

        assert unit.a.b == 1
        assert unit.c == "test"
