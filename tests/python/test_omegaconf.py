from argparse import ArgumentParser
from dataclasses import dataclass
from unittest import TestCase

from omegaconf import MissingMandatoryValue

from dolma.cli import field, make_parser, namespace_to_nested_omegaconf


@dataclass
class _1:
    a: int = field(help="a")
    b: str = field(help="b")


@dataclass
class _2:
    a: _1 = field(help="a")
    c: float = field(help="c", default=1.0)


class TestOmegaconf(TestCase):
    def test_make_parser(self):
        ap = ArgumentParser()
        parser = make_parser(ap, _1)

        args = parser.parse_args(["--a", "1", "--b", "2"])
        opts = namespace_to_nested_omegaconf(args=args, structured=_1)

        self.assertEqual(opts.a, 1)
        self.assertEqual(opts.b, "2")

    def test_nested_parser(self):
        ap = ArgumentParser()
        parser = make_parser(ap, _2)

        args = parser.parse_args(["--a.a", "1", "--a.b", "2", "--c", "3"])
        opts = namespace_to_nested_omegaconf(args=args, structured=_2)

        self.assertEqual(opts.a.a, 1)
        self.assertEqual(opts.a.b, "2")
        self.assertEqual(opts.c, 3.0)

    def test_fail_omegaconf(self):
        ap = ArgumentParser()
        parser = make_parser(ap, _2)

        args = parser.parse_args(["--a.a", "1", "--c", "3"])
        conf = namespace_to_nested_omegaconf(args, _2)

        with self.assertRaises(MissingMandatoryValue):
            conf.a.b
