import argparse

from .om_utils import make_parser, namespace_to_nested_omegaconf, field

__all__ = [
    'BaseCli',
    'make_parser',
    'namespace_to_nested_omegaconf',
    'field',
]


class BaseCli:
    @classmethod
    def make_parser(cls, parser: argparse.ArgumentParser):
        raise NotImplementedError("Abstract method; must be implemented in subclass")

    @classmethod
    def run_from_args(cls, args: argparse.Namespace):
        raise NotImplementedError("Abstract method; must be implemented in subclass")
