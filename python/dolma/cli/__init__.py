import argparse
from argparse import Namespace
from typing import Optional

from .om_utils import field, make_parser, namespace_to_nested_omegaconf

__all__ = [
    "BaseCli",
    "make_parser",
    "namespace_to_nested_omegaconf",
    "field",
]


class BaseCli:
    @classmethod
    def make_parser(cls, parser: argparse.ArgumentParser):
        raise NotImplementedError("Abstract method; must be implemented in subclass")

    @classmethod
    def run_from_args(cls, args: Namespace, config: Optional[dict] = None):
        raise NotImplementedError("Abstract method; must be implemented in subclass")
