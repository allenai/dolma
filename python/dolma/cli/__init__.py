import argparse


class BaseCli:
    @classmethod
    def make_parser(cls, parser: argparse.ArgumentParser):
        raise NotImplementedError("Abstract method; must be implemented in subclass")

    @classmethod
    def run_from_args(cls, args: argparse.Namespace):
        raise NotImplementedError("Abstract method; must be implemented in subclass")
