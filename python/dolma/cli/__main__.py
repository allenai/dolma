import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from yaml import safe_load

from .analyzer import AnalyzerCli
from .deduper import DeduperCli
from .mixer import MixerCli

# must import these to register the resolvers
from .resolvers import *  # noqa: F401,F403,W0401
from .tagger import ListTaggerCli, TaggerCli
from .tokenizer import TokenizerCli

AVAILABLE_COMMANDS = {
    "dedupe": DeduperCli,
    "mix": MixerCli,
    "tag": TaggerCli,
    "list": ListTaggerCli,
    "stat": AnalyzerCli,
    "tokens": TokenizerCli,
    # following functionality is not yet implemented
    # "train-ft": None,
    # "train-lm": None,
}


def main(argv: Optional[List[str]] = None):
    try:
        # attempting to set start method to spawn in case it is not set
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # method already set, check if it is set to spawn
        if multiprocessing.get_start_method() != "spawn":
            raise RuntimeError("Multiprocessing start method must be set to spawn")

    parser = ArgumentParser(
        prog="dolma",
        usage="dolma [command] [options]",
        description="Command line interface for the DOLMa dataset processing toolkit",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration optional file",
        type=Path,
        default=None,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.choices = AVAILABLE_COMMANDS.keys()  # type: ignore

    for command, cli in AVAILABLE_COMMANDS.items():
        cli.make_parser(subparsers.add_parser(command, help=cli.DESCRIPTION))

    args = parser.parse_args(argv)

    # try parsing the config file
    config: Optional[dict] = None
    if config_path := args.__dict__.pop("config"):
        assert config_path.exists(), f"Config file {config_path} does not exist"
        with open(config_path) as f:
            config = dict(safe_load(f))

    AVAILABLE_COMMANDS[args.__dict__.pop("command")].run_from_args(args=args, config=config)
