import multiprocessing
import sys
from argparse import ArgumentParser, Namespace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import smart_open
from rich.console import Console
from rich.table import Table
from yaml import safe_load

from ..core.paths import exists
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


def check_if_version_requested(options: Namespace, *_):
    """Check if the user has requested the version of the package; if so, print and exit"""

    if not getattr(options, "dolma_version", False):
        return

    # user has requested the version; print it and exit
    module_name = __name__.split(".")[0]
    print(version(module_name))
    sys.exit(0)


def check_if_list_commands_requested(options: Namespace, *_):
    """Check if the user has requested the list of available commands; if so, print and exit"""

    if not getattr(options, "dolma_commands", False):
        return

    # for easy printing, create a table
    table = Table(title="Dolma commands", style="bold")
    table.add_column("name", justify="left", style="cyan")
    table.add_column("description", justify="left", style="magenta")

    # add the commands to the table with their descriptions
    for command, cli in AVAILABLE_COMMANDS.items():
        table.add_row(command, cli.DESCRIPTION)

    # print the table and exit
    Console().print(table)
    sys.exit(0)


def read_config(path: Union[None, str]) -> Dict[str, Any]:
    """Read a configuration file if it exists"""
    if path is None:
        return {}

    if not exists(path):
        raise FileNotFoundError(f"Config file {path} does not exist")

    with smart_open.open(path, mode="rt") as f:
        return dict(safe_load(f))


def main(argv: Optional[List[str]] = None):
    """Main entry point for the CLI"""

    try:
        # attempting to set start method to spawn in case it is not set
        multiprocessing.set_start_method("spawn")
    except RuntimeError as ex:
        # method already set, check if it is set to spawn
        if multiprocessing.get_start_method() != "spawn":
            raise RuntimeError("Multiprocessing start method must be set to spawn") from ex

    parser = ArgumentParser(
        prog="dolma",
        usage="dolma {global options} [command] {command options}",
        description="Command line interface for the Dolma processing toolkit",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration optional file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--dolma-version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "-l",
        "--dolma-commands",
        action="store_true",
        help="Print list of available commands and exit",
    )

    # check if user has requested the version
    check_if_version_requested(*parser.parse_known_args(argv))

    # check if user has requested the list of available commands
    check_if_list_commands_requested(*parser.parse_known_args(argv))

    # Continue by adding subparsers and parsing the arguments
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.choices = AVAILABLE_COMMANDS.keys()  # type: ignore
    for command, cli in AVAILABLE_COMMANDS.items():
        cli.make_parser(subparsers.add_parser(command, help=cli.DESCRIPTION))

    # parse the arguments
    args = parser.parse_args(argv)

    # first, get the command and config path to run
    command = args.__dict__.pop("command")
    config_path = args.__dict__.pop("config", None) or None

    # remove the other optional arguments from the top level parser
    args.__dict__.pop("dolma_version", None)
    args.__dict__.pop("dolma_commands", None)

    # read the config file if one was provided
    config = read_config(config_path)

    # get the cli for the command and run it with the config we just loaded + the args
    cli = AVAILABLE_COMMANDS[command]
    return cli.run_from_args(args=args, config=config)
