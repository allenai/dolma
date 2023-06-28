from .deduper import DeduperCli
from .mixer import MixerCli
from .tagger import TaggerCli
from argparse import ArgumentParser


AVAILABLE_COMMANDS = {
    'dedupe': DeduperCli,
    'mix': MixerCli,
    'tag': TaggerCli,
    'visualize': None,
    'browse': None,
    'stats': None,
    'fttrain': None,
}


def main():
    parser = ArgumentParser(
        prog='dolma',
        usage="domla [command] [options]",
        description="Command line interface for the DOLMa dataset processing toolkit"
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    subparsers.choices = AVAILABLE_COMMANDS.keys()  # pyright: ignore

    for command, cli in AVAILABLE_COMMANDS.items():
        cli.make_parser(subparsers.add_parser(command))

    args = parser.parse_args()
    AVAILABLE_COMMANDS[args.command].run_from_args(args)
