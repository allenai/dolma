from .deduper import DeduperCli
from .mixer import MixerCli
from .tagger import TaggerCli
from argparse import ArgumentParser


AVAILABLE_COMMANDS = {
    'deduper': DeduperCli,
    'mixer': MixerCli,
    'tagger': TaggerCli,
}


def main():
    parser = ArgumentParser(
        prog='dolma',
        usage="domla [command] [options]",
        description="Command line interface for dolma corpus processing toolkit"
    )
    subparsers = parser.add_subparsers(dest='command')
    breakpoint()
    for command, cli in AVAILABLE_COMMANDS.items():
        cli.make_parser(subparsers.add_parser(command))

    args = parser.parse_args()
    AVAILABLE_COMMANDS[args.command].run_from_args(args)
