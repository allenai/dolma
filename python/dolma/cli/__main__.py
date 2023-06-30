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
    'ft-train': None,
}


def main():
    parser = ArgumentParser(
        prog='dolma',
        usage="domla [command] [options]",
        description="Command line interface for the DOLMa dataset processing toolkit"
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration optional file",
        type=str,
        default=None,
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    subparsers.choices = AVAILABLE_COMMANDS.keys()  # pyright: ignore

    for command, cli in AVAILABLE_COMMANDS.items():
        if cli is not None:
            cli.make_parser(subparsers.add_parser(command))

    args = parser.parse_args()
    breakpoint()
    AVAILABLE_COMMANDS[args.command].run_from_args(args)
