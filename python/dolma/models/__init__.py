from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional, TypeVar

from ..cli import BaseCli
from .fasttext_model import FastTextSupervisedCli, FastTextUnsupervisedCli, FastTextQuantizeCli

A = TypeVar("A", bound="ArgumentParser")

MODELS = {
    "fasttext": FastTextSupervisedCli,
    "ft-unsupervised": FastTextUnsupervisedCli,
    "ft-quantize": FastTextQuantizeCli,
}


@dataclass
class ModelsConfig:
    """This CLI has no options, the downstream CLIs will have their own options."""

    pass


class ModelsCli(BaseCli):
    CONFIG = ModelsConfig
    DESCRIPTION = "Subcommand for training and evaluating model-based taggers."

    @classmethod
    def make_parser(cls, parser: A) -> A:
        # parser.add_argument("model", choices=MODELS.keys(), help=MODEL_DESCRIPTION)
        # return parser
        subparsers = parser.add_subparsers(
            dest="model",
            title="dolma model",
            description="Command line interface for training and evaluating models for Dolma filtering.",
        )
        subparsers.required = True
        subparsers.choices = MODELS.keys()  # type: ignore
        for command, cli in MODELS.items():
            cli.make_parser(subparsers.add_parser(command, help=cli.DESCRIPTION))

        return parser

    @classmethod
    def run_from_args(cls, args: Namespace, config: Optional[dict] = None):
        # get the cli for the command and run it with the config we just loaded + the args
        command = args.__dict__.pop("model")
        cli = MODELS[command]
        return cli.run_from_args(args=args, config=config)
