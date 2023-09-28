"""

Code to train a Filter.

@kylel, @lucy3

"""
import argparse

from .ft_tagger import BaseFastTextTagger

def main(args):
    tagger = BaseFastTextTagger.train(**vars(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        required=True,
        type=str,
        help="Path to input training dataset created by ft_dataset",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        type=str,
        help="Path to save trained model",
    )
    parser.add_argument(
        "--min-word-count",
        required=False,
        type=int,
        default=1
    )
    parser.add_argument(
        "--word-vectors-dim",
        required=False,
        type=int,
        default=100
    )
    parser.add_argument(
        "--max-word-ngram",
        required=False,
        type=int,
        default=2
    )

    args = parser.parse_args()
    main(args)
