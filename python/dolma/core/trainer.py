"""

Code to train a Filter.

@kylel, @lucy3

"""
import argparse

from .ft_tagger import BaseFastTextTagger

def main(args):
    tagger = BaseFastTextTagger.train(args.train_file, args.save_path)

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

    args = parser.parse_args()
    main(args)
