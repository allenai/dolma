import random
from typing import TYPE_CHECKING
from dolma.core.paths import glob_path

import necessary

with necessary.necessary("click") as CLICK_AVAILABLE:
    if CLICK_AVAILABLE or TYPE_CHECKING:
        import click


@click.command()
@click.option("--prefix")
@click.option("--ratio", type=float)
@click.option("--seed", type=int, default=0)
def main(prefix: str, ratio: float, seed: int):
    assert 0 < ratio < 1
    random.seed(seed)

    for path in glob_path(prefix):
        if random.random() < ratio:
            print(path)


if __name__ == "__main__":
    main()
