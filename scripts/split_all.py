from necessary import necessary
import smart_open
from dolma.core.paths import split_path, join_path, exists


with necessary("click"):
    import click


@click.command()
@click.argument("input_path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def main(input_path: str):
    assert exists(input_path), f"Path {input_path} does not exist"
    prot, parts = split_path(input_path)
    all_path = join_path(prot, parts, "all")
    assert exists(all_path), f"Path {all_path} does not exist"
