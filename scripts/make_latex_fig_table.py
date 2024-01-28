# import argparse
# import bisect
# import fnmatch
# import re
# from collections import defaultdict
# from functools import lru_cache, partial
# from math import ceil, floor, log10
# from pathlib import Path
# from statistics import stdev
# from typing import List, Optional, Sequence, Tuple

from math import ceil
from pathlib import Path
from typing import Dict, List

from necessary import necessary

with necessary(
    modules=["click"],
    message="Please run `pip install click`",
):
    import click


@click.command()
@click.option("-d", "--paper-directory", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("-i", "--input-prefix", type=str, default="experiments", show_default=True)
@click.option("-o", "--output-prefix", type=str, default="appendix/results", show_default=True)
@click.option("--num-columns", type=int, default=3, show_default=True)
def main(
    paper_directory: Path,
    input_prefix: str,
    output_prefix: str,
    num_columns: int,
):
    (destination := paper_directory / output_prefix).mkdir(parents=True, exist_ok=True)

    group_by_path: Dict[str, List[str]] = {}
    for fn in (source := paper_directory / input_prefix).glob("**/*.pdf"):
        group = str(fn.relative_to(source).parent)
        group_by_path.setdefault(group, []).append(str(input_prefix / fn.relative_to(source)))

    grouped_sections = {}

    for figure_group, paths in group_by_path.items():
        figure_group, subgroup = figure_group.rsplit("/", 1)

        group_num_columns = min(num_columns, len(paths))

        grouped_paths = [paths[i : i + group_num_columns] for i in range(0, len(paths), group_num_columns)]

        current_section_components = []

        for i, paths in enumerate(sorted(grouped_paths)):
            width = round(1 / group_num_columns - (0.01 * (group_num_columns - 1)), 2)

            subfigures = "\n\t\\quad\n".join(
                [
                    f"\t\\begin{{subfigure}}{{{width}\\textwidth}}\n"
                    f"\t\t\\includegraphics[width=\\linewidth]{{{path}}}\n"
                    f"\t\t\\label{{fig:{path.replace('/', ':').rsplit(':', -1)[0]}}}\n"
                    f"\t\\end{{subfigure}}"
                    for path in paths
                ]
            )

            figure = (
                f"\\begin{{figure}}[h!]\n"
                f"\t\\centering\n"
                f"{subfigures}\n"
                f"\t\\caption{{...}}\n"
                f"\\end{{figure}}"
            )
            current_section_components.append(figure)

        all_section_figures = "\n\n".join(current_section_components)
        current_section = (
            "\\subsection{...}\n"
            f"\\label{{sec:{figure_group.replace('/', ':')}:{subgroup}}}\n\n"
            f"{all_section_figures}\n"
        )
        grouped_sections.setdefault(figure_group, []).append(current_section)

    for figure_group, sections in sorted(grouped_sections.items()):
        sections = "\n\n".join(sections)
        appendix_name = figure_group.replace("/", "_")
        full_group = f"\\section{{...}}\n" f"\\label{{sec:{figure_group.replace('/', ':')}}}\n\n" f"{sections}\n"
        figure_group_path = destination / (appendix_name + ".tex")
        figure_group_path.write_text(full_group)

        print(f"\\input{{{output_prefix}/{appendix_name}}}")


if __name__ == "__main__":
    main()
