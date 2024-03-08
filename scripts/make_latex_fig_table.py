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

    # \item \textbf{AI2 Reasoning Challenge}~\citep{arc}: A science question-answering dataset broken into \emph{easy} and \emph{challenge} subsets. Only the easy subset was used in online evaluations. The challenge subset was, however,  included in offline evaluations.
    # %Only the easy subset is included in online evaluations. The challenge subset additionally included in offline evaluations.
    # \item \textbf{BoolQ}~\citep{clark2019boolq}: A reading comprehension dataset consisting of naturally occurring yes/no boolean questions and background contexts.
    # % \item \textbf{Choice Of Plausible Alternatives (COPA)}~\citep{copa}: A commonsense reasoning dataset that involves selecting plausible sentences given input premises.
    # \item \textbf{HellaSwag}~\citep{zellers2019hellaswag}: A multiple-choice question-answering dataset that tests situational understanding and commonsense.
    # \item \textbf{OpenBookQA}~\citep{openbookQA}: A multiple-choice question-answering dataset modeled on open-book science exams.
    # \item \textbf{Physical Interaction: Question Answering (PIQA)}~\citep{piqa}: A multiple-choice question-answering dataset that focuses on physical commonsense and naive physics.
    # \item \textbf{SciQ}~\citep{sciq}: A crowdsourced multiple-choice question-answering dataset consisting of everyday questions about physics, chemistry and biology, among other areas of science.
    # \item \textbf{WinoGrande}~\citep{winogrande}: A dataset of pronoun resolution problems involving various forms of commonsense. Modeled after the Winograd challenge of \cite{wsc}.


metrics_names = {
    "wikitext_103.pdf": "WikiText 103~\\citep{merity2016pointer}",
    "4chan.pdf": "4chan~\\citep{papasavva2020raiders}",
    "c4_100_domains.pdf": "C4 100 dom~\\citep{chronopoulou-etal-2022-efficient}",
    "c4.pdf": "C4~\\citep{raffel2020exploring,dodge-etal-2021-documenting}",
    "dolma_books.pdf": "Dolma Books Subset",
    "dolma_common_crawl.pdf": "Dolma Web Subset",
    "dolma_pes2o_v2.pdf": "Dolma Papers Subset",
    "dolma_reddit.pdf": "Dolma Reddit Subset",
    "dolma_stack_v5.pdf": "Dolma Code Subset",
    "dolma_wiki.pdf": "Dolma Wikipedia Subset",
    "gab.pdf": "Gab~\\citep{zannettou2018gab}",
    "ice.pdf": "ICE~\\citep{greenbaum1991ice}",
    "m2d2_s2orc.pdf": "M2D2~\\citep{reid-etal-2022-m2d2} (S2ORC)",
    "m2d2_wiki.pdf": "M2D2~\\citep{reid-etal-2022-m2d2} (Wiki)",
    "manosphere.pdf": "Manosphere~\\citep{ribeiroevolution2021}",
    "mc4_english.pdf": "mC4~\\citep{mc4} (English)",
    "penn_treebank.pdf": "Penn Tree Bank~\\citep{marcus-etal-1994-penn}",
    "pile.pdf": "Pile~\\citep{Gao2020ThePA} (Val)",
    "twitteraee.pdf": "Twitter AAE~\\citep{blodgett-etal-2016-demographic}",
    "winogrande.pdf": "WinoGrande~\\citep{winogrande}",
    "sciq.pdf": "SciQ~\\citep{sciq}",
    "openbookqa.pdf": "OpenBookQA~\\citep{openbookQA}",
    "hellaswag.pdf": "HellaSwag~\\citep{zellers2019hellaswag}",
    "piqa.pdf": "PIQA~\\citep{piqa}",
    "arc_easy.pdf": "ARC-E~\\citep{arc}",
    "train_cross_entropy.pdf": "Training Cross Entropy",
    "train_ppl.pdf": "Training Perplexity",
}

subsets = {
    "train": "Training Metrics",
    "ppl": "Perplexity on Paloma",
    "downstream": "Downstream Evaluation",
}

abl_names = {
    # "ablations_code_15p_stack_v2_v4_starcoder": ""
    "150b_runs": "Comparing \\dolma With Other Corpora",
    "ablations_cc_dedupe": "Deduping Strategy",
    "ablations_cc_pii_filtering": "Filtering of Personal Identifiable Information",
    # "ablations_cc_quality": ""
    "ablations_cc_quality_only": "Comparing Quality Filters for Web Pipeline",
    "ablations_cc_to_quality_plus_content": "Full Comparison of Web Pipeline",
    "ablations_cc_toxic_filtering": "Toxicity Filtering in Web Pipeline",
    "ablations_code_stack_v2_vs_v4": "Comparing Code Processing Pipeline",
    "ablations_dolma_mix": "Studying \\dolma Mixture",
    "ablations_reddit_selection": "Strategies to Format Conversational Forums Pipeline",
    "ablations_reddit_toxic_filtering": "Evaluating Toxicity Filtering in Conversational Forums Pipeline",
    "long_1b_run": "Training \\OlmoTiny",
}


@click.command()
@click.option("-d", "--paper-directory", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("-i", "--input-prefix", type=str, default="experiments", show_default=True)
@click.option("-o", "--output-prefix", type=str, default="appendix/results", show_default=True)
@click.option("-w", "--num-columns", type=int, default=3, show_default=True)
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

        if figure_group not in abl_names:
            continue

        subsection_name_cands = [s for s in subsets if s in subgroup]
        if subsection_name_cands:
            subsection_name = subsets[subsection_name_cands[0]]
        else:
            continue

        group_num_columns = num_columns  # min(num_columns, len(paths))

        paths = [p for p in paths if p.rsplit("/", 1)[1] in metrics_names]
        grouped_paths = [paths[i : i + group_num_columns] for i in range(0, len(paths), group_num_columns)]

        current_section_components = []

        for i, paths in enumerate(sorted(grouped_paths)):
            width = round(1 / group_num_columns - (0.01 * (group_num_columns - 1)), 2)

            group_subfigures = []
            captions = []
            for path in paths:
                metric_name = metrics_names.get(path.rsplit("/", 1)[1], None)
                if metric_name is None:
                    continue

                captions.append(metric_name.replace("\\\\", "\\"))

                group_subfigures.append(
                    f"\t\\begin{{subfigure}}{{{width}\\textwidth}}\n"
                    f"\t\t\\includegraphics[width=\\linewidth]{{{path}}}\n"
                    # f"\t\t\\caption{{{metric_name}}}\n"
                    # f"\t\t\\label{{fig:{path.replace('/', ':').rsplit(':', -1)[0]}}}\n"
                    f"\t\\end{{subfigure}}"
                )

            subfigures = "\n\t\\quad\n".join(group_subfigures)

            if len(captions) > 1:
                caption_text = ", ".join(captions[:-1]) + " and " + captions[-1]
                if len(captions) > 2:
                    # oxford comma
                    caption_text = caption_text.replace(" and ", ", and ")
            elif len(captions) == 1:
                caption_text = captions[0]
            else:
                breakpoint()

            if "train" in caption_text.lower():
                caption = caption_text
            elif "perplexity" in subsection_name.lower():
                caption = f"Perplexity results on Paloma~\\citep{{paloma}}; subsets {caption_text}"
            else:
                caption = f"Results downstream tasks {caption_text}"

            figure = (
                f"\\begin{{figure}}[h!]\n"
                f"\t\\centering\n"
                f"{subfigures}\n"
                f"\t\\caption{{{caption}}}\n"
                f"\\end{{figure}}"
            )
            current_section_components.append(figure)

        all_section_figures = "\n\n".join(current_section_components) + "\n"
        # grouped_sections.setdefault(figure_group, []).append(all_section_figures)
        current_section = (
            # f"\\subsection{{{subsection_name}}}\n"
            f"\\label{{sec:{figure_group.replace('/', ':')}:{subgroup}}}\n\n"
            f"{all_section_figures}\n"
        )
        grouped_sections.setdefault(figure_group, []).append(current_section)

    for figure_group, sections in sorted(grouped_sections.items()):
        sections = "\n\n".join(sections)
        appendix_name = figure_group.replace("/", "_")
        figure_group_title = abl_names[figure_group]
        full_group = (
            f"\\subsection{{{figure_group_title}}}\n"
            f"\\label{{sec:{figure_group.replace('/', ':')}}}\n\n"
            f"{sections}\n"
            f"\\clearpage\n"
        )
        figure_group_path = destination / (appendix_name + ".tex")
        figure_group_path.write_text(full_group)

        print(f"\\input{{{output_prefix}/{appendix_name}}}")


if __name__ == "__main__":
    main()
