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

metrics_names = {
    "wikitext_103.png": "[WikiText 103](https://api.semanticscholar.org/CorpusID:16299141)",
    "4chan.png": "[4chan](https://api.semanticscholar.org/CorpusID:210839359)",
    "c4_100_domains.png": "[C4 100 domains](https://api.semanticscholar.org/CorpusID:245218833)",
    "c4.png": "[C4](https://api.semanticscholar.org/CorpusID:204838007)",
    "dolma_books.png": "Dolma Books Subset",
    "dolma_common_crawl.png": "Dolma Web Subset",
    "dolma_pes2o_v2.png": "Dolma Papers Subset",
    "dolma_reddit.png": "Dolma Reddit Subset",
    "dolma_stack_v5.png": "Dolma Code Subset",
    "dolma_wiki.png": "Dolma Wikipedia Subset",
    "gab.png": "[Gab](https://api.semanticscholar.org/CorpusID:13853370)",
    "ice.png": "[ICE](https://api.semanticscholar.org/CorpusID:143852991)",
    "m2d2_s2orc.png": "[M2D2](https://api.semanticscholar.org/CorpusID:252907924) (S2ORC)",
    "m2d2_wiki.png": "[M2D2](https://api.semanticscholar.org/CorpusID:252907924) (Wiki)",
    "manosphere.png": "[Manosphere](https://api.semanticscholar.org/CorpusID:210839299)",
    "mc4_english.png": "[mC4](https://api.semanticscholar.org/CorpusID:225040574) (English)",
    "penn_treebank.png": "[Penn Treebank](https://api.semanticscholar.org/CorpusID:252796)",
    "pile.png": "[Pile](https://api.semanticscholar.org/CorpusID:230435736) (Validation)",
    "twitteraee.png": "[Twitter AAE](https://api.semanticscholar.org/CorpusID:1066490)",
    "winogrande.png": "[WinoGrande](https://api.semanticscholar.org/CorpusID:198893658)",
    "sciq.png": "[SciQ](https://api.semanticscholar.org/CorpusID:1553193)",
    "openbookqa.png": "[OpenBookQA](https://api.semanticscholar.org/CorpusID:52183757)",
    "hellaswag.png": "[HellaSwag](https://api.semanticscholar.org/CorpusID:159041722)",
    "piqa.png": "[PIQA](https://api.semanticscholar.org/CorpusID:208290939)",
    "arc_easy.png": "[ARC-E](https://api.semanticscholar.org/CorpusID:3922816)",
    "train_ce.png": "Training Cross Entropy",
    "train_perplexity.png": "Training Perplexity",
}

subsets = {
    "train": "Training Metrics",
    "ppl": "Perplexity on Paloma",
    "downstream": "Downstream Evaluation",
}

abl_names = {
    # "ablations_code_15p_stack_v2_v4_starcoder": ""
    "150b_runs": "Comparing Dolma With Other Corpora",
    "ablations_cc_dedupe": "Deduping Strategy",
    "ablations_cc_pii_filtering": "Filtering of Personal Identifiable Information",
    # "ablations_cc_quality": ""
    "ablations_cc_quality_only": "Comparing Quality Filters for Web Pipeline",
    "ablations_cc_to_quality_plus_content": "Full Comparison of Web Pipeline",
    "ablations_cc_toxic_filtering": "Toxicity Filtering in Web Pipeline",
    "ablations_code_stack_v2_vs_v4": "Comparing Code Processing Pipeline",
    "ablations_dolma_mix": "Studying Dolma Mixture",
    "ablations_reddit_selection": "Strategies to Format Conversational Forums Pipeline",
    "ablations_reddit_toxic_filtering": "Evaluating Toxicity Filtering in Conversational Forums Pipeline",
    "long_1b_run": "Training DolmaLM",
}

PALOMA = "[Paloma](https://api.semanticscholar.org/CorpusID:266348815)"


@click.command()
@click.option("-d", "--paper-directory", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("-i", "--input-prefix", type=str, default="experiments", show_default=True)
@click.option("-e", "--extension", type=str, default="png", show_default=True)
def main(
    paper_directory: Path,
    input_prefix: str,
    extension: str,
):
    # (destination := paper_directory / output_prefix).mkdir(parents=True, exist_ok=True)

    group_by_path: Dict[str, Dict[str, List[str]]] = {}
    for fn in (source := paper_directory / input_prefix).glob(f"**/*.{extension}"):
        group = str(fn.relative_to(source).parent)
        group_by_path.setdefault(group.split('/')[0], {}).setdefault(
            fn.parent.name, []
        ).append(str(input_prefix / fn.relative_to(source)))

    print("# Results")

    for group, subgroups in group_by_path.items():
        if group not in abl_names:
            continue

        print(f"## {abl_names[group]}\n\n\n")

        for subgroup, paths in subgroups.items():
            if subgroup not in subsets:
                continue

            print(f"### {subsets[subgroup]}\n\n")

            # if "train" in subgroup:
            #     breakpoint()

            for path in paths:
                metric_name = metrics_names.get(path.rsplit("/", 1)[1], None)

                if metric_name is None:
                    continue

                if "/ppl/" in path:
                    metric_name = f"Perplexity results on {PALOMA}; subset {metric_name}"
                elif "/downstream/" in path:
                    metric_name = f"Results downstream task {metric_name}"

                print(f'#### {metric_name}')
                print(f"![](https://dolma-artifacts.org/{path})\n")

    # for figure_group, paths in group_by_path.items():
    #     figure_group, subgroup = figure_group.rsplit("/", 1)

    #     if figure_group not in abl_names:
    #         continue

    #     print(f"## {abl_names[figure_group]}\n")

    #     subsection_name_cands = [s for s in subsets if s in subgroup]
    #     if subsection_name_cands:
    #         subsection_name = subsets[subsection_name_cands[0]]
    #     else:
    #         continue

    #     paths = [p for p in paths if p.rsplit("/", 1)[1] in metrics_names]

    #     breakpoint()

    #         width = round(1 / group_num_columns - (0.01 * (group_num_columns - 1)), 2)

    #         group_subfigures = []
    #         captions = []
    #         for path in paths:
    #             metric_name = metrics_names.get(path.rsplit("/", 1)[1], None)
    #             if metric_name is None:
    #                 continue

    #             captions.append(metric_name.replace("\\\\", "\\"))

    #             group_subfigures.append(
    #                 f"\t\\begin{{subfigure}}{{{width}\\textwidth}}\n"
    #                 f"\t\t\\includegraphics[width=\\linewidth]{{{path}}}\n"
    #                 # f"\t\t\\caption{{{metric_name}}}\n"
    #                 # f"\t\t\\label{{fig:{path.replace('/', ':').rsplit(':', -1)[0]}}}\n"
    #                 f"\t\\end{{subfigure}}"
    #             )

    #         subfigures = "\n\t\\quad\n".join(group_subfigures)

    #         if len(captions) > 1:
    #             caption_text = ", ".join(captions[:-1]) + " and " + captions[-1]
    #             if len(captions) > 2:
    #                 # oxford comma
    #                 caption_text = caption_text.replace(" and ", ", and ")
    #         elif len(captions) == 1:
    #             caption_text = captions[0]
    #         else:
    #             breakpoint()

    #         if "train" in caption_text.lower():
    #             caption = caption_text
    #         elif "perplexity" in subsection_name.lower():
    #             caption = f"Perplexity results on Paloma~\\citep{{paloma}}; subsets {caption_text}"
    #         else:
    #             caption = f"Results downstream tasks {caption_text}"

    #         figure = (
    #             f"\\begin{{figure}}[h!]\n"
    #             f"\t\\centering\n"
    #             f"{subfigures}\n"
    #             f"\t\\caption{{{caption}}}\n"
    #             f"\\end{{figure}}"
    #         )
    #         current_section_components.append(figure)

    #     all_section_figures = "\n\n".join(current_section_components) + "\n"
    #     # grouped_sections.setdefault(figure_group, []).append(all_section_figures)
    #     current_section = (
    #         # f"\\subsection{{{subsection_name}}}\n"
    #         f"\\label{{sec:{figure_group.replace('/', ':')}:{subgroup}}}\n\n"
    #         f"{all_section_figures}\n"
    #     )
    #     grouped_sections.setdefault(figure_group, []).append(current_section)

    # for figure_group, sections in sorted(grouped_sections.items()):
    #     sections = "\n\n".join(sections)
    #     appendix_name = figure_group.replace("/", "_")
    #     figure_group_title = abl_names[figure_group]
    #     full_group = (
    #         f"\\subsection{{{figure_group_title}}}\n"
    #         f"\\label{{sec:{figure_group.replace('/', ':')}}}\n\n"
    #         f"{sections}\n"
    #         f"\\clearpage\n"
    #     )
    #     figure_group_path = destination / (appendix_name + ".tex")
    #     figure_group_path.write_text(full_group)

    #     print(f"\\input{{{output_prefix}/{appendix_name}}}")


if __name__ == "__main__":
    main()
