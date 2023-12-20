import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLNAME_TO_LABEL = {
    "gopher_spans": "Gopher Rules",
    "decontamination_spans": "Decontamination",
    "hatespeech_spans": "Hate Speech",
    "pii_spans": "PII",
    "dedupe_paragraphs_spans": "Deduplication",
}


if os.path.exists("corr.csv"):
    corr = pd.read_csv("corr.csv", index_col=0)
else:
    # A line is e.g.
    # {"gopher_span": [], "decontamination_span": [], "hatespeech_span": [], "pii_span": [], "dedupe_paragraphs_span": [[0, 615, 1.0], [615, 1214, 1.0], [1214, 1853, 1.0], [1853, 2417, 1.0], [2417, 2849, 1.0]]}
    df = pd.read_json(
        # "/home/niklas/dolma/tmp.jsonl/cc_en_head-0000.json", lines=True
        "cc_en_head_stats10.jsonl",
        lines=True,
    )
    ### Matching based on the entire doc ###
    # Where the span is not empty turn it into True, elsewhere into False
    # Compute correlations between the attributes to later turn it into a heatmap
    corr = df.map(lambda x: bool(x)).corr(method="pearson")

    ### Matching based on spans ###
    """
    matrix = np.zeros((len(df.columns), len(df.columns)))
    columns = df.columns
    for _, row in df.iterrows():
        # Iterate over the columns
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                # If the columns are the same, skip
                if col1 == col2: continue
                # Increment if the spans overlap
                # e.g. [0, 615, 1.0] & [614, 1214, 1.0] -> 1
                # while [0, 615, 1.0] & [700, 1214, 1.0] -> 0
                matrix[i, j] += float(
                    any(
                        [span1[0] <= span2[0] and span1[1] >= span2[0] for span2 in row[col2]]
                        for span1 in row[col1]
                    )
                )

    corr = matrix / len(df)
    corr *= 100
    # Add the column names
    corr = pd.DataFrame(corr, columns=columns, index=columns)
    """

# Plot the heatmap
plt.figure(figsize=(36, 24))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(corr, dtype=bool))
heatmap = sns.heatmap(
    corr.rename(columns=COLNAME_TO_LABEL, index=COLNAME_TO_LABEL),
    mask=mask,
    vmin=corr.values.min(),
    vmax=corr.values[~mask].max(),  # Max ignoring the ones in corr
    annot=True,
    cmap="Blues",
    linewidths=0.5,
    annot_kws={"fontsize": 32},
    cbar=False,  # No legend
)

heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=32)  # , fontweight="bold")
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=32)  # , fontweight="bold")

corr.to_csv("corr.csv")
plt.savefig("attributes_heatmap_docbased_9mdocs.pdf", dpi=450, bbox_inches="tight")
plt.savefig("attributes_heatmap_docbased_9mdocs.png", dpi=450, bbox_inches="tight")
