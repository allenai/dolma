import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load jsonl as pandas dframe `dolma/tmp.jsonl/cc_en_head-0000.json`
# A line is e.g. `{"gopher_span": [], "decontamination_span": [], "hatespeech_span": [], "pii_span": [], "dedupe_paragraphs_span": [[0, 615, 1.0], [615, 1214, 1.0], [1214, 1853, 1.0], [1853, 2417, 1.0], [2417, 2849, 1.0]]}`
df = pd.read_json(
    "cc_en_head_stats10.jsonl", lines=True
)

# """
### Matching based on the entire doc ###
# Where the span is not empty turn it into True, elsewhere into False
# Compute correlations between the attributes to later turn it into a heatmap
corr = df.map(lambda x: bool(x)).corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(36, 24))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(corr, dtype=bool))
heatmap = sns.heatmap(
    corr, 
    mask=mask, 
    vmin=corr.values.min(), 
    vmax=corr.values[~mask].max(), # Max ignoring the ones in corr
    annot=True, 
    cmap='Blues'
)
#heatmap.set_title('Similarity', fontdict={'fontsize':18}, pad=16)

# Save
plt.savefig('attributes_heatmap_docbased_9mdocs.pdf', dpi=450, bbox_inches='tight')
plt.savefig('attributes_heatmap_docbased_9mdocs.png', dpi=450, bbox_inches='tight')

# """

"""
### Matching based on individual spans ###
# Create the corr-like matrix of cols by cols
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

# Turn into percentages
corr = matrix / len(df)
corr *= 100
# Add the column names
corr = pd.DataFrame(corr, columns=columns, index=columns)

# Plot the heatmap
plt.figure(figsize=(36, 24))
mask = np.triu(np.ones_like(corr, dtype=bool))
heatmap = sns.heatmap(
    corr, 
    mask=mask, 
    vmin=corr.values.min(), 
    vmax=corr.values[~mask].max(), # Max ignoring the ones in corr
    annot=True, 
    cmap='Blues'
)

# Save
plt.savefig('attributes_heatmap_spanbased.pdf', dpi=450, bbox_inches='tight')
plt.savefig('attributes_heatmap_spanbased.png', dpi=450, bbox_inches='tight')
"""
