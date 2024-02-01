import os

import pandas as pd

PATH = "/data/niklas/dolma/stack"
TABLE = "Language & RPJ % flagged & SC % flagged & RPJ SC Corr."
TABLE += "& Language & RPJ % flagged & SC % flagged & RPJ SC Corr."
TABLE += "\\\\" + "\n"
TABLE_ROWS = []

paths = sorted(os.listdir(PATH))
halfway = len(paths) // 2

for i, lang in enumerate(paths):
    all_data = []

    for dataset in os.listdir(os.path.join(PATH, lang)):
        # Load json.gz
        data = pd.read_json(os.path.join(PATH, lang, dataset), lines=True)
        all_data.append(data)

    # Concatenate all dataframes
    all_data = pd.concat(all_data)
    # Cast to int
    all_data["rpj"] = all_data["rpj"].astype(int)
    all_data["starcoder"] = all_data["starcoder"].astype(int)
    rpj_flagged = round(100 * (all_data["rpj"].sum() / len(all_data)), 1)
    sc_flagged = round(100 * (all_data["starcoder"].sum() / len(all_data)), 1)
    if rpj_flagged == 0 or sc_flagged == 0:
        corr = "N/A"
    else:
        corr = round(all_data["rpj"].corr(all_data["starcoder"]), 3)

    if i >= halfway:
        TABLE_ROWS[i - halfway] += f" & {lang} & {rpj_flagged} & {sc_flagged} & {corr} \\\\"
    else:
        TABLE_ROWS.append(f"{lang} & {rpj_flagged} & {sc_flagged} & {corr}")
        # TABLE_ROWS.append(f"{lang} & {rpj_flagged} & {sc_flagged} & {corr} \\\\")

TABLE += "\n".join(TABLE_ROWS)
print(TABLE)
