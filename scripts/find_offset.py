"""
Find source document given `dolma tokens` metadata and offset.

@soldni
"""


import argparse
import json
from bisect import bisect
from typing import Any, Dict, List

import pandas as pd
import smart_open
import tqdm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type=str, required=True)
    ap.add_argument("-o", "--offsets", type=int, nargs="+", required=True)
    ap.add_argument("-d", "--destination", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    opts = parse_args()

    print("Loading metadata...", end=" ", flush=True)
    data = pd.read_csv(opts.file, header=0, names=["start", "end", "sha", "path", "pos"])

    locs: List[int] = []
    for offset in opts.offsets:
        loc = bisect(data["start"].values, offset)
        if data["start"][loc] > offset:
            # in case of not exact match
            loc -= 1

        if data.iloc[loc].start >= offset or data.iloc[loc].end <= offset:
            raise ValueError(f"Offset {offset} not found in {opts.file}.")
        locs.append(loc)

    print(f"offset(s) found at {locs} row(s)", flush=True)

    rows = data.iloc[locs]

    documents: List[Dict[str, Any]] = []
    for i, row in rows.iterrows():
        with smart_open.open(row.path) as f:
            for i, ln in tqdm.tqdm(enumerate(f, start=1), desc="Searching doc...", unit="lines", total=row.pos):
                if i == row.pos:
                    documents.append(json.loads(ln))
                    break
                elif i > row.pos:
                    raise ValueError("location in document not found")

    with smart_open.open(opts.destination, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
