from collections import defaultdict
from contextlib import ExitStack
import csv
import datetime
import json
from typing import Dict, Optional
import numpy as np
import smart_open
from dolma.core.paths import glob_path, split_path
from tqdm import tqdm


CLOUDFLARE_RANKS = "s3://dolma-artifacts/cloudflare_ranking_buckets"
CLOUDFLARE_MAP = {
    "ranking_top_1000.csv.gz": 1000,
    "ranking_top_2000.csv.gz": 2000,
    "ranking_top_5000.csv.gz": 5000,
    "ranking_top_10000.csv.gz": 10000,
    "ranking_top_20000.csv.gz": 20000,
    "ranking_top_50000.csv.gz": 50000,
    "ranking_top_100000.csv.gz": 100000,
    "ranking_top_200000.csv.gz": 200000,
    "ranking_top_500000.csv.gz": 500000,
    "ranking_top_1000000.csv.gz": 1000000,
}
DESTINATION = "s3://dolma-artifacts/cloudflare_ranking_buckets_merged/merged_{last_date}.jsonl.gz"
FLOOR = max(CLOUDFLARE_MAP.values()) * 10


def score_fn(rank_counts: Dict[int, int], total: Optional[int]) -> float:
    total = max(total or 0, seen := sum(rank_counts.values()))
    score = sum(rank * count / total for rank, count in rank_counts.items())
    if seen < total:
        score += FLOOR * (total - seen) / total
    return score


def main():
    url_ranks: Dict[str, Dict[int, int]] = {}
    last_date = 0

    with ExitStack() as stack:
        all_week_paths = list(glob_path(CLOUDFLARE_RANKS))
        files_pbar = stack.enter_context(
            tqdm(desc="Cloudflare rank files", unit=" files", unit_scale=True, total=len(all_week_paths) * len(CLOUDFLARE_MAP))
        )
        domains_pbar = stack.enter_context(tqdm(desc="Aggregated domains", unit=" domains", unit_scale=True))
        for week_path in all_week_paths:
            _, (*_, date) = split_path(week_path)
            last_date = max(last_date, int(date.rsplit('-', 1)[1]))
            seen_week_ranks = set()

            # go from smallest to largest rank
            week_paths = sorted(
                list(glob_path(f"{week_path}/*.csv.gz")),
                key=lambda x: int(x.rstrip(".csv.gz").rsplit("_", 1)[1])
            )
            for path in week_paths:
                _, (*_, fn) = split_path(path)
                rank = CLOUDFLARE_MAP.get(fn, None)
                if rank is None:
                    continue

                with smart_open.open(path, "rt") as f:
                    headers = next(f).strip().split(",")
                    reader = csv.DictReader(f, fieldnames=headers)
                    for row in reader:

                        # in the same week, we only want to count each domain once
                        if row["domain"] in seen_week_ranks:
                            continue
                        seen_week_ranks.add(row["domain"])

                        # if we haven't seen this domain before, add it to the dict
                        if row["domain"] not in url_ranks:
                            url_ranks[row["domain"]] = defaultdict(int)
                            domains_pbar.update(1)

                        # increment the count for this rank
                        url_ranks[row["domain"]][rank] += 1
                files_pbar.update(1)

    with smart_open.open(DESTINATION.format(last_date=last_date), "wt") as f:
        sorted_urls = sorted([
            (score_fn(rank_counts=ranks, total=len(all_week_paths)), url, ranks)
            for url, ranks in url_ranks.items()
        ])
        for score, url, ranks in sorted_urls:
            sorted_ranks = dict(sorted((round(np.log10(k), 3), v) for k, v in ranks.items()))
            row = {"url": url, "ranks": sorted_ranks, "score": round(np.log10(score), 3)}
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
