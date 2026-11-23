"""Normal installed Dolma imports; baseline function from pinned source AST."""

import ast
import json
import logging
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from dolma.tokenizer.reshard import TokensMetadataPaths, group_paths_by_max_size

here = Path(__file__).parent
tree = ast.parse((here / "baseline_reshard.py").read_text(encoding="utf-8-sig"))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "group_paths_by_max_size")
ns = dict(TokensMetadataPaths=TokensMetadataPaths, Counter=Counter, logger=logging.getLogger("baseline"))
exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned-baseline", "exec"), ns)
baseline = ns["group_paths_by_max_size"]
with tempfile.TemporaryDirectory() as tmp:
    rng = random.Random(719)
    paths = []
    for i in range(1000):
        p = Path(tmp) / f"{i}.npy"
        p.write_bytes(bytes(rng.randrange(1, 257)))
        paths.append(TokensMetadataPaths(str(p), str(p.with_suffix(".csv.gz"))))
    for label, items, limit in [
        ("one_group", paths, 1000000),
        ("bounded", paths, 4096),
        ("repeated", paths[:200] * 3, 4096),
    ]:
        times = {"baseline": [], "candidate": []}
        for repeat in range(5):
            results = {}
            for version in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
                fn = baseline if version == "baseline" else group_paths_by_max_size
                start = time.perf_counter()
                results[version] = fn(items, limit)
                times[version].append(time.perf_counter() - start)
            assert results["baseline"] == results["candidate"]
        print(
            json.dumps(
                dict(
                    case=label,
                    n=len(items),
                    limit=limit,
                    rounds=5,
                    seconds=times,
                    speedup=statistics.median(times["baseline"]) / statistics.median(times["candidate"]),
                    exact_groups_equal=True,
                )
            ),
            flush=True,
        )

# #313 class semantics: selection views sharing a physical shard have distinct
# logical sizes and identities. Execute its actual dataclass, not a reimplementation.
from dataclasses import dataclass

import numpy as np

pr_tree = ast.parse((here / "pr313_reshard.py").read_text(encoding="utf-8-sig"))
cls = next(n for n in pr_tree.body if isinstance(n, ast.ClassDef) and n.name == "TokensMetadataPaths")
space = dict(dataclass=dataclass, Path=Path, np=np, os=os)
exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned-pr313-class", "exec"), space)
View = space["TokensMetadataPaths"]
views = [View("shared.npy", "shared.csv.gz", f"select-{i}.csv.gz", i + 1) for i in range(30)]
for limit in (0, 1, 4, 32, 100000):
    assert baseline(views * 3, limit) == group_paths_by_max_size(views * 3, limit)
assert views[0].size == 4 and views[-1].size == 120
print(json.dumps({"pr313_selection_cases": 5, "exact_groups_equal": True}))
