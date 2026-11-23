# Reshard grouping microbenchmark

Use an installed source checkout with `dolma[dev]`. Prepare exact reference sources:

```bash
git show 669f534823b08d266a8fff01f8a1c916a5a56576:python/dolma/tokenizer/reshard.py > benchmarks/reshard_grouping/baseline_reshard.py
gh api 'repos/allenai/dolma/contents/python/dolma/tokenizer/reshard.py?ref=d53f103a74d477f33597c34dfef936eed0ed4cf0' -H 'Accept: application/vnd.github.raw+json' > benchmarks/reshard_grouping/pr313_reshard.py
python benchmarks/reshard_grouping/benchmark.py
python -m pytest -q tests/python/test_reshard_grouping.py tests/python/test_resharding.py
```

The candidate is imported normally from installed Dolma; the pinned baseline function and PR313 path dataclass are AST-loaded without executing unrelated imports. Five alternating-order rounds use actual temporary synthetic files, and compare exact groups each round. Token-copy/S3/model training are not measured; the OS cache is warm. The PR313 check covers logical selected-byte size and repeated selection views, not its entire pipeline. Inputs must remain immutable during grouping.
