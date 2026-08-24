# Dolma 3.5 14T plan inputs

This provenance lives here rather than as comments inside the YAML on purpose.
A build's identity is a hash of the mix and the catalog, so editing either file
changes its `build_id`, invalidates any plan already built from it, and changes
every destination path derived from that id. Documenting the inputs in a
separate file keeps the plan reproducible.

## `mix-v2.yaml` — authoritative

The mixture the 14T materialization reproduces, and `plan.py`'s default.
210 top-level mixes, weights summing to 1.0.

Differences from v1:

- `dolma3_finemath_v3:finemath` and `rpj-proofpile-arxiv:proofpile` now read
  decontaminated, ngram-filtered, minhash-deduplicated `dolma2-tokenizer`
  sources. v1 pointed both at `dolma3-tokenizer` fallbacks because the intended
  copies were missing; see the "Missing from our copy" comments in v1.
- Adds `the-stack-v2:HTML` (weight 0.0137159618, five quality categories).
- Rebalances 49 top-level weights.

## `dolma3p5-14t-optimal-mix.yaml` — superseded

Retained for provenance. Do not use for new runs. 209 top-level mixes.

Materialized once, on 2026-08-05, producing ~14T tokens, but its output
validation did not pass: 10 of 681 destinations failed.

## `dolma3p5-reference-all-npy-s3-paths.csv`

The reference superset of source NPY paths. Planning resolves each mix path
pattern against this catalog rather than listing the bucket blindly, so the
catalog is part of the build identity alongside the mix.

## The token target

`target_uint32_values` in `../settings.yaml` (14,000,000,000,000), not in any
mix file. The mixes carry only relative weights.
