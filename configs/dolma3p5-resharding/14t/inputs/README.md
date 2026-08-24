# Dolma 3.5 14T plan inputs

A build's id is a hash of the mix and the catalog. Editing either file changes the
id, invalidates plans already built from it, and changes every destination path
derived from it. Record notes here rather than in the YAML.

## `mix-v2.yaml`

`plan.py`'s default. 210 top-level mixes, weights summing to 1.0.

Differences from v1:

- `dolma3_finemath_v3:finemath` and `rpj-proofpile-arxiv:proofpile` read
  decontaminated, ngram-filtered, minhash-deduplicated `dolma2-tokenizer`
  sources. v1 points both at `dolma3-tokenizer` fallbacks; see its "Missing from
  our copy" comments.
- Adds `the-stack-v2:HTML`, weight 0.0137159618, five quality categories.
- 49 top-level weights differ.

## `dolma3p5-14t-optimal-mix.yaml`

Superseded by v2, retained for provenance. 209 top-level mixes. Materialized once
on 2026-08-05 to ~14T tokens; output validation failed on 10 of 681 destinations.

## `dolma3p5-reference-all-npy-s3-paths.csv`

Superset of source NPY paths. Planning resolves mix path patterns against this
catalog instead of listing the bucket, so the catalog is part of the build id.

## Token target

`target_uint32_values` in `../settings.yaml`, 14,000,000,000,000. The mix files
carry relative weights only.
