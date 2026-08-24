# Dolma 3.5 14T materialization

Run from the repository root. Artifacts go to `runs/dolma3p5-resharding/14t/`,
which is gitignored.

`plan.py` defaults to `inputs/mix-v2.yaml`. The token target is
`target_uint32_values` in `settings.yaml`. `inputs/README.md` describes the input
files.

## Requirements

- Python 3.10–3.12
- `s5cmd` on `PATH`
- `uv`, to run `materialize.py`
- `pip install -e .`, for the boto3 and PyYAML imports in `plan.py`,
  `validate.py`, and `verify.py`
- AWS credentials: read on the source and destination buckets for plan, preflight,
  and verify; EC2 create, tag, resume, and pause for `materialize.py --execute`

`materialize.py` runs under `uv` because it needs `poormanray` for the `pmr`
binary, which is not a repo dependency. The other scripts run under `python`.

All scripts share one build directory. Keep `plan.py --output` and `--build` on
the others consistent, or leave both at the default.

## 1. Plan

```bash
python scripts/dolma3p5_resharding/plan.py \
  --mix configs/dolma3p5-resharding/14t/inputs/mix-v2.yaml \
  --profile YOUR_READ_ONLY_PROFILE \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000

python scripts/dolma3p5_resharding/validate.py
open runs/dolma3p5-resharding/14t/01-plan/report.html
```

`--local-temp-root` is worker-local scratch and must be on NVMe.
`--max-unit-working-bytes` caps the estimated local working set of one execution
unit; the planner splits units that would exceed it.

Check before continuing:

- source volume, target volume, and sampling rates at every level
- execution unit working sets fit their instance types
- destination paths, output shard counts, worker counts
- `01-plan/execution/validation-summary.json` reports `passed: true`
- resolution and execution failure CSVs are empty

## 2. Smoke test one category

```bash
uv run scripts/dolma3p5_resharding/materialize.py --list-categories finemath

uv run scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' --parallelism 2 --dry-run

uv run scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' --parallelism 2 \
  --preflight --execute --verbose
```

Preflight must pass, every planned unit must dispatch, workers must stop on
completion, and final verification must pass. Stop if an input drifted or a
destination is occupied.

## 3. Materialize the rest

Exclude what step 2 materialized. Selectors match by exact string, and an unknown
selector is an error. `MIX_NAME` excludes every category of that mix;
`MIX_NAME::CATEGORY_NAME` excludes one. `dolma3_finemath_v3:finemath` has a single
category, so the mix name covers step 2 exactly.

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all --exclude-category 'dolma3_finemath_v3:finemath' \
  --parallelism 128 --dry-run
```

Compare the dry run unit count against the plan, then:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all --exclude-category 'dolma3_finemath_v3:finemath' \
  --parallelism 128 --preflight --execute --verbose
```

`--parallelism` caps active workers. VM creation runs in batches of 5 with a 3
second delay; worker setup runs 32 at a time. Each worker takes another compatible
unit as it finishes one, and stops when its queue is empty. Defaults are cluster
`dolma3p5-14t`, project `oe-other`, region `us-east-1`. `PMR_REGION` overrides the
region. Use `--ssh-key-path` if the cluster key is not poormanray's default.

## 4. Verify

```bash
python scripts/dolma3p5_resharding/verify.py \
  --category 'dolma3_finemath_v3:finemath::default'

python scripts/dolma3p5_resharding/verify.py --all
open runs/dolma3p5-resharding/14t/03-output-validation/report.html
```

Accept only when `output-summary.json` reports `passed: true`, destinations and
shard counts match the plan, the aggregate residual is within bounds, and the
problem and error CSVs are empty.

## Validation gates

- `validate.py`: plan consistency, before any worker launches
- `--preflight`: source drift and destination occupancy for the selected units,
  immediately before provisioning; a failure blocks the launch and is recorded in
  `02-preflight/`
- `materialize.py`: output shard counts, NPY sizes, metadata partners, and
  residuals, after workers stop
- `verify.py`: repeats the output checks and writes `03-output-validation/`

## Safety

Plan, preflight, and verification artifacts are disposable. Source objects and
materialized destinations are not. Materialization refuses occupied destinations
and never overwrites token or metadata files. Investigate a partially written
destination rather than retrying into it.
