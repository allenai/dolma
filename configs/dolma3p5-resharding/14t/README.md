# Dolma 3.5 14T materialization

Run commands from the repository root. Local artifacts are written to
`runs/dolma3p5-resharding/14t/`.

## 1. Build and review the plan

```bash
python scripts/dolma3p5_resharding/plan.py \
  --profile YOUR_READ_ONLY_PROFILE \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000

python scripts/dolma3p5_resharding/validate.py

open runs/dolma3p5-resharding/14t/01-plan/report.html
```

Before continuing, confirm:

- Source volume, target volume, and sampling rates are correct at every level.
- Execution-unit working sets fit their selected instance types.
- Destination paths, output-shard counts, and worker counts are reasonable.
- `01-plan/execution/validation-summary.json` reports `passed: true`.
- Resolution and execution failure CSVs are empty.

The materializer consumes the configs, manifests, and launchers under
`01-plan/execution/` exactly as planned.

### Validation gates

- `validate.py` checks that the plan is internally consistent before any
  workers are launched.
- `--preflight` reruns source-drift and destination-occupancy checks for the
  exact selected units immediately before provisioning. A failure blocks the
  launch and is recorded under `02-preflight/`.
- After workers stop, `materialize.py` checks output shard counts, NPY sizes,
  metadata partners, and target residuals before reporting success.
- `verify.py` repeats the output checks and writes the reviewable report under
  `03-output-validation/`.

## 2. Run a category smoke test

Find the exact selector:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --list-categories finemath
```

Review the dry run:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --parallelism 2 \
  --dry-run
```

Launch it:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --parallelism 2 \
  --preflight \
  --execute \
  --verbose
```

Confirm that preflight passes, every planned unit is dispatched, resharding
progress is visible, workers stop after completion, and final verification
passes. Do not continue if an input drifted or a destination is occupied.

## 3. Materialize the remaining plan

Exclude each mix already materialized by a smoke test. Exclusions are exact
selectors; they do not remove similarly named categories from other mixes.

Review the remaining dry run:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all \
  --exclude-category 'dolma3_finemath_v3:finemath' \
  --exclude-category 'cc_all_dressed/all_dressed_v5:health' \
  --parallelism 128 \
  --dry-run
```

Launch it:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all \
  --exclude-category 'dolma3_finemath_v3:finemath' \
  --exclude-category 'cc_all_dressed/all_dressed_v5:health' \
  --parallelism 128 \
  --preflight \
  --execute \
  --verbose
```

`--parallelism` caps active workers. VM creation defaults to batches of five
with a three-second delay, and worker setup defaults to 32 concurrent workers.
Each healthy worker starts one unit immediately and receives another compatible
unit whenever it finishes. It remains warm until its queue is empty, then stops.
Defaults are cluster `dolma3p5-14t`, project `oe-other`, and region `us-east-1`.

## 4. Verify outputs

Verify the same selection that was materialized:

```bash
python scripts/dolma3p5_resharding/verify.py \
  --category 'dolma3_finemath_v3:finemath::default'

python scripts/dolma3p5_resharding/verify.py --all

open runs/dolma3p5-resharding/14t/03-output-validation/report.html
```

Accept the output only when `output-summary.json` reports `passed: true`, all
destinations and shard counts match the plan, the aggregate residual is within
its bound, and the problem and error CSVs are empty.

## Safety

Plan, preflight, and verification artifacts are replaceable. Source objects and
materialized token destinations are not. Materialization refuses occupied
destinations and never overwrites token or metadata files. Investigate a
partially written destination; do not retry into it.
