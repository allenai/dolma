# Dolma 3.5 14T materialization

Run commands from the repository root. Local artifacts are written to
`runs/dolma3p5-resharding/14t/`, which is gitignored.

## Which mix is the 14T mixture

`inputs/mix-v2.yaml` is authoritative and is what `plan.py` uses by default.
`inputs/dolma3p5-14t-optimal-mix.yaml` is the superseded v1, retained for
provenance. `inputs/README.md` records what each contains and how they differ.
The commands below pass `--mix` explicitly so the reproduced mixture stays
unambiguous even if the default changes.

The 14T target itself is `target_uint32_values` in `settings.yaml`, not in
either mix file.

Do not edit a mix or the catalog to add notes. A build's id is a hash of both,
so any edit invalidates plans already built from it and changes every
destination path derived from that id.

## Prerequisites

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/), used to run `materialize.py`.
- `s5cmd` on `PATH`. Planning refuses to start without it.
- `boto3` and `PyYAML` importable by the `python` that runs `plan.py`,
  `validate.py`, and `verify.py`. Both are base dependencies of this
  repository, so `pip install -e .` provides them. These three scripts import
  `workflow.py` at startup, so a missing dependency fails even `--help`.
- AWS credentials. Planning, preflight, and verification need read access to
  the source and destination buckets (`ai2-llm` by default, see
  `settings.yaml`); `materialize.py --execute` additionally needs permission to
  create, tag, resume, and pause EC2 instances.

`materialize.py` carries its own inline dependency block, including
`poormanray`, which supplies the `pmr` binary that provisions workers. That is
why it runs under `uv run` while the other scripts run under `python`:
`poormanray` is not a dependency of this repository. Running `materialize.py`
with plain `python` works only if `pmr` happens to already be installed.

Every script shares the same build directory. `plan.py --output` and
`--build` on the other scripts must agree if you keep more than one mix's
artifacts side by side; otherwise leave them at the default so stale artifacts
from an earlier run are not silently reused.

## 1. Build and review the plan

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

`--local-temp-root` is the worker-local scratch directory used while
materializing one execution unit; it must be on the worker's NVMe storage.
`--max-unit-working-bytes` (1.5 TB above) caps the estimated local working set
of a single execution unit, which bounds how large a unit the planner will
build before splitting it. Both are recorded in the plan and enforced on the
workers.

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

Exclude exactly what step 2 already materialized, and nothing else. Selectors
are matched by exact string equality, so an unknown selector fails loudly
rather than being ignored.

Exclusion granularity matters. `MIX_NAME` excludes **every** category of that
mix, while `MIX_NAME::CATEGORY_NAME` excludes one. If a smoke test materialized
only one category of a multi-category mix, excluding the whole mix name would
silently leave the rest of it unmaterialized. Add one exclusion per selector
you actually smoke-tested:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all \
  --exclude-category 'dolma3_finemath_v3:finemath' \
  --parallelism 128 \
  --dry-run
```

`dolma3_finemath_v3:finemath` has `default` as its only category, so excluding
the mix name here covers exactly the leaf materialized in step 2. Compare the
dry run's unit count against the plan before launching.

Launch it:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all \
  --exclude-category 'dolma3_finemath_v3:finemath' \
  --parallelism 128 \
  --preflight \
  --execute \
  --verbose
```

`--parallelism` caps active workers. VM creation defaults to batches of five
with a three-second delay, and worker setup defaults to 32 concurrent workers.
Each healthy worker starts one unit immediately and receives another compatible
unit whenever it finishes. It remains warm until its queue is empty, then stops.
Defaults are cluster `dolma3p5-14t`, project `oe-other`, and region
`us-east-1`; a `PMR_REGION` environment variable overrides the region default,
so unset it unless you intend to provision elsewhere. Pass `--ssh-key-path` if
poormanray's default key is not the one for this cluster.

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
