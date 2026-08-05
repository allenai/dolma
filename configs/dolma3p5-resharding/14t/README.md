# Dolma 3.5 14T resharding runbook

Run commands from the repository root. Preparation artifacts default to
`runs/dolma3p5-resharding/14t/`.

## 1. Build the plan

```bash
python scripts/dolma3p5_resharding/plan.py \
  --profile YOUR_READ_ONLY_PROFILE \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000
```

This resolves the mix, inventories the source files, calculates sampling, and
creates the execution manifests, configs, and launchers. It does not write the
destination dataset.

Review `runs/dolma3p5-resharding/14t/01-plan/report.html`:

- In **Source inventory & sampling**, confirm source and target tokens and the
  sampling ratio at the source-family, subcategory, category, and lower-group
  levels.
- In **Materialization execution**, confirm unit working sets fit the selected
  worker storage, inspect the planned i4i fleet, and inspect categories split
  across multiple units. Confirm the aggregate output-shard and file counts are
  reasonable for training. The planner targets 64 GiB output shards, emits one
  shard for small units, and caps each unit at eight shards.
- Confirm the destinations in `01-plan/execution/config-index.csv` are correct
  and unique.
- Confirm `01-plan/resolution/resolution-failures.csv` and
  `01-plan/execution/validation-failures.csv` are empty.
- Confirm `01-plan/execution/validation-summary.json` has `passed: true`.

The materializer consumes the exact manifests and launchers under
`01-plan/execution/`.

The output-shard target and per-unit cap, worker grid, workload thresholds, disk
headroom, and concurrency limits are set in `settings.yaml`. The workload
estimate includes planned output plus both metadata passes for each source using
document selection. Worker vCPU count does not determine the number of final
output files.

## 2. Validate the preparation build

```bash
python scripts/dolma3p5_resharding/validate.py
```

Continue only when the result contains `"passed": true`.

## 3. Preflight the selected work

Preflight must cover the same selection that will be materialized. For one
category:

```bash
python scripts/dolma3p5_resharding/preflight.py \
  --category 'dolma3_finemath_v3:finemath::default'
```

For the complete dataset:

```bash
python scripts/dolma3p5_resharding/preflight.py --all
```

Before launching, confirm:

- `02-preflight/preflight-summary.json` has `passed: true`.
- `drifted_input_objects`, `occupied_destinations`, and `errors` are zero.
- Every selected row in `input-drift.csv` is `unchanged`.
- Every selected row in `destination-status.csv` is `empty`.

Do not launch if a source changed or a destination is occupied.

## 4. Materialize

`materialize.py` manages the worker lifecycle. It batches the selected units by
their planned i4i type, creates or resumes the required poormanray workers,
prepares the planned single-disk or RAID-0 layout, installs the runtime,
dispatches every worker group before waiting, and stops each worker after its
assigned units finish. Units assigned to different i4i types run concurrently.

Find an exact category selector without contacting AWS:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --list-categories finemath
```

Print the complete lifecycle for one category without creating workers:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --parallelism 2 \
  --dry-run
```

Run its matching preflight and launch the category:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --parallelism 2 \
  --preflight \
  --execute
```

For the complete dataset:

```bash
uv run scripts/dolma3p5_resharding/materialize.py \
  --all \
  --parallelism 128 \
  --dry-run

uv run scripts/dolma3p5_resharding/materialize.py \
  --all \
  --parallelism 128 \
  --preflight \
  --execute
```

The important worker options are:

- `--parallelism` is the maximum number of concurrent workers. The actual
  count is the smaller of this value and the selected execution-unit count.
  It must allow at least one worker for every i4i type selected by the plan.
- `--provision-batch-size` limits concurrent VM-create requests; it defaults
  to 5. Workers are launched in detached batches, so earlier batches continue
  booting while later batches and worker types are submitted.
- `--provision-batch-delay-seconds` controls the pause between launch batches;
  it defaults to 3 seconds. Increase it when a provider reports API throttling,
  or reduce it only after checking the applicable project/account quotas.
- `--preflight` reruns the read-only source and destination checks for the exact
  selection immediately before provisioning workers. It requires `--execute`.
- `--cluster` defaults to `dolma3p5-14t` and sets the worker `cluster` tag.
- `--project` defaults to `oe-other` and sets the worker `project` tag.
- `--region` defaults to `us-east-1` and can be overridden directly or with
  `PMR_REGION`.
- `--verbose` streams poormanray output and every new resharding-log line from
  active workers. Source downloads show s5cmd's native progress statistics for
  every copy and its final operation totals. Resharding logs cover
  source validation, document-selection passes, aggregate merge progress,
  input/shard completion milestones, and output upload progress.
- `--completion-poll-seconds` controls the worker-state and log polling
  interval; it defaults to 30 seconds.
- `--profile` selects the AWS profile used for provisioning and worker setup.
- `--instance-type` overrides the per-unit type selected by the plan. When the
  override differs from the plan, also provide `--storage-layout` explicitly.
- `--root-storage-type` and `--root-storage-size` default to a 200 GiB gp3 root
  volume. Materialization data uses local NVMe, not the root volume.
- `--storage-layout auto` uses the plan: one NVMe device selects `single` and
  multiple devices select `raid0`.

At execution time, the materializer refuses a cluster containing active or
transitioning workers. Compatible stopped workers are reused; missing workers
are launched in bounded, detached batches. All planned worker types are
submitted before one combined readiness wait. Every setup and dispatch command
is scoped to the exact selected instance IDs so unrelated stopped workers
cannot be resumed or assigned work.
If setup or dispatch fails, the materializer attempts to pause the workers it
started. The command does not treat poormanray's detached job submission as
completion: it waits for every selected worker to stop, then verifies the NPY
sizes and metadata pairs at every selected destination. Only that verification
produces the final success message.

Each unit refuses an occupied destination and uploads with no-clobber
semantics. Do not retry a partially written destination. Investigate it and
prepare a new destination instead.

## 5. Verify the output

After every selected unit has completed:

```bash
python scripts/dolma3p5_resharding/verify.py \
  --category 'dolma3_finemath_v3:finemath::default'

python scripts/dolma3p5_resharding/verify.py --all

python -m json.tool \
  runs/dolma3p5-resharding/14t/03-output-validation/output-summary.json

open runs/dolma3p5-resharding/14t/03-output-validation/report.html
```

Accept the output only when:

- `output-summary.json` has `passed: true`, zero errors, and zero failed
  destinations.
- Expected and checked destination counts are equal.
- Planned and actual output-shard counts are equal.
- `aggregate_target_residual_within_bound` is `true`.
- Every row in `output-validation.csv` is `passed`.
- `output-problems.csv` and `output-errors.csv` are empty.

## Rebuilding preparation artifacts

The plan, preflight, and verification artifacts created by these tools are
replaceable. Source files and materialized token destinations are not. The
materializer never overwrites an existing destination.

To keep an earlier preparation build, choose another local output and pass it
to every later command with `--build`:

```bash
python scripts/dolma3p5_resharding/plan.py \
  --output runs/dolma3p5-resharding/14t-NEW-LABEL \
  --profile YOUR_READ_ONLY_PROFILE \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000
```
