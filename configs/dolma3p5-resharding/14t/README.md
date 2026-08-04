# Dolma 3.5 14T resharding preparation

This directory contains the checked-in settings for translating the Dolma 3.5
14T mix into an auditable resharding proposal. The preparation commands do not
materialize data. The inventory, preflight, and verification scripts
access AWS using only read-only `ListObjectsV2` and `HeadObject` calls.

The checked-in inputs are:

- `inputs/dolma3p5-14t-optimal-mix.yaml`: the authoritative optimal-mix
  definition.
- `inputs/dolma3p5-reference-all-npy-s3-paths.csv`: the reference superset of
  S3 NPY paths, used only while translating YAML paths into inventory work.

The mix YAML is authoritative. The all-NPY CSV is a reference superset used
only to resolve `dolma3p5_pool/...` expressions. Direct `preprocessed/...`
expressions are resolved from S3. The known duplicate second
`the-stack-v2:Tcl/high/quality_p95` expression is removed; no other duplicates
are silently removed.

All volumes are estimated from NPY object size divided by four. The tools never
open production NPY files or count their contents.

The configured 14T value is a nominal target, not an exact-output invariant.
Per-category targets retain the YAML weights, while whole-object allocation
reports the proposed total and residuals without failing merely because the
result is slightly above or below 14T.

## Distributed execution model

Fourteen trillion uint32 values are roughly 56 TB of NPY output before
metadata, source downloads, or working space. The materialization is therefore
never designed as one machine or one command.

The scheduling boundary is an active YAML category (`leaf_id`). Every active
category receives at least one independent execution unit. A category is split
into additional deterministic units whenever its estimated working set would
exceed the explicit per-worker budget supplied to `propose.py`.

For each unit, the size-only working-set estimate is:

```text
unique input NPY bytes
+ unique input metadata bytes
+ planned output NPY bytes
+ estimated output metadata bytes
```

Output NPY bytes are exact relative to the proposed whole-object allocation.
Output metadata bytes are estimated from the inventoried metadata size and
repetition count. The budget is therefore a planning ceiling, not a filesystem
quota; select a value below usable worker storage to leave room for the OS,
tokenizer cache, logs, compression variance, and other runtime overhead.

Execution units have unique manifests, configs, launcher scripts, temporary
directories, random seeds, and S3 destination prefixes. Units do not share
local state and may run in any order. All unit destinations live below one
dataset root, so the finished dataset is the union of the successfully
validated unit prefixes.

## 1. Create a local plan

From the repository root, run:

```bash
python scripts/dolma3p5_resharding/plan.py
```

The checked-in mix, reference-path catalog, settings, and local build path are
the defaults. The build is created at `runs/dolma3p5-resharding/14t/`, which is
ignored by Git. Every later preparation command uses that same default build
path. `--output` and the corresponding `--build` flags remain available for an
intentional alternate build. If `plan.py` uses an alternate `--output`, pass
that same path as `--build` to each later command.

The default remains create-only: if it already exists, the command stops
instead of replacing it. Preserve it as an audit record, or deliberately give
a different `--output` path for a separate preparation run.

Review `01-plan/`, especially `resolution-failures.csv`, `corrections.csv`,
`duplicate-paths.csv`, `catalog-matches.csv`, and `listing-plan.csv`.

## 2. Collect read-only inventory in AWS

The inventory script performs fast bulk prefix listings and concurrent exact
heads only for objects missing from the bulk results:

```bash
python scripts/dolma3p5_resharding/inventory.py \
  --profile YOUR_READ_ONLY_PROFILE
```

When `s5cmd` is installed, the script uses it internally and retains the raw
JSONL in `02-inventory/` for auditing. Otherwise it automatically uses
concurrent boto3 listings. This choice does not change the command or output
schema.

Review `02-inventory/` before continuing. Missing objects, absent metadata
partners, or NPY sizes not divisible by four are blocking failures.

## 3. Create the distributed proposal

The destination, worker-local temporary root, and estimated per-unit working
budget must be intentionally supplied. No machine-capacity default is assumed.
The S3 root must have at least two components below the bucket.

```bash
python scripts/dolma3p5_resharding/propose.py \
  --destination-root s3://YOUR-BUCKET/new-datasets/dolma3p5 \
  --local-temp-root /mnt/raid0/dolma3p5-resharding \
  --max-unit-working-bytes WORKER_PLANNING_BUDGET_BYTES
```

This command performs no materialization. It creates:

- `category-allocation.csv`: target, available, proposed, and residual values
  for every active category.
- `category-execution-summary.csv`: number of execution units and largest
  working-set estimate for every category.
- `config-index.csv` and `execution-units.jsonl`: the complete unit schedule,
  resource estimates, config paths, and destinations.
- `config/` and `manifests/`: one exact resharding config and manifest per unit.
- `launcher-scripts/`: self-contained executable scripts. Each script embeds
  only its reviewed config and manifest, so a launcher can transfer it without
  copying the entire preparation build to the worker.
- `dataset-layout.json` and `dataset-prefixes.txt`: the common dataset root and
  all unit output prefixes.
- `runtime-requirements.json`: the resharding schema workers must provide;
  launcher scripts check it before doing work.
- `plots/`, `plot-data/`, and `report.html`: target/residual plots plus largest
  working sets and the categories split into the most units.
- `LOCAL-UNIT-COMMANDS.txt`: inert commands for debugging an individual unit;
  it must not be treated as a single-machine batch script.
- `DISTRIBUTED-LAUNCH.txt`: an inert example for distributing the launcher
  scripts.

Before continuing, verify that every row in `config-index.csv` is within the
chosen budget, inspect the largest-unit plot, and confirm that large categories
were split as expected. Changing the budget requires a brand-new preparation
build; proposal phases are never replaced in place.

## 4. Validate the proposal

```bash
python scripts/dolma3p5_resharding/validate.py
```

Validation checks that every active category has at least one unit, unit totals
reconstruct the category proposal, every estimated working set is within the
supplied budget, destinations and unit IDs are unique, manifests exist, and
launcher scripts are executable.

## 5. Run the read-only preflight

Run this immediately before materialization. It repeats the approved inventory,
checks size/ETag/last-modified drift, and verifies that every proposed unit
destination is still empty.

```bash
python scripts/dolma3p5_resharding/preflight.py \
  --profile YOUR_READ_ONLY_PROFILE
```

Do not materialize unless `04-preflight/preflight-summary.json` says `passed:
true`. The resharder repeats the destination check when each config starts and
uses no-clobber uploads to protect against a race after preflight.

## 6. Distribute units only after approval

The preparation tools never start machines or execute unit scripts. A launcher
such as [allenai/poormanray](https://github.com/allenai/poormanray) can transfer
and map the self-contained scripts across a worker cluster. Its `map` command
shuffles scripts, distributes them across instances, and runs each instance's
assigned scripts sequentially, which matches the independent bounded-unit
design.

An operator-reviewed launch resembles:

```bash
pmr create \
  --name YOUR_CLUSTER@YOUR_PROJECT \
  --number WORKER_COUNT \
  --instance-type INSTANCE_TYPE \
  --storage-size WORKER_DISK_GB

pmr wait --name YOUR_CLUSTER@YOUR_PROJECT
pmr setup-dolma-python --name YOUR_CLUSTER@YOUR_PROJECT

# Infrastructure-specific step: install the exact reviewed Dolma revision
# containing this preparation workflow on every worker before mapping units.

pmr map \
  --name YOUR_CLUSTER@YOUR_PROJECT \
  --script runs/dolma3p5-resharding/14t/03-proposal/launcher-scripts
```

`pmr setup-dolma-python` installs the published Dolma package and is only a
baseline setup step. It is not sufficient by itself for this workflow. Deploy
the exact reviewed commit or wheel containing the manifest-aware resharder to
every worker. Each launcher script checks the resharding schema before creating
its config or touching its destination and exits immediately if the worker
runtime is incompatible.

Choose worker storage so usable bytes exceed `--max-unit-working-bytes` with
operational headroom. Choose worker count independently of unit count; workers
process their assigned units sequentially and release run-owned temporary data
after each unit. Other launchers can consume `execution-units.jsonl` or the
same launcher-script directory.

Do not use `LOCAL-UNIT-COMMANDS.txt` as a cluster launcher. It exists only to
reproduce one selected unit during debugging.

If a unit fails before writing output, it may be relaunched after confirming
its destination remains empty. If it partially wrote its destination, the
no-overwrite policy intentionally blocks a blind retry; investigate the failed
prefix and prepare a new destination rather than forcing an overwrite.

## 7. Verify outputs using sizes only

After all distributed units finish, enumerate every unit destination and
compare actual NPY bytes divided by four with its proposed unit size:

```bash
python scripts/dolma3p5_resharding/verify.py \
  --profile YOUR_READ_ONLY_PROFILE
```

This produces `05-output-validation/`, including the exact output inventory,
per-unit pairing and size checks, whole-dataset totals, SVG plots, and an HTML
report. The dataset is not complete unless the number of passing destinations
equals the execution-unit count in `dataset-layout.json`. Verification does not
read array contents or write to S3.

## Overwrite protections

- Every preparation phase uses create-only files and directories.
- Existing build or phase directories are refused.
- Generated configs set `allow_existing_destination: false`.
- Every execution unit has a unique destination and an estimated working set
  no larger than the operator-supplied planning budget.
- The resharder checks that an S3 destination prefix is empty before work.
- Uploads use `s5cmd cp --no-clobber`, protecting against races after preflight.
- A bucket root is never accepted as a destination.
- Temporary work occurs in a unique run-owned child directory. Cleanup removes
  only that child, never the configured temporary base.
- Exact manifests use `s5cmd cp --raw --no-clobber`; they do not recursively
  expand a reviewed object into a broader prefix.
- At materialization time, exact manifest objects are headed concurrently and
  compared with approved sizes/ETags before download. Downloaded sizes are
  checked again before merging.

There is deliberately no force/overwrite flag. If an output prefix already
exists, choose a new destination rather than bypassing the guard.
