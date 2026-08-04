# Dolma 3.5 14T resharding runbook

Run every command from the repository root and complete the review gate before
moving to the next step. Preparation artifacts are written to
`runs/dolma3p5-resharding/14t/` by default.

## 1. Build the path-resolution plan

```bash
python scripts/dolma3p5_resharding/plan.py
```

This reads the checked-in mix inputs and settings. It makes no AWS requests and
creates `runs/dolma3p5-resharding/14t/01-plan/`.

Before continuing:

- Open `01-plan/report.html`. Click each source bar and confirm its category
  targets, lower groups (such as vigintiles), and resolved YAML paths.
- Confirm `01-plan/resolution-failures.csv` is empty.
- Confirm `01-plan/corrections.csv` and `01-plan/duplicate-paths.csv` contain
  only changes you explicitly intend.
- Inspect `01-plan/catalog-matches.csv` and `01-plan/direct-s3-patterns.csv` for
  unexpected path translations.

Do not inventory until every YAML path has the intended S3 resolution.

## 2. Inventory the source objects

Run with credentials that can list and head the source objects:

```bash
python scripts/dolma3p5_resharding/inventory.py \
  --profile YOUR_READ_ONLY_PROFILE
```

The collector uses `s5cmd` for bulk listings when available and falls back to
concurrent boto3 requests. It only reads S3 metadata.

Before continuing:

- Open `02-inventory/report.html`. Click each source bar and confirm the
  original-to-target token counts and sampling ratios for every category and
  lower group.
- In `02-inventory/inventory-summary.json`, confirm all four values are zero:
  `missing_objects`, `direct_resolution_failures`, `invalid_npy_sizes`, and
  `head_errors`.
- Spot-check `02-inventory/required-objects.csv`, including the NPY/metadata
  pairings, sizes, and paths for large or unusual categories.

Do not generate configs with missing objects, missing metadata partners, failed
direct-prefix resolutions, or NPY sizes that are not divisible by four.

## 3. Generate the distributed materialization proposal

Choose a new S3 destination, a real temporary-filesystem path on the workers,
and a per-unit working-set limit:

```bash
python scripts/dolma3p5_resharding/propose.py \
  --destination-root s3://YOUR-BUCKET/datasets/dolma3p5-14t \
  --local-temp-root /WORKER/TEMP/dolma3p5-resharding \
  --max-unit-working-bytes WORKER_BUDGET_BYTES
```

The working-set estimate includes unique input files and planned output files,
including metadata. Set the limit below usable worker storage so the OS,
tokenizer cache, logs, and other runtime files still have headroom. Large
categories are divided into multiple independent units to stay under this
limit.

This command creates configs and manifests but does not materialize data.

Before continuing:

- Open `03-proposal/report.html`. The source counts are the inventoried S3 NPY
  bytes divided by four. Compare them with the exact proposed counts after
  whole-object sampling and with the targets. Click each source and check every
  category and lower group for its token change, effective repetition factor,
  per-object repetition range, repeated and dropped NPY counts, and total
  object uses.
- In `03-proposal/proposal-summary.json`, confirm the source, proposed, and
  target totals; token change from the source; destination; unit count; and
  largest working set are acceptable.
- Inspect `03-proposal/plot-data/proposed-sampling-by-category.csv` and
  `proposed-sampling-by-lower-group.csv` when exact numeric review is easier
  than the HTML. Inspect `03-proposal/category-allocation.csv` for active
  category residuals and repetition counts.
- Inspect `03-proposal/config-index.csv`. Confirm every unit is within the
  working-set budget, large categories were split sensibly, and every S3
  destination is correct and unique.
- Confirm the configured worker temporary path exists on the intended instance
  type and has more usable space than the largest estimated unit.

The materialization inputs are the exact files under
`03-proposal/manifests/`; the runnable units are the executable files under
`03-proposal/launcher-scripts/`.

## 4. Validate the complete preparation build

```bash
python scripts/dolma3p5_resharding/validate.py
```

Continue only if the printed result contains `"passed": true`. Also confirm
`03-proposal/validation-summary.json` has `passed: true` and
`03-proposal/validation-failures.csv` is empty.

Validation checks category coverage, unit totals, working-set limits, unique
destinations, exact manifests, and executable launchers.

## 5. Run the preflight immediately before materialization

```bash
python scripts/dolma3p5_resharding/preflight.py \
  --profile YOUR_READ_ONLY_PROFILE
```

Preflight repeats the approved source inventory and checks every proposed
destination. It makes read-only AWS requests.

Before continuing:

- Confirm `04-preflight/preflight-summary.json` has `passed: true`, zero
  `drifted_input_objects`, zero `occupied_destinations`, and zero `errors`.
- Confirm every row in `04-preflight/input-drift.csv` is `unchanged`.
- Confirm every row in `04-preflight/destination-status.csv` is `empty`.

Do not launch if a source changed or any destination contains an object.

## 6. Materialize the execution units

Workers need:

- AWS credentials that can read the approved sources and write the new
  destination prefixes.
- `s5cmd` and the exact reviewed Dolma revision containing the manifest-aware
  resharder. A generic published Dolma install may not contain it.
- A writable `--local-temp-root` with enough headroom for the largest unit.

For example, create and prepare a poormanray cluster:

```bash
pmr create \
  --name YOUR_CLUSTER@YOUR_PROJECT \
  --number WORKER_COUNT \
  --instance-type INSTANCE_TYPE \
  --storage-size WORKER_DISK_GB \
  --region AWS_REGION

pmr wait --name YOUR_CLUSTER@YOUR_PROJECT
pmr setup-dolma-python --name YOUR_CLUSTER@YOUR_PROJECT
```

Install the exact reviewed Dolma revision and `s5cmd` on every worker, then map
the reviewed unit scripts:

```bash
pmr map \
  --name YOUR_CLUSTER@YOUR_PROJECT \
  --script runs/dolma3p5-resharding/14t/03-proposal/launcher-scripts
```

`pmr map` distributes scripts across workers; each worker processes its
assigned units sequentially. Confirm every execution unit completes before
verification.

Each unit rechecks its source objects and refuses an occupied destination.
Uploads use no-clobber semantics. If a failed unit wrote nothing, it can be
retried after confirming its destination is still empty. Never blindly retry a
partially written destination; investigate it and prepare a new destination.

## 7. Verify the materialized dataset

After every unit completes, run:

```bash
python scripts/dolma3p5_resharding/verify.py \
  --profile YOUR_READ_ONLY_PROFILE
```

Verification lists output objects and compares NPY bytes divided by four with
the proposal. It does not read array contents or modify S3.

Accept the dataset only when:

- `05-output-validation/output-summary.json` has `passed: true`, zero errors,
  zero failed destinations, and equal expected and checked destination counts.
- `actual_uint32_values` equals `predicted_uint32_values`. It does not need to
  equal exactly 14T.
- Every row in `05-output-validation/output-validation.csv` is `passed`.
- `05-output-validation/output-problems.csv` and `output-errors.csv` are empty.
- The plots and totals in `05-output-validation/report.html` match the reviewed
  proposal.

## Rerunning preparation

Local artifacts created by this workflow are replaceable. Rerunning a phase
also removes later local phases that would otherwise be stale:

- `plan.py` replaces the complete local preparation build.
- `inventory.py` replaces inventory, proposal, preflight, and verification.
- `propose.py` replaces proposal, preflight, and verification.
- `preflight.py` replaces preflight and verification.
- `verify.py` replaces verification only.

Replacement is allowed only when the directory contains a valid preparation
`build.json` and no unknown top-level files. To preserve an earlier build, use
a different local path:

```bash
python scripts/dolma3p5_resharding/plan.py \
  --output runs/dolma3p5-resharding/14t-NEW-LABEL
```

Pass that same path with `--build` to each later preparation command.

This does not apply to source shards or materialized token destinations. Source
objects are read-only, and materialization still refuses an existing
destination and uploads with no-clobber semantics.
