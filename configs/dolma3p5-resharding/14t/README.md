# Dolma 3.5 14T resharding runbook

Run every command from the repository root and complete the review gate before
moving to the next step. Preparation artifacts are written to
`runs/dolma3p5-resharding/14t/` by default.

## 1. Build the inventory-backed sampling plan

```bash
python scripts/dolma3p5_resharding/plan.py \
  --profile YOUR_READ_ONLY_PROFILE
```

This resolves the checked-in YAML paths, inventories their source objects with
`s5cmd`, estimates token counts from uint32 file sizes, and builds the
source-to-target sampling report. It makes only read-only metadata requests.

Before continuing:

- Open `02-inventory/report.html`. Review the source families, then click into
  each subcategory and its categories/lower groups. Confirm the source and
  target tokens, sampling ratio, and each level's distribution.
- Inspect `02-inventory/inventory-details.json` for the same hierarchy and exact
  numeric values in machine-readable form.
- In `02-inventory/inventory-summary.json`, confirm the aggregate source,
  target, token delta, sampling ratio, and source-family/subcategory/category/
  lower-group counts.
- Confirm `01-plan/resolution-failures.csv` is empty.
- Confirm `01-plan/corrections.csv` and `01-plan/duplicate-paths.csv` contain
  only changes you explicitly intend.
- In the same summary, confirm all four values are zero:
  `missing_objects`, `direct_resolution_failures`, `invalid_npy_sizes`, and
  `head_errors`.
- Spot-check `02-inventory/required-objects.csv`, including the NPY/metadata
  pairings, sizes, and paths for large or unusual categories.

Do not generate configs with missing objects, missing metadata partners, failed
direct-prefix resolutions, or NPY sizes that are not divisible by four.

## 2. Generate the distributed materialization proposal

This concrete proposal uses a new build-specific prefix below
`s3://ai2-llm/preprocessed/dolma3p5-14t/materialized`, the worker's local NVMe
instance store mounted at `/mnt/dolma`, and a 1.5 TB per-unit working-set
ceiling:

```bash
python scripts/dolma3p5_resharding/propose.py \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000

open runs/dolma3p5-resharding/14t/03-proposal/report.html
python -m json.tool \
  runs/dolma3p5-resharding/14t/03-proposal/proposal-summary.json
```

The destination is not written by this command. Change it before proposal if
that is not the intended materialized dataset root. The build ID is appended to
the destination automatically, and every unit gets a unique prefix below it.

The working-set estimate includes unique input files and planned output files,
including metadata. Set the limit below usable worker storage so the OS,
tokenizer cache, logs, and other runtime files still have headroom. Large
categories are divided into multiple independent units to stay under this
limit.

This command creates configs and manifests but does not materialize data.

Before continuing:

- Open `03-proposal/report.html`. Compare the inventoried source counts with the
  exact proposed counts after whole-object sampling and with the targets. Click
  through source family, subcategory, category, and lower group; check token
  changes, effective repetition factors, per-object repetition ranges, repeated
  and dropped NPY counts, and total object uses.
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

## 3. Validate the complete preparation build

```bash
python scripts/dolma3p5_resharding/validate.py

python -m json.tool \
  runs/dolma3p5-resharding/14t/03-proposal/validation-summary.json
```

Continue only if the printed result contains `"passed": true`. Also confirm
`03-proposal/validation-summary.json` has `passed: true` and
`03-proposal/validation-failures.csv` is empty.

Validation checks category coverage, unit totals, working-set limits, unique
destinations, exact manifests, and executable launchers.

## 4. Run the preflight immediately before materialization

```bash
python scripts/dolma3p5_resharding/preflight.py

python -m json.tool \
  runs/dolma3p5-resharding/14t/04-preflight/preflight-summary.json
```

Preflight repeats the approved source inventory and checks every proposed
destination. It makes read-only AWS requests.

Before continuing:

- Confirm `04-preflight/preflight-summary.json` has `passed: true`, zero
  `drifted_input_objects`, zero `occupied_destinations`, and zero `errors`.
- Confirm every row in `04-preflight/input-drift.csv` is `unchanged`.
- Confirm every row in `04-preflight/destination-status.csv` is `empty`.

Do not launch if a source changed or any destination contains an object.

## 5. Materialize the execution units

Workers need:

- AWS credentials that can read the approved sources and write the new
  destination prefixes.
- `s5cmd` and the exact reviewed Dolma revision containing the manifest-aware
  resharder. A generic published Dolma install may not contain it.
- A writable `--local-temp-root` with enough headroom for the largest unit.

The command below is the single-device baseline: 128 `i4i.2xlarge` workers.
Each worker has one 1.875 TB NVMe instance-store device; the 200 GB EBS root
volume is only for the OS, Python environment, logs, and status files. The
current inventory produces about 680 independently scheduled units with the
1.5 TB ceiling.

Before production, decide whether this baseline or a multi-NVMe i4i topology is
the better execution shape. RAID0 can increase per-worker local throughput even
when one device has enough capacity. Evaluate it together with worker count and
per-host concurrency: `pmr map` runs a worker's assigned scripts sequentially,
so a larger RAID-backed host does not automatically run more units at once. If
concurrency is added, the combined working sets of simultaneous units must stay
below the array's measured usable capacity.

```bash
PMR_REGION=$(aws s3api get-bucket-location \
  --bucket ai2-llm \
  --query LocationConstraint \
  --output text)
case "$PMR_REGION" in
  None|null|"") PMR_REGION=us-east-1 ;;
esac

pmr create \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --number 128 \
  --instance-type i4i.2xlarge \
  --storage-type gp3 \
  --storage-size 200

pmr wait \
  --name dolma3p5-14t \
  --region "$PMR_REGION"
```

Inspect the detected instance-store devices on every new worker before allowing
any format or RAID operation. The setup script identifies devices by the AWS
`EC2 NVMe Instance Storage` model and refuses mounted devices, child mappings,
and existing filesystem or RAID signatures. Like poormanray's `setup-d2tk`, it
supports both a direct mount and RAID0. Capacity is not the only consideration:
RAID0 over multiple local devices may increase per-worker I/O throughput. When
multiple devices are present, the script requires an explicit `single` or
`raid0` choice instead of inferring the layout from capacity. It also avoids
assuming that the root disk is always `nvme0`.

```bash
pmr transfer \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --source scripts/dolma3p5_resharding/setup_worker_storage.sh:/home/ec2-user/setup_worker_storage.sh

pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'bash /home/ec2-user/setup_worker_storage.sh --check --layout single'
```

Review every worker's output. For `i4i.2xlarge`, each should report exactly one
1.875 TB instance-store device and a direct XFS mount plan; a RAID array is not
possible on that shape. Evaluating RAID0 requires a multi-NVMe shape. If the
production cluster is changed to one, inspect both plans after transferring the
setup script and compare local I/O throughput before choosing the layout:

```bash
pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'bash /home/ec2-user/setup_worker_storage.sh --check --layout single'

pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'bash /home/ec2-user/setup_worker_storage.sh --check --layout raid0'
```

The production `--apply` command must name the reviewed layout. The selected
`i4i.2xlarge` plan uses `single`; replace it with `raid0` only if the cluster was
changed to a reviewed multi-device shape:

```bash
pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'bash /home/ec2-user/setup_worker_storage.sh --apply --layout single'

pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'findmnt /mnt/dolma; df -h /mnt/dolma; test -w /mnt/dolma/dolma3p5-resharding'
```

The instance store is ephemeral: stopping, terminating, or losing a worker
discards its local data. That is acceptable here because the approved source
objects and completed destination units are remote, and every unit can be
reconstructed from its reviewed manifest.

Install Dolma and `s5cmd` after storage preparation:

```bash
pmr setup-dolma-python \
  --name dolma3p5-14t \
  --region "$PMR_REGION"
```

`setup-dolma-python` installs the base package and `s5cmd`. Replace its
resharding module with the reviewed manifest-aware module from this checkout,
then verify every worker before dispatch:

```bash
pmr transfer \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --source python/dolma/tokenizer/reshard.py:/home/ec2-user/.venv/lib/python3.12/site-packages/dolma/tokenizer/reshard.py

pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command '$HOME/.venv/bin/python -c "from dolma.tokenizer.reshard import RESHARDING_MANIFEST_SCHEMA_VERSION; assert RESHARDING_MANIFEST_SCHEMA_VERSION == 1; print(\"manifest resharder: ready\")" && s5cmd version'
```

Run preflight again immediately before dispatch, then map the reviewed unit
scripts:

```bash
python scripts/dolma3p5_resharding/preflight.py

pmr map \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --script runs/dolma3p5-resharding/14t/03-proposal/launcher-scripts
```

`pmr map` distributes scripts across workers; each worker processes its
assigned units sequentially and returns after dispatch. Each unit records
`running`, `succeeded`, or `failed EXIT_CODE` in
`~/dolma3p5-resharding-status/`. Check aggregate worker status with:

```bash
pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'echo "$(hostname)"; find "$HOME/dolma3p5-resharding-status" -maxdepth 1 -name "*.status" -type f -exec cat {} \; 2>/dev/null | sort | uniq -c; pgrep -af "dolma.tokenizer.reshard" || true'
```

Do not verify until the total `succeeded` count equals the execution-unit count
in `03-proposal/proposal-summary.json`, with no `running` or `failed` statuses.
Per-unit logs are beside the status files with a `.log` suffix.

Each unit rechecks its source objects and refuses an occupied destination.
Uploads use no-clobber semantics. If a failed unit wrote nothing, it can be
retried after confirming its destination is still empty. Never blindly retry a
partially written destination; investigate it and prepare a new destination.

## 6. Verify the materialized dataset

After every unit completes, run:

```bash
python scripts/dolma3p5_resharding/verify.py

python -m json.tool \
  runs/dolma3p5-resharding/14t/05-output-validation/output-summary.json

open runs/dolma3p5-resharding/14t/05-output-validation/report.html
```

Verification lists output objects and estimates token counts from uint32 file
sizes for comparison with the proposal. It does not read array contents or
modify source data.

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
