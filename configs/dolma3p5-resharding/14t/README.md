# Dolma 3.5 14T resharding runbook

Run every command from the repository root and complete the review gate before
moving to the next step. Preparation artifacts are written to
`runs/dolma3p5-resharding/14t/` by default.

## 1. Build and review the complete plan

```bash
python scripts/dolma3p5_resharding/plan.py \
  --profile YOUR_READ_ONLY_PROFILE \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000
```

This one command resolves the mix, inventories its source objects, calculates
sampling, and creates the execution-unit configs. Inventory access is
read-only, and the destination is not written.

The build ID is appended to the destination root. That dataset root replaces
each source's top-level storage prefix (for example,
`s3://ai2-llm/preprocessed`). The remaining source directory, including lower
groups such as vigintiles, is unchanged. Execution units are partitioned within
those source directories and numbered with eight-digit directories
(`00000000`, `00000001`, ...). Execution-unit IDs use the same globally
incrementing eight-digit format; category names and hashes are not embedded in
IDs.

The output is grouped by purpose:

```text
01-plan/
  report.html   source/sampling and execution views in one tabbed report
  resolution/   normalized mix, path matches, corrections, and failures
  inventory/    source sizes, source-to-target sampling data, and audits
  execution/    worker plan, exact manifests, configs, and launchers
```

Before continuing:

- Open `01-plan/report.html`. In **Source inventory & sampling**, review the
  source families, then click into each subcategory and its categories/lower
  groups. Confirm the source and target tokens, sampling ratio, and each
  level's distribution.
- In `01-plan/inventory/inventory-summary.json`, confirm the aggregate source,
  target, token delta, sampling ratio, and source-family/subcategory/category/
  lower-group counts.
- Confirm `01-plan/resolution/resolution-failures.csv` is empty.
- Confirm `01-plan/resolution/corrections.csv` and
  `01-plan/resolution/duplicate-paths.csv` contain
  only changes you explicitly intend.
- In the same summary, confirm all four values are zero: `missing_objects`,
  `path_resolution_failures`, `invalid_npy_sizes`, and `head_errors`.
- Review the highest ratios in `01-plan/inventory/sampling-rate-audit.csv`
  against the source and target values shown in the report. Sampling ratios are
  consequences of the mix and inventoried source sizes; they are not rejected
  against an arbitrary global ceiling.
- Spot-check `01-plan/inventory/required-objects.csv`, including the NPY/metadata
  pairings, sizes, and paths for large or unusual categories.
- In **Materialization execution**, review the worker-disk distribution,
  categories split across workers, and the concrete execution units.
- Inspect `01-plan/execution/config-index.csv`. Confirm every unit is within
  the working-set budget and every destination is correct and unique.
- Confirm the worker temporary path has more usable space than the largest
  estimated unit.
- Confirm `01-plan/execution/validation-summary.json` has `passed: true` and
  `validation-failures.csv` is empty.

For exact totals:

```bash
python -m json.tool \
  runs/dolma3p5-resharding/14t/01-plan/execution/proposal-summary.json
```

The materialization inputs are the exact files under
`01-plan/execution/manifests/`; the runnable units are under
`01-plan/execution/launcher-scripts/`.

## 2. Validate the complete preparation build

```bash
python scripts/dolma3p5_resharding/validate.py
```

Continue only if the printed result contains `"passed": true`.

## 3. Run the preflight immediately before materialization

```bash
python scripts/dolma3p5_resharding/preflight.py

python -m json.tool \
  runs/dolma3p5-resharding/14t/02-preflight/preflight-summary.json
```

Preflight repeats the approved source inventory and checks every proposed
destination. It makes read-only AWS requests.

Before continuing:

- Confirm `02-preflight/preflight-summary.json` has `passed: true`, zero
  `drifted_input_objects`, zero `occupied_destinations`, and zero `errors`.
- Confirm every row in `02-preflight/input-drift.csv` is `unchanged`.
- Confirm every row in `02-preflight/destination-status.csv` is `empty`.

Do not launch if a source changed or any destination contains an object.

## 4. Materialize the execution units

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
# All preparation and launch commands default to us-east-1. Set this to
# override the region for the cluster and object store.
export PMR_REGION="${PMR_REGION:-us-east-1}"

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
resharding modules with the reviewed manifest-aware modules from this checkout,
then verify every worker before dispatch:

```bash
pmr transfer \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --source python/dolma/tokenizer/reshard.py:/home/ec2-user/.venv/lib/python3.12/site-packages/dolma/tokenizer/reshard.py

pmr transfer \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --source python/dolma/tokenizer/document_selection.py:/home/ec2-user/.venv/lib/python3.12/site-packages/dolma/tokenizer/document_selection.py

pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command '$HOME/.venv/bin/python -c "from dolma.tokenizer.reshard import RESHARDING_MANIFEST_SCHEMA_VERSION; assert RESHARDING_MANIFEST_SCHEMA_VERSION == 2; print(\"manifest resharder: ready\")" && s5cmd version'
```

Find the exact category selector before dispatch. The filter is optional and
does not contact AWS:

```bash
python scripts/dolma3p5_resharding/materialize.py \
  --list-categories finemath
```

Use the printed `MIX_NAME::CATEGORY_NAME` value for one exact YAML category. An
exact mix name without the final `::CATEGORY_NAME` selects all categories
under that mix entry.

Dry-run one category. This stages only its reviewed launchers locally and
prints every selected unit, destination, working-set estimate, and the exact
`pmr map` command. It does not invoke poormanray:

```bash
python scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --cluster dolma3p5-14t \
  --region "$PMR_REGION" \
  --dry-run
```

Run a matching read-only preflight immediately before launching that category:

```bash
python scripts/dolma3p5_resharding/preflight.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --region "$PMR_REGION"
```

Launch only that category by repeating the reviewed command with `--execute`:

```bash
python scripts/dolma3p5_resharding/materialize.py \
  --category 'dolma3_finemath_v3:finemath::default' \
  --cluster dolma3p5-14t \
  --region "$PMR_REGION" \
  --execute
```

For the complete dataset, dry-run and preflight the full selection first:

```bash
python scripts/dolma3p5_resharding/materialize.py \
  --all \
  --cluster dolma3p5-14t \
  --region "$PMR_REGION" \
  --dry-run

python scripts/dolma3p5_resharding/preflight.py \
  --all \
  --region "$PMR_REGION"

python scripts/dolma3p5_resharding/materialize.py \
  --all \
  --cluster dolma3p5-14t \
  --region "$PMR_REGION" \
  --execute
```

`materialize.py` refuses `--execute` unless the most recent preflight covers
the same selection (or all units), every selected destination is empty, and
the selected inputs are unchanged. `pmr map` then distributes only the staged
selection across workers; each worker processes its assigned units
sequentially and returns after dispatch. Each unit records
`running`, `succeeded`, or `failed EXIT_CODE` in
`~/dolma3p5-resharding-status/`.

For partial copies, each worker reads the paired metadata twice, selects a
deterministic hash-ranked set of whole documents, and logs the realized token
count and residual before writing output. Source token volume is still derived
from file size; the planner never tokenizes or scans arrays to count tokens.

Check aggregate worker status with:

```bash
pmr run \
  --name dolma3p5-14t \
  --region "$PMR_REGION" \
  --command 'echo "$(hostname)"; find "$HOME/dolma3p5-resharding-status" -maxdepth 1 -name "*.status" -type f -exec cat {} \; 2>/dev/null | sort | uniq -c; pgrep -af "dolma.tokenizer.reshard" || true'
```

Do not verify until the total `succeeded` count equals the execution-unit count
in `01-plan/execution/proposal-summary.json`, with no `running` or `failed`
statuses. Per-unit logs are beside the status files with a `.log` suffix.

Each unit rechecks its source objects and refuses an occupied destination.
Uploads use no-clobber semantics. If a failed unit wrote nothing, it can be
retried after confirming its destination is still empty. Never blindly retry a
partially written destination; investigate it and prepare a new destination.

## 5. Verify the materialized dataset

After every unit completes, run:

```bash
python scripts/dolma3p5_resharding/verify.py

python -m json.tool \
  runs/dolma3p5-resharding/14t/03-output-validation/output-summary.json

open runs/dolma3p5-resharding/14t/03-output-validation/report.html
```

Verification lists output objects and estimates token counts from uint32 file
sizes for comparison with the proposal. It does not read array contents or
modify source data.

Accept the dataset only when:

- `03-output-validation/output-summary.json` has `passed: true`, zero errors,
  zero failed destinations, and equal expected and checked destination counts.
- `aggregate_target_residual_within_bound` is `true`. Whole-document boundaries
  can make the materialized count differ slightly from the exact proposal; the
  summary records both the realized residual and the allowed bound.
- Every row in `03-output-validation/output-validation.csv` is `passed`.
- `03-output-validation/output-problems.csv` and `output-errors.csv` are empty.
- The plots and totals in `03-output-validation/report.html` match the reviewed
  proposal.

## Rerunning preparation

Local artifacts created by this workflow are replaceable. Rerunning a phase
also removes later local phases that would otherwise be stale:

- `plan.py` replaces the complete plan plus preflight and verification output.
- `preflight.py` replaces preflight and verification.
- `verify.py` replaces verification only.

Replacement is allowed only when the directory contains a valid preparation
`build.json` and no unknown top-level files. To preserve an earlier build, use
a different local path:

```bash
python scripts/dolma3p5_resharding/plan.py \
  --output runs/dolma3p5-resharding/14t-NEW-LABEL \
  --profile YOUR_READ_ONLY_PROFILE \
  --destination-root s3://ai2-llm/preprocessed/dolma3p5-14t/materialized \
  --local-temp-root /mnt/dolma/dolma3p5-resharding \
  --max-unit-working-bytes 1500000000000
```

Pass that same path with `--build` to each later preparation command.

This does not apply to source shards or materialized token destinations. Source
objects are read-only, and materialization still refuses an existing
destination and uploads with no-clobber semantics.
