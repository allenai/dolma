# Dolma 3.5 resharding scripts

Code layout. The run procedure is in `configs/dolma3p5-resharding/14t/README.md`.

## Layout

`workflow.py` holds the planning and validation logic. The other Python files are
CLI wrappers around its entry points:

| Script | Entry points | Writes |
| --- | --- | --- |
| `plan.py` | `plan_build`, `collect_inventory`, `propose_configs` | `01-plan/` |
| `validate.py` | `validate_build` | nothing, checks the build |
| `preflight.py` | `preflight_build` | `02-preflight/` |
| `verify.py` | `verify_output` | `03-output-validation/` |
| `materialize.py` | its own, calls `workflow.py` for selection and preflight | provisions workers, dispatches units |

`output_report.py` renders the output-validation report.
`setup_worker_storage.sh` runs on each worker to prepare local NVMe.

## Scope

- `python/dolma/tokenizer/reshard.py` and `document_selection.py` are the library
  and contain no reference to Dolma 3.5 or the 14T target.
- `scripts/resharding/dispatch.py` builds poormanray command lines and is
  mixture-agnostic.
- This directory is specific to the Dolma 3.5 14T campaign. It hard-codes the
  default build path, cluster, and project, and reads the mix, catalog, and
  settings from `configs/dolma3p5-resharding/14t/`.

Target token count, source bucket, worker instance grid, output shard sizing, and
residual bounds come from `settings.yaml`. The build-id prefix and default build
path are in code.

## Build directory

`runs/dolma3p5-resharding/14t` by default, gitignored, marked by a `build.json`
whose id derives from the mix and catalog hashes. Phases are `01-plan/` (stages
`resolution`, `inventory`, `execution`), `02-preflight/`, and
`03-output-validation/`. Tools refuse a directory that is not a recognized build
of this workflow, and refuse a plan whose mix or catalog changed after creation.

## Tests

`tests/python/test_dolma3p5_resharding.py` covers this directory and the
library's resharding safety behavior. `tests/python/test_resharding.py` covers
`scripts/resharding/dispatch.py` and end-to-end resharding. Both import
`scripts.*`, which requires `pythonpath = ["."]` in `pyproject.toml`. Neither
needs network or AWS credentials.
