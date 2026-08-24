# Dolma 3.5 resharding scripts

Developer orientation. For the operator procedure — how to actually reproduce
the 14T mixture — read `configs/dolma3p5-resharding/14t/README.md`.

## Layout

`workflow.py` holds the whole planning and validation workflow; every other
Python file here is a thin CLI wrapper around one of its entry points:

| Script | Entry points | Writes |
| --- | --- | --- |
| `plan.py` | `plan_build`, `collect_inventory`, `propose_configs` | `01-plan/` |
| `validate.py` | `validate_build` | nothing; checks the build |
| `preflight.py` | `preflight_build` | `02-preflight/` |
| `verify.py` | `verify_output` | `03-output-validation/` |
| `materialize.py` | its own; calls into `workflow.py` for selection and preflight | provisions workers, dispatches units |

`output_report.py` renders the output-validation report. `setup_worker_storage.sh`
is shipped to each worker to prepare its local NVMe before resharding starts.

## What is mixture-specific and what is not

- `python/dolma/tokenizer/reshard.py` and `document_selection.py` are the
  library: mixture-agnostic, no knowledge of Dolma 3.5 or the 14T target.
- `scripts/resharding/` is mixture-agnostic orchestration. `dispatch.py` builds
  poormanray command lines and nothing else.
- This directory is the Dolma 3.5 14T campaign. It hard-codes campaign defaults
  (build path, cluster, project) and reads its mix, catalog, and settings from
  `configs/dolma3p5-resharding/14t/`.

Most campaign parameters — target token count, source bucket, worker instance
grid, output shard sizing, residual bounds — come from `settings.yaml` rather
than code, so another target size is largely a config change. The campaign
identity itself is not: the build-id prefix and the default build path name this
mixture.

## Build directory

All state lives under one build directory, `runs/dolma3p5-resharding/14t` by
default (gitignored), marked by a `build.json` whose id is derived from the mix
and catalog hashes. The phases are `01-plan/` (stages `resolution`, `inventory`,
`execution`), `02-preflight/`, and `03-output-validation/`. Tools refuse to
touch a directory that is not a recognized build of this workflow, and refuse to
run against a plan whose mix or catalog has changed since it was created.

## Tests

`tests/python/test_dolma3p5_resharding.py` covers this directory and the
library's resharding safety behavior; `tests/python/test_resharding.py` covers
`scripts/resharding/dispatch.py` and end-to-end resharding. Both import
`scripts.*`, which works because `pythonpath = ["."]` is set in
`pyproject.toml`. They need no network or AWS credentials.
