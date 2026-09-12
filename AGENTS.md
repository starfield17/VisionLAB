# AGENTS.md

This document defines the rules that coding agents must follow when collaborating in this repository. The rules can be extended over time; currently there are two hard requirements.

## Rules

1. **Use conda environments by default**
   - All Python-related operations (dependency install, script execution, training, evaluation, export, etc.) must be done inside a conda environment by default.
   - Example: `conda activate <env>` followed by `python` / `pip`, or use `conda run -n <env> python ...` / `conda run -n <env> pip ...` directly.
   - Do not switch to venv / virtualenv / bare system Python; conda takes priority.

2. **Never write local environment information into documents**
   - No documentation that may appear later (README, architecture docs, tutorials, script comments, etc.) may contain anything tied to this machine's environment, including but not limited to:
     - Local absolute paths (e.g., `/Users/...`, `/home/...`)
     - Local conda environment names
     - Local hardware / platform details (GPU model, RK3588, macOS/Linux, etc.)
     - Personal configuration and credentials on this machine
   - Documents must only describe generic, portable operations (e.g., let the reader supply their own `conda activate` env name) or use placeholders.

## Notes

- This file itself is a repo constraint file, not subject to the "documents must not contain local environment info" rule; actual env names can be maintained here.
- If rules change or new ones are added, append them here. Keep it concise and executable.

## Repository map and checks

- Install development tools from the repository root with `python -m pip install -e '.[dev]'` inside conda. Run `python tools/check.py` before delivery.
- `contracts/schemas/`: canonical 1.0.0 schemas, included as installable resources. `contracts/examples/`: illustrative, non-executable assets.
- `contractcheck/`: offline document/package validators and CLI. `deploy/`: numerical core, ports, lifecycle and output adapters.
- Tests live in `contractcheck/tests/`, `deploy/tests/` and `tests/`. Dependency policy and test baseline are enforced by `tools/check_policy.py`; changing a boundary requires updating that explicit policy and its negative tests.
- `docs/IMPLEMENTATION_HANDOFF.md` lists remaining adapter/producer tasks; `docs/CONTRACT_COVERAGE.md` records what verification does and does not establish.

- `model_lab/`: offline dataset ingestion, labeling/training preparation, guarded execution, evaluation, export and Model Package assembly. The Auto Label → Train milestone is recorded in `docs/AUTOLABEL_TRAIN_INFORMATION.md`; the current round is `docs/PUBLIC_DATASET_ROUND.md`.
- `deploy/`: synchronous numerical core and ports plus one real image-file Source and one ONNX Runtime detection adapter. Real packages are run with `python -m deploy detect --package <dir> --image <file>`.
- All downloaded models, datasets, generated overlays, run logs, checkpoints, and other temporary execution artifacts MUST be stored under the repository-root `workdir/` directory. `workdir/` is ignored by Git and must not be used for source code, schemas, or committed documentation.
