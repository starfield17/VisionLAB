# Implementation handoff

## Immediate priority: Auto Label → Train

Follow [Auto Label / Train information](AUTOLABEL_TRAIN_INFORMATION.md) first. Use LocateAnything community Q4_K for candidate production and YOLO26n for training, with the provided low-memory profiles. Deploy work is paused. `model_lab/` supplies offline converters and a guarded subprocess runner; actual model execution and human review remain the next task.

## Current state

The repository contains a working contract validator and tested synchronous deployment core. Read AGENTS.md, contracts/README.md, contractcheck/README.md and deploy/README.md. `python tools/check.py` is the acceptance gate, run inside a conda environment after `python -m pip install -e '.[dev]'`.

Implemented: strict/schema/semantic validation, package preflight/audit, static adapter descriptions, portable distributions, NumPy preprocessing/NMS, Source/Adapter/Sink ports, lifecycle, event/package validation, and JSONL stdout output. No real model, image decoder, training or network integration exists yet. See CONTRACT_COVERAGE.md for verification limits.

## Fixed boundaries

- Model Lab produces versioned Model Packages; Deploy consumes them.
- `contractcheck` has no Deploy/model/NumPy dependencies.
- Deploy Core depends on contractcheck and NumPy, never a concrete Sink or model framework.
- Concrete ports are composed by the caller; static factories are paired with adapter metadata.
- Ontology labels are data. Preserve original-image coordinates and explicit preprocessing.
- Keep contract 1.0.0 unchanged unless a task explicitly authorizes contract evolution.

Dependency changes need matching `tools/check_policy.py` rules and a deliberate negative check. Do not reduce `tools/test_baseline.json`, disable tests, or weaken assertions to make a task pass. The baseline protects the initial tested behavior; extend it when new accepted scenarios become permanent. Ruff enforces zero unused imports/locals.

## Tasks ready for focused implementation

| Task | Files/capability to add | Interface and acceptance |
| --- | --- | --- |
| Traditional model adapter | One new inference adapter module and composition entrypoint | Select runtime/export format against actual deployment needs first; provide AdapterSpec and factory; decode real outputs, map indices, invert preprocessing; real image → event; unknown outputs/error cases fail |
| Image file Source | One Source module and its tests | Decode file, apply EXIF orientation, convert grayscale/RGBA to RGB under a documented alpha policy; emit Frame then EOF; corrupt image raises, never EOF/empty detection |
| Camera Source | One Source module after image Source is proven | Own acquisition resources and timestamps/session IDs; monotonic indices; release on partial open/read/interruption; hardware tests supplement portable port tests |
| Network or Serial Sink | One module per protocol with explicit config | Encode contract event without importing inference; fake transport tests cover framing, partial writes, disconnect and close; document delivery guarantees |
| Model Lab export | Model-producing capability outside Deploy Core | Reviewed dataset → selected traditional detector → evaluation → export with hash-linked records; valid real package, executable model and retained dataset; full audit plus actual runtime smoke test |

After the Auto Label → Train milestone, resume the real model adapter and image Source to get an actual image-to-event slice. Model/runtime choice remains intentionally open pending real deployment requirements; a model family is not an architectural assumption. Never turn the fixture adapter into production support, or invent a successful training result.

## Acceptance for subsequent changes

Run all gates, then demonstrate the task's real scenario. Preserve unrelated tests. API/schema or dependency changes must be called out explicitly. Keep docs portable; no personal paths, credentials or machine-specific environment settings. Add findings to FRICTION.md only when a genuine ambiguity or constraint problem is discovered.
