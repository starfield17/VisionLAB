# Implementation handoff

## Immediate priority: Post-training → Model Package → real Deploy slice

The Auto Label → Train milestone is complete ([record](AUTOLABEL_TRAIN_INFORMATION.md)), and the Model Package boundary has been closed end to end: a real checkpoint was evaluated, exported, packaged, fully audited and executed by the real deployment core. The current round is [Public dataset → Model Package](PUBLIC_DATASET_ROUND.md): ingest an already-annotated public dataset, gate it with preflight, train on real hardware, and repeat the same producer/consumer chain on a model that has measurable skill.

## Current state

The repository contains a working contract validator, a tested synchronous deployment core, an offline Model Lab producer and a real (non-fixture) deployment slice. Read AGENTS.md, contracts/README.md, contractcheck/README.md, model_lab/AGENTS.md and deploy/README.md. `python tools/check.py` is the acceptance gate, run inside a conda environment after `python -m pip install -e '.[dev]'`.

Implemented: strict/schema/semantic validation, package preflight/audit, static adapter descriptions, portable distributions, NumPy preprocessing/NMS, Source/Adapter/Sink ports, lifecycle, event/package validation, JSONL stdout output; Model Lab acquisition/conversion/guard/training, public COCO-format ingestion with duplicate grouping and deterministic splits, dataset preflight, guarded evaluation and ONNX export with an executable probe, and Model Package assembly that refuses to publish unless preflight and the full local provenance audit pass; a real image-file Source and an ONNX Runtime detection adapter composed through a static registry plus `python -m deploy detect`. Camera sources, transports, Serial sinks and quantization do not exist yet. See CONTRACT_COVERAGE.md for verification limits.

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
| Camera Source | One Source module after image Source is proven | Own acquisition resources and timestamps/session IDs; monotonic indices; release on partial open/read/interruption; hardware tests supplement portable port tests |
| Network or Serial Sink | One module per protocol with explicit config | Encode contract event without importing inference; fake transport tests cover framing, partial writes, disconnect and close; document delivery guarantees |
| Wider input policies | Preprocessing contract vNext | Letterbox or other resize policies require an explicit versioned contract change; the exported models used today are trained with framework letterbox and are deployed through the contract's documented stretch profile, with the measured difference recorded in FRICTION.md |
| Quantization / other export formats | Adapter modules beside the ONNX one | One AdapterSpec and factory per runtime; strict input/output validation, no shape guessing, and a real package smoke test per format |
| Repository split | Release process, not code | Split only after a Model Package has been produced by Model Lab and consumed by Deploy in separate runs of this repository |

Model/runtime choice remains intentionally open pending real deployment requirements; a model family is not an architectural assumption. Never turn the fixture adapter into production support, or invent a successful training result.

## Acceptance for subsequent changes

Run all gates, then demonstrate the task's real scenario. Preserve unrelated tests. API/schema or dependency changes must be called out explicitly. Keep docs portable; no personal paths, credentials or machine-specific environment settings. Add findings to FRICTION.md only when a genuine ambiguity or constraint problem is discovered.
