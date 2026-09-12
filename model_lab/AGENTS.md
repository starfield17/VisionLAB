# Model Lab preparation

- `acquisition.py` decodes source images into canonical RGB PNGs (EXIF orientation, white composite for alpha, proportional downsize without upscaling) with portable hash/dimension/transform records. It never runs inference.
- `conversion.py` converts community LocateAnything pixel JSON, approves candidate revisions, assembles validated Dataset snapshots and exports approved datasets to YOLO. No training framework imports belong here.
- `ingest.py` ingests already-annotated COCO-format public datasets: adopted labels keep upstream provenance, unrepresentable boxes are dropped with recorded counts, duplicate families share a group and splits are deterministic with per-class train coverage.
- `preflight.py` reports per-split image/object/class distribution; blocking findings stop `train` unless an explicit reason overrides them, and the override is written next to the run.
- `guard.py` runs an explicitly supplied child process with memory/time observation; it does not choose or download models.
- `train.py` builds a guarded YOLO command from caller-supplied relative paths, resolving them to absolute before `project=` so Ultralytics never nests output under `runs/`; it never imports a training framework.
- `runrecords.py` builds portable immutable training/evaluation/export records; `evaluation.py`, `export.py` and `package.py` orchestrate guarded runs and refuse to publish a package unless preflight and the full audit pass.
- `runners/` holds the only modules allowed to import a training or inference framework; they are executed as guarded subprocesses.
- `__main__.py` exposes `export-yolo`, `overlay` and `train` (run via `python -m model_lab train --project <relative> ...`).
- `records.py` validates portable immutable labeling-run audit records (checkpoint revision/hash, prompt, backend, decode mode, command, raw-result/canonical refs); machine-specific paths are rejected.
- `overlay.py` draws review overlays on canonical images; it is a review aid, not contract data.
- `configs/` contains portable training profiles; device and artifact paths are runtime inputs.
- Each command changes the running system state only by running the guard; execute the auto-label sequence in `docs/AUTOLABEL_TRAIN_INFORMATION.md` and the public-dataset sequence in `docs/PUBLIC_DATASET_ROUND.md`. Candidates require actual human review before release/training.
- Run the root `python tools/check.py`. Dependency policy forbids Model Lab imports into Deploy or contractcheck.
