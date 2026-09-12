# Auto Label → Review → Dataset → YOLO26n

## Scope and status

**Status: this milestone is complete.** Offline converters, two training configs, a subprocess memory guard, canonical image acquisition, portable labeling-run records and review overlays are implemented, and the integration exercise was executed for real: eight images were labeled, human-reviewed, released as a versioned dataset, exported to YOLO files, smoke-trained and trained with the initial profile. The "Execution status" section at the end of this document is the authoritative record; the sections above it remain the reproducible procedure. Do not confuse synthetic tests with model execution.

The collected data is intentionally tiny: four train images are confirmed negatives and every positive instance sits in validation, so the trained checkpoint demonstrates the route rather than detection quality. Producing a model with measurable skill requires a larger annotated dataset; that follow-on route is described in [Public dataset → Model Package](PUBLIC_DATASET_ROUND.md).

First run uses COCO8's eight images, original four train/four validation split, and a one-class ontology: ID `person`, label `person`. Treat each image as a distinct group unless inspection reveals related scenes. Do not use the distributed labels as this experiment's training labels. Human review must confirm all people and negative images. The pretrained checkpoint has seen COCO; this is an integration exercise, not an unbiased accuracy benchmark. No test split or accuracy target is claimed.

## Verified sources and pinned inputs

Sources checked for this handoff on 2026-09-09:

| Component | Selected input | Reference |
| --- | --- | --- |
| Detector | `yolo26n.pt`, object detection Nano pretrained weights | [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/) |
| Training library | `ultralytics==8.4.145` | [Release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.145), [pinned defaults](https://github.com/ultralytics/ultralytics/blob/v8.4.145/ultralytics/cfg/default.yaml) |
| Grounding model | NVIDIA LocateAnything-3B | [Official card](https://huggingface.co/nvidia/LocateAnything-3B), [official source](https://github.com/NVlabs/Eagle/tree/main/Embodied) |
| Local inference implementation | Community `mudler/locate-anything.cpp`, commit `77376ab332de918220f7a7e391542eefb5407c9f` | [Pinned source](https://github.com/mudler/locate-anything.cpp/tree/77376ab332de918220f7a7e391542eefb5407c9f) |
| Quantized checkpoint | `locate-anything-q4_k.gguf`, approximately 4.7 GB file | [Community weights](https://huggingface.co/mudler/locate-anything.cpp-gguf) |
| Sample data | COCO8 images | [Dataset](https://docs.ultralytics.com/datasets/detect/coco8/), [pinned dataset config](https://github.com/ultralytics/ultralytics/blob/v8.4.145/ultralytics/cfg/datasets/coco8.yaml) |

Record download revision, exact SHA-256 and file size for both checkpoints and the dataset archive before first use. The weight revision is intentionally unresolved until download; never substitute a guessed digest. Preserve image source/attribution records and consult the dataset's original image terms.

The official LocateAnything example uses custom Transformers model code and BF16 accelerator inference; it is not the selected local low-memory backend. The community runtime offers multiple backends, but a build option is not proof that a particular backend works. Probe the selected backend on one image and record results. Community parity/quantization claims are upstream claims, not verification on our images.

LocateAnything weights currently permit academic/non-profit research, not commercial use under the model card's NVIDIA license. Community code licensing does not replace weight licensing. Ultralytics uses AGPL-3.0 or its enterprise terms. Keep links and notices with the experiment.

## Environment and execution order

Use conda for all Python commands. Supply environment names and paths yourself. Separate labeling and training environments if dependency constraints conflict. No model download or training happens when importing `model_lab`.

```sh
conda activate <env>
python -m pip install -e '.[lab,dev]'
# Only in the environment selected for training:
python -m pip install -e '.[train]'
```

Record resolved library versions with the experiment. Do not upgrade the pinned training library to fix a failure without documenting the change and rerunning config/compatibility checks.

1. Download only the small public sample and selected pretrained/quantized artifacts. Use explicit destinations outside version control. Do not convert full-precision LocateAnything weights locally.
2. Clone the community runtime recursively and checkout the pinned commit, including its submodules. Build the CLI with at most two parallel compiler jobs; choose an available backend explicitly. First run its info command and one image under the guard.
3. Decode images, apply EXIF orientation, convert to RGB, and proportionally downsize without upscaling to longest side 640. Save canonical PNGs; record source/canonical hashes, dimensions and transform. Canonical images are the Dataset contract's coordinate frame. If decode fails, stop that item; do not invent a negative annotation.
4. Run one query per image, one process at a time. Prompt exactly: `Locate all the instances that matches the following description: person.` Use `--mode hybrid --threads 2`. The pinned CLI accepts `--model`, `--input`, `--prompt`, `--output`, and optional `--annotated`; it does not expose a generation-token-limit flag. Its engine's default is 256 new tokens. Empty/short output does not establish that all objects were found.
5. Preserve raw JSON, stdout, stderr, guard result and immutable run metadata. Convert to candidates, create overlays for human review, and stop at the review boundary.
6. After explicit human review, build the versioned Dataset manifest and export YOLO files. Validate before training.
7. Ensure all LocateAnything processes have exited, then run training smoke profile including validation. Only after successful smoke run and acceptable memory measurements run the initial profile.

Example guarded labeling invocation (substitute paths and the executable):

```sh
python -m model_lab.guard --output <new-run-directory> --timeout 1800 -- \
  <locate-anything-cli> detect --model <checkpoint.gguf> --input <canonical.png> \
  --prompt 'Locate all the instances that matches the following description: person.' \
  --mode hybrid --threads 2 --output <raw-result.json>
```

The guard has no shell interpolation, retries or model-loading logic. It executes only the supplied command. Never run labeling and training concurrently.

## Candidate conversion and review

`model_lab.conversion.candidate(raw, annotation_id=..., item_id=..., image_hash=..., width=..., height=..., ontology=..., source=..., created_at=...)` returns a contract annotation.

Expected community response: `{"detections":[{"label":"person","box":[x1,y1,x2,y2]}]}`. The pinned parser already converts quantized model coordinates to input-image pixels. Do not divide by 1000 again. Unknown labels, nonfinite/inverted/out-of-bounds boxes are errors; do not silently discard them. No confidence is emitted because this runtime does not supply one. Successful empty output remains an unreviewed candidate.

`source` contains `kind: locate_anything`, actor ID, immutable run ID, and pinned tool version. The associated run record must retain checkpoint revision/hash, prompt, canonical image hash, backend, decode mode, command options and raw-result hash. Keep machine-specific transient command paths out of portable records; save relative artifact references and logical backend identifiers.

Human review covers every object and every image, including omitted objects and negatives. Generate overlays for inspection and editable candidate copies; do not build a dashboard. For each approval create a new annotation ID with `status: approved`, `supersedes` pointing to the original candidate, unchanged original source, and a human reviewer ID/time/reason. Keep both files available to DatasetValidator. Corrections change objects in the new revision, not the immutable candidate. Rejections do not enter the release.

Flash must not fabricate a reviewer, assume silence is approval, or treat a successful model response as human confirmation. If no reviewer is available, deliver candidates and the pending review list, with training explicitly blocked.

## Dataset and YOLO bridge

After constructing the Dataset contract with approved annotation references:

```sh
contractcheck validate <dataset.json>
python -m model_lab export-yolo <dataset.json> <new-yolo-directory>
```

The exporter checks all hashes/annotations/history before publishing a new directory. Both train and validation must be nonempty. It writes images/labels per split, `data.yaml` and `mapping.json`, preserving ontology declaration order as YOLO's contiguous class indices. Opaque IDs are not filenames. Mapping records include dataset version/hash, class IDs and item/annotation identities.

For original pixel box `(x1,y1,x2,y2)` and canonical dimensions `(W,H)`, YOLO labels contain index and `((x1+x2)/(2W), (y1+y2)/(2H), (x2-x1)/W, (y2-y1)/H)`. Empty text is emitted only for an approved negative. `data.yaml` has relative image paths and no machine-specific dataset root. Decode/orientation correctness is the acquisition stage's responsibility; the converter itself does not decode image bytes.

## Training profiles

Configs are JSON-form YAML accepted as mappings: `model_lab/configs/smoke.yaml` and `initial.yaml`. Paths/device are deliberately supplied by the caller.

| Parameter | Smoke | Initial |
| --- | --- | --- |
| model | yolo26n.pt | yolo26n.pt |
| epochs / imgsz / batch | 1 / 320 / 1 | 30 / 416 / 2 |
| workers / cache | 0 / false | 0 / false |
| amp / compile | false / false | false / false |
| optimizer / lr0 / lrf | AdamW / 0.001 / 0.01 | AdamW / 0.001 / 0.01 |
| weight_decay / nbs | 0.0005 / 16 | 0.0005 / 16 |
| warmup_epochs / warmup_bias_lr | 0 / 0 | 3 / 0 |
| mosaic / mixup / cutmix / multi_scale | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| seed / plots / val / patience | 42 / false / true / 10 | 42 / false / true / 10 |

AdamW is a controlled small-data fine-tuning choice, not a claim to reproduce the official MuSGD training recipe. `nbs` influences accumulation/loss scaling; it is not the actual physical batch. Record all effective upstream defaults, including augmentations not overridden here. The tiny smoke run verifies execution, not convergence or accuracy.

Before loading weights, Flash must check the pinned parser:

```python
import json
from ultralytics.cfg import get_cfg
config = json.load(open("model_lab/configs/smoke.yaml", encoding="utf-8"))
get_cfg(overrides=config)  # validate keys and types with the installed pinned release
```

Then use an explicitly selected available device and already verified local checkpoint. Run the CLI under the memory guard; `model=<local-checkpoint>` overrides the config model name and avoids implicit model download. Train validation can use twice the training batch; include that peak in the smoke measurement.

`model_lab train` resolves every caller-supplied relative path to absolute before passing
`project=` to Ultralytics, so output never nests under its default `runs_dir` (and `runs/`
is git-ignored as a defensive fallback). Use a fresh run name per profile:

```sh
python -m model_lab train --project <relative-run-root> --name <unique-run-name> \
  --cfg model_lab/configs/smoke.yaml --model <relative-checkpoint> \
  --data <relative-yolo-data.yaml> --device <device> \
  --guard-output <relative-guard-dir> [--timeout 1800] [--minimum-available-gib 3]
```

For the initial profile use `initial.yaml` and a fresh run name; choose an explicit wall-clock limit suitable for the experiment. Store best/last checkpoints, results CSV, effective args, logs, finite losses, dataset/weight hashes and a training run record. Framework logs can contain runtime paths: retain locally as raw logs and redact/canonicalize before sharing or committing. This stage does not create an export record or deployable package; do not claim full Model Package audit success.

Training uses Ultralytics' own resizing/augmentation. Do not route it through Deploy's stretch preprocessing or describe future exported models as stretch-compatible without verifying their actual training/inference requirements.

## Memory limits and failure policy

The file size of the quantized weights is not peak memory: vision tensors, decode cache, backend allocations and temporary copies add overhead. No safe-memory guarantee is made without measurements.

The guard polls system available memory and process-tree RSS, records peaks/minima and stops the child on available memory below 3 GiB or the wall-clock limit. Polling cannot prevent all short-lived peaks; accelerator allocations are not fully represented by RSS. Keep framework allocator protections enabled. Do not set allocator thresholds to disable limits. The default 30-minute timeout is a smoke/probe bound, not an accuracy-training promise.

Labeling memory failure: exit completely, create smaller canonical images with longest side 448 (new hashes and transform records), and try one image once more. Do not reuse the old annotation coordinate frame. If it still fails, stop and report backend/error/measurements; do not escalate to higher precision or an unverified official local model.

Training memory failure: stop the run, preserve failure records, start a new run at batch 1 and imgsz 320. Do not pretend this is an unchanged resume. If that fails, stop. Backend incompatibility is recorded separately from memory failure; no silent fallback. Always finish the labeling process before starting the training process.

## Flash checklist and acceptance

1. Record pinned tool versions, verified artifact hashes and sample provenance; keep downloads outside Git.
2. Probe backend/one image with guard and inspect raw output format. Success requires actual detections/empty response from the real tool, not a fixture.
3. Produce eight candidates plus overlays; obtain real human review and immutable approved revisions.
4. Validate Dataset and export YOLO files; compare every exported box and label with the reviewed annotations.
5. Validate config through pinned Ultralytics, run one real epoch including validation, and verify nonempty checkpoint plus finite recorded losses. Report metrics as observed, with no threshold or generalization claim.
6. Only then consider the initial profile. Provide run summary, memory measurements, failures and outstanding review items.
7. Run `python tools/check.py` for repository changes. Never skip tests, fabricate runtime success, bypass review or change Deploy to make this route work.

## Execution status (probe through training smoke)

- Community CLI built (pinned commit, Metal backend) and checkpoint loads (`info` passes). Guard probe succeeded under a recorded lower memory floor; see FRICTION.md.
- All eight COCO8 images were canonicalized (longest side 640, no upscaling) and labeled one image per process (`--mode hybrid --threads 2`, pinned prompt). Each run directory keeps `raw.json`, `stdout.log`, `stderr.log`, `result.json` and an immutable `labeling-record.json`.
- Raw results were converted to candidates (pixel boxes in the canonical frame); review overlays were produced and manually reviewed. Empty/short output does not establish that all objects were found; the negative verdicts were confirmed by inspection.
- Human review completed for all eight items: four train images confirmed negative; validation keeps one (036) and two (061) detections; 049 was corrected to two large persons (nested and uncertain small boxes dropped). Approved revisions were written with reviewer ID, time and reason; a versioned Dataset manifest was assembled and validated; YOLO files were exported.
- Smoke training completed: pinned Ultralytics accepted both configs; one real epoch ran with validation; best/last checkpoints are nonempty and recorded losses are finite. Observed metrics (no threshold or generalization claim): P 0.013, R 0.4, mAP50 0.275, mAP50-95 0.203. All training instances sit in validation; the train split is entirely negative, which the framework warns about.
- Initial profile (30 epochs / 416 / batch 2) completed: early-stopped at epoch 11 (patience 10) because the positive instances all sit in validation; best checkpoint nonempty, losses finite. Observed metrics from the final epoch row (no claim): P 0.00455, R 0.2, mAP50 0.0975; best.pt revalidation gives P 0.00683, R 0.4, mAP50 0.212. A schema-valid training run record with dataset copy and checkpoint hashes is stored next to the checkpoints.
- This stage does not create an export record or deployable package; no Model Package audit success is claimed.

Follow-on work is deliberate and lives outside this document: public-dataset ingestion, preflight gating, post-training evaluation/export/package production and the real deployment slice. Those steps are tracked in [Implementation handoff](IMPLEMENTATION_HANDOFF.md) and [Public dataset → Model Package](PUBLIC_DATASET_ROUND.md). Lightweight conversion and memory-guard behavior remain unit-tested independently of external data production.
