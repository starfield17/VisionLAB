# Public dataset → Model Package (current round)

This round replaces the tiny integration dataset with an already-annotated public dataset, trains a real detector, and drives the result through the Model Package boundary into Deploy. It reuses the [contract baseline](../contracts/README.md) unchanged and the producer/consumer commands described in [implementation handoff](IMPLEMENTATION_HANDOFF.md).

## Dataset

TACO (Trash Annotations in Context) is published under CC BY 4.0 as a COCO-format detection dataset. The annotations file used here is byte-identical to the upstream file (`sha256 ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449`); the downloaded archive's `sha256` is recorded next to the extracted copy in the untracked work directory. Most images carry no per-image license field in this distribution, so attribution relies on the dataset-level license and reference.

The ontology is data, not code: `model_lab/configs/taco_ontology.json` maps all 60 published categories onto five material superclasses (`plastic`, `metal`, `glass`, `paper`, `other`) and declares ontology `litter-material@1`. A repository test asserts that every published category is mapped exactly once.

Ingestion (`model_lab ingest-coco`) decodes each image, applies EXIF orientation, converts to RGB and stores a canonical PNG whose longest side is at most 1280 pixels, rescaling every box into that frame. Observations from the run:

| Observation | Value |
| --- | --- |
| Source images | 1500 |
| Released items | 1346 |
| Images skipped: EXIF-oriented frame | 153 |
| Images skipped: exact duplicate bytes | 1 |
| Objects released | 4259 |
| Objects dropped: box outside the canonical frame | 1 |
| Objects smaller than 32 px on the short side | 1648 |
| Groups after near-duplicate grouping | 1217 (20 multi-image families) |
| Train split | 1067 images / 3467 objects |
| Validation split | 279 images / 792 objects |

Per-class validation counts are non-trivial for every class (glass 46, metal 106, other 218, paper 84, plastic 338), so all five classes are measurable. Adopted annotations keep the upstream producer identity and state in their review reason that this repository did not independently re-review them.

Preflight reports no blocking finding for this dataset and one warning: no train image is a confirmed negative, so false-positive behaviour is not measured by the training split.

## Training

Fine-tuning started from `yolo26n.pt` with `model_lab/configs/taco-detection.yaml` (640 px, batch 8, AdamW, mosaic enabled with `close_mosaic`, seed 42, patience 20) under the memory guard on a CUDA host, batch 8, no automatic model download.

| Observation | Value |
| --- | --- |
| Guard status | `success` |
| Wall clock | 1455 s |
| Peak process-tree RSS | 9.6 GB |
| Minimum system available memory | 11.3 GB (floor 3 GB) |
| Epochs recorded | 69 (early stopped) |
| Best epoch | 49 |
| Validation metrics at best epoch (279 images, conf 0.001, IoU 0.7) | P 0.306, R 0.287, mAP50 0.245, mAP50-95 0.175 |
| Per-class mAP50 | plastic 0.44, metal 0.41, paper 0.20, other 0.13, glass 0.04 |

`model_lab record-training` then wrote a schema-valid training record from the run's own `args.yaml`, `results.csv` and weights, naming `best.pt` as the single trained model and linking the dataset manifest by hash.

## Evaluation, export and package

`model_lab evaluate` ran a guarded validation on the declared validation split and wrote `evaluation-taco-litter-v1` with `split`, `metric_definitions`, `thresholds`, `sample_count` and hash links to the dataset manifest and the trained checkpoint.

`model_lab export-onnx` exported with `nms=False` on the pinned training library and immediately probed the produced graph:

| Probed fact | Value |
| --- | --- |
| Input | `images`, float32, `[1, 3, 640, 640]` |
| Output | `output0`, float32, `[1, 300, 6]` |
| Declared decoder | `end2end_detections_v1` (xyxy input pixels, score, class index) |
| ONNX checker | passed |
| Agreement with the framework's own ONNX reader on the same file | 20/20 top detections matched at IoU ≥ 0.5, max box delta 0.0002 px, max score delta 9e-8 |
| Agreement with the PyTorch checkpoint path | different score scale; recorded in FRICTION.md |

The packaged model is the probed graph (`sha256 4deb10d25e760e81f398bd00ca6f9f7201cd0d8b4edbe4368ff324aa1b72c091`). `model_lab package` assembled `taco-litter-yolo26n-onnx-1` with adapter `yolo26-onnx-detections@1`, preprocessing `640×640` NCHW RGB stretch with `scale 1/255`, postprocessing `confidence_threshold 0.25` and embedded NMS (`none`), and published it only after `python -m contractcheck package audit` passed with the dataset store and the training run as retained store roots.

## Deployment smoke

```sh
python -m deploy detect --package <package-dir> --image <validation-image>
```

On a real validation image the command emitted one Detection Event with three detections in original-image coordinates (`metal` at 0.988, 0.965 and 0.344). The same command run on a second machine reproduced identical boxes and scores from the same package bytes. A deliberately corrupt image exited non-zero with no event written, and an image whose scores stay below the packaged threshold produced a valid empty event rather than a failure.

Because the dataset store stays on the training host, the local machine holds the package, records, logs and a few smoke images; the full provenance audit is run where the retained store lives, and the local check is preflight plus package integrity.

## Reproducing the chain

```sh
# ingest, gate and export the labelled data
python -m model_lab ingest-coco --images <coco-images> --annotations <annotations.json> \
  --ontology model_lab/configs/taco_ontology.json --destination <dataset-dir> \
  --dataset-id taco-litter --version 1 --actor-id <upstream-id> --run-id <distribution-id> \
  --tool-version <ingest-version> --acquired-at <rfc3339> --longest-side 1280 --seed 42
python -m model_lab preflight --dataset <dataset-dir>/dataset.json --report <report.json>
python -m model_lab export-yolo <dataset-dir>/dataset.json <yolo-dir>

# train, then produce the package
python -m model_lab train --project <run-root> --name <run-name> \
  --cfg model_lab/configs/taco-detection.yaml --model <checkpoint> --data <yolo-dir>/data.yaml \
  --device <device> --guard-output <guard-dir> --dataset <dataset-dir>/dataset.json --timeout <seconds>
python -m model_lab record-training --run <run-root>/<run-name> --dataset <dataset-dir>/dataset.json \
  --id <record-id> --profile taco-detection --config-file model_lab/configs/taco-detection.yaml \
  --pretrained <checkpoint> --dataset-yolo <yolo-dir> --implementation-version <version>
python -m model_lab evaluate --checkpoint <run-root>/<run-name>/weights/best.pt \
  --data <yolo-dir>/data.yaml --dataset <dataset-dir>/dataset.json --project <eval-root> \
  --name <eval-name> --guard-output <guard-dir> --metrics-out <metrics.json> \
  --record <evaluation-record.json> --id <record-id> --split validation --imgsz 640 --batch 8 \
  --device <device>
python -m model_lab export-onnx --checkpoint <run-root>/<run-name>/weights/best.pt \
  --dataset <dataset-dir>/dataset.json --training-record <run-root>/<run-name>/training-record.json \
  --probe-image <image> --staging <staging-dir> --guard-output <guard-dir> \
  --probe-out <staging-dir>/probe.json --record <export-record.json> --id <record-id> --imgsz 640
python -m model_lab package --dataset <dataset-dir>/dataset.json \
  --training-record <run-root>/<run-name>/training-record.json \
  --evaluation-record <evaluation-record.json> --export-record <export-record.json> \
  --model <staging-dir>/<checkpoint-stem>.onnx --checkpoint <run-root>/<run-name>/weights/best.pt \
  --metrics <metrics.json> --probe <staging-dir>/probe.json --out <package-dir> \
  --package-id <package-id> --adapter-spec deploy/adapters/specs/yolo26_onnx.adapter.json \
  --imgsz 640 --confidence-threshold 0.25 \
  --store-root <dataset-dir> --store-root <run-root>/<run-name>

# run it
python -m deploy detect --package <package-dir> --image <image>
```

## What this round establishes

- A real, already-annotated public dataset becomes a versioned Dataset snapshot with recorded curation, grouping and split policy, and a preflight gate that would have blocked the earlier all-negative training split.
- A trained checkpoint is evaluated, exported and packaged with hash-linked lineage that passes the full provenance audit.
- The packaged bytes are executed by Deploy through the contract's own preprocessing, producing contract-valid Detection Events in original-image coordinates on two machines.

It does not establish detection quality claims, camera or streaming behaviour, other export formats, quantization, or letterbox support: framework-trained detectors currently run through the contract's stretch profile, and the measured difference is recorded in FRICTION.md as the trigger for a versioned preprocessing change.
