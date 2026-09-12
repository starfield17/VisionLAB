# Project v2 — Architecture Blueprint

## 1. System

```text
MODEL LAB
Raw Data
   ↓
Auto Label / Review
   ↓
Versioned Dataset
   ↓
Train → Evaluate → Export
   ↓
Model Package
   ↓
DEPLOY
Source → Inference → Detection Event → Statistics / Sink
```

There are only two repositories:

- **Model Lab**: Produces models.
- **Deploy**: Runs models.

Dependency direction is fixed and one-way:

```text
Agent / VLM / LocateAnything
            ↓
          Dataset
            ↓
           Train
            ↓
      Model Package
            ↓
          Deploy
```

---

## 2. Model Lab

Responsibilities:

- Dataset / Ontology
- Auto Label
- Agent workflow
- Training
- Evaluation
- Export / Quantization
- Model Package

### Auto Label

Default sources:

```text
LocateAnything → localization / grounding
Agent + VLM   → semantic labeling / verification
Human         → review / Gold Set
```

YOLO/Faster R-CNN are **not the default Auto Label producers**. The pseudo-labels they produce belong to a Self-Training Strategy.

All machine results first enter:

```text
Candidate Annotation → Validation/Review → Dataset
```

Validation/Test use a human-confirmed **Gold Set**.

---

## 3. Core Contracts

### Dataset Contract
Defines ontology, annotations, splits, provenance.

### Annotation Contract
Unifies candidate annotation formats from LocateAnything, Agent/VLM, and Human, and preserves provenance.

### Model Package Contract
The only long-term boundary between Model Lab and Deploy:

```text
model-package/
├── model
├── manifest
├── labels
└── preprocessing
```

### Detection Event Contract
Unifies deployment output from different traditional models:

```text
source
model
detections[]
  ├── bbox
  ├── class_id
  ├── label
  └── confidence
```

---

## 4. Deploy

Deploy runs only traditional vision models and does not include:

```text
Training / Auto Label / Agent / VLM / Ollama / LM Studio / LocateAnything
```

Execution chain:

```text
Source
  ↓
Inference Adapter
  ↓
Detection Event
  ├── Statistics
  └── Sink
```

### Edge / Remote

They are **profiles**, not two separate codebases:

```text
Edge Local:   Camera → Model → Event → Sink
Edge Stream:  Camera → Transport → Remote
Remote:       Stream/Camera → Model → Event → Statistics/Sink
```

### Adapters

Replaceable boundaries:

- `Source`: Camera / RTSP / Stream
- `Inference`: YOLO / Faster R-CNN / future model
- `Transport`: Edge ↔ Remote
- `Sink`: Serial / Stdout / Network

Serial/STM32 is only a Sink, not part of Core.

Deploy Core is responsible only for:

```text
configuration
contracts
pipeline
lifecycle
adapter composition
```

Core should not contain concrete business/implementation assumptions such as dataset, YOLO, STM32, Serial, RK3588, etc.

---

## 5. Repository Shape

```text
model-lab/
├── AGENTS.md
├── datasets/
├── autolabel/
│   ├── locate_anything/
│   └── validation/
├── prompts/
├── training/
├── evaluation/
├── export/
└── experiments/

deploy/
├── core/
├── inference/
├── sources/
├── transports/
├── sinks/
├── statistics/
└── profiles/
```

---

## 6. Construction Order

1. Freeze the Intention and the core Contracts.
2. Remove RK3588/RKNN and hardcoded label logic.
3. Build Model Lab: Dataset → Auto Label → Train → Evaluate → Export.
4. Refine Deploy: Source → Inference → Detection Event → Sink.
5. Move Serial to Sink; convert Edge/Remote to Profiles.
6. After logical boundaries are stable, physically split into two repositories.
7. Finally, do quantization, Remote Streaming, and new-model extension.

## 7. Contract Baseline

The implementation baseline is [Core Contracts 1.0.0](contracts/README.md), including JSON Schemas, linked illustrative examples, and semantic validation rules. The [implementation handoff](docs/IMPLEMENTATION_HANDOFF.md) defines the first coding slice. Contract v1 supports image object detection; wider preprocessing and task support require explicit versioned changes.

## 8. Implemented Foundation

The root Python distribution contains `contractcheck` (offline contracts), `deploy` (synchronous numerical core, ports, one ONNX Runtime detection adapter and one image-file Source) and `model_lab` (offline data production, training orchestration, evaluation, export and package assembly), plus the canonical schema resources. Deploy depends on contractcheck; the reverse dependency is forbidden by an executable check. Runtime adapters are statically composed with trusted metadata, and the concrete runtime frameworks stay inside boundary modules that the dependency check names explicitly. A Model Package produced by Model Lab has been evaluated, exported, packaged, fully audited and executed by Deploy. See [implementation handoff](docs/IMPLEMENTATION_HANDOFF.md), [coverage](docs/CONTRACT_COVERAGE.md) and [the current data round](docs/PUBLIC_DATASET_ROUND.md).
