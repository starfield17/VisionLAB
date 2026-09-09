# Core Contracts 1.0.0

Status: v1 implementation baseline. These contracts make INTENTION.md and ARCHITECTURE.md concrete. Scope is image object detection; video is processed as independent frames. Segmentation, tracking, classification-only tasks, rotated boxes, and dynamic batch inputs require future contracts.

Normative words MUST and MUST NOT describe required behavior. JSON Schema Draft 2020-12 validates document structure; the semantic rules below are equally mandatory. Implemented validation and runtime coverage is recorded in `../docs/CONTRACT_COVERAGE.md`.

## Common rules

- Every main document has `schema_version: "1.0.0"`. Consumers MUST reject unsupported versions before use. Version changes require explicit consumer support; major versions signal incompatible changes, minor versions additive changes, and patches clarifications. Strict schemas reject unknown fields. Adapter config and run config are the only explicitly open implementation-specific objects.
- IDs are nonempty, opaque, case-sensitive strings. Class IDs are strings, never array positions. IDs MUST be unique within their owner; released dataset `(id, version)`, annotation IDs, package IDs and run IDs are immutable. Changed content receives a new version or ID. Ontology IDs/versions and label changes follow the same rule.
- Timestamps MUST be RFC 3339 UTC with `Z`; enable JSON Schema format checking and additionally reject non-UTC offsets. Hashes are SHA-256 of exact stored bytes, lowercase hex. No JSON reserialization precedes hashing.
- Artifact paths are relative POSIX paths. Reject empty segments, `.`, `..`, absolute paths, drive prefixes and backslashes. Resolve and check containment, including symlinks, before reading. Paths in dataset documents resolve from their containing directory. Manifest and run-record artifact paths resolve from package root. IDs and provenance refs are audit identifiers, not executable expressions or automatic network fetch instructions.
- Image dimensions refer to decoded images after EXIF orientation. Boxes are `[x_min, y_min, x_max, y_max]`, floating point pixels, origin top-left, exclusive maxima. Require `0 <= x_min < x_max <= width` and `0 <= y_min < y_max <= height`. Reject NaN and infinity throughout. Images have positive integer dimensions.
- JSON Schema cannot enforce uniqueness by selected fields, foreign keys, hashes, box ordering/bounds, time ordering, or leakage rules. These need semantic validation; passing Schema alone is insufficient.

## Dataset contract

Schema: `schemas/dataset.schema.json`, with `ontology.schema.json`.

A dataset manifest contains an inline ontology and items with immutable image hashes, dimensions, provenance, split, group ID, and exactly one approved annotation document reference. Class IDs and labels are unique within the ontology. Item IDs and image hashes are unique within a dataset version. Annotation item ID, image hash and ontology ID/version MUST match the dataset item and ontology.

An annotation document describes the entire image, not one object. Its object IDs are unique within that document. Every object's class ID exists in the ontology. Empty objects means a reviewer confirmed a negative image. Missing annotations never mean negative images.

Only approved annotations enter a released dataset, including training in v1. Validation and test are human-confirmed Gold Sets. Candidate/rejected records stay in the annotation audit store, outside released item references. Self-training experiments may keep separate experimental datasets; changing the release policy requires an explicit future contract decision.

Each item has exactly one split. Items from the same source sequence, scene or duplicate family MUST share a group ID, and all items in a group MUST share a split. Producers must detect known duplicates and related frames before assigning groups. A manifest may contain only one split (for example a test-only dataset); training jobs independently require a nonempty train split. Validation/test MUST NOT be used for gradient updates, and test MUST NOT be used for model or threshold selection.

Image provenance retains a stable source reference and acquisition time; source records may hold acquisition details externally. Dataset versions must be immutable snapshots, not mutable directories under a reused version.

## Annotation contract

Schema: `schemas/annotation.schema.json`.

`source` identifies the original producer, run, and tool version. VLM output additionally records a model and prompt reference; these references MUST resolve to immutable versions in the producer's audit store. LocateAnything model/config details belong to its immutable run record. Human production identifies a person or review-system identity without embedding credentials.

Machine output MUST initially be `candidate`. Approval or rejection creates a new immutable annotation ID with `supersedes` pointing to the prior record and preserves original source provenance. Human-created annotations may be approved immediately with a review record. Approved/rejected records require a human reviewer ID, review timestamp and reason. Review covers class correctness, geometry, and missing objects across the entire image; an automated check alone cannot set approved. Review time MUST be at or after creation time. Revision history MUST be acyclic and refer to the same item/ontology. Correcting an approved record creates a new record; published datasets remain unchanged.

Object confidence is optional evidence about the producing tool, not a substitute for review. Source kind `self_training` requires experiment and model references and MUST be explicitly selected by an experiment; it is never the default annotation producer.

## Model Package contract

Schemas: `model-package.schema.json` (manifest), `labels.schema.json`, `preprocessing.schema.json`, `run-record.schema.json`.

A package is a directory or archive containing `manifest.json`, the model artifact, labels, preprocessing, and the dataset/run records referenced by the manifest. File names other than `manifest.json` are selected by manifest paths. All referenced audit records and model configuration needed for inference MUST be included. Raw dataset images and annotations may be retained in a separate immutable dataset store: Deploy does not open those files. Dataset manifest hashes preserve the exact snapshot identity.

Deploy MUST verify manifest structure, all directly referenced file hashes, labels/preprocessing schemas, trace record identities, and adapter support before starting. The caller supplies a trusted expected manifest hash when integrity of the manifest itself is required; the package cannot self-authenticate. Unknown formats, adapters, versions, or config fields MUST fail startup with an actionable error. No fallback model or inferred preprocessing defaults.

Labels map each supported model output index to a stable ontology class ID and display label. Index, ID and label are each unique. Indices may be sparse for adapters with a background class. Labels MUST match the traced ontology; a declared subset is permitted. Model output classes not in this mapping are errors except background outputs explicitly handled by the adapter's validated config.

Preprocessing v1 accepts one oriented 8-bit, three-channel image and produces one float32 tensor with batch size 1. Convert channels to the declared RGB/BGR order; bilinear stretch to `(width, height)` with half-pixel sampling (`align_corners=false`) and edge clamping; then calculate `(pixel * scale - mean[channel]) / std[channel]`. Mean/std use declared channel order. NCHW means `[1,3,height,width]`, NHWC means `[1,height,width,3]`. Do not quantize resized floating point values back to bytes. Grayscale/alpha handling belongs to Source and must yield three channels before inference. Letterbox, integer inputs, dynamic shapes and other transforms are unsupported by this version.

An inference adapter's exact ID/version selects a statically registered implementation, not downloadable code. Its config schema is owned and validated by that adapter. Before Core postprocessing it returns decoded detections in original-image pixel coordinates, class IDs resolved through labels, and scores in `[0,1]`. It MUST invert preprocessing, clip boxes to the image and discard zero-area boxes. Raw tensor layouts, score activations and background handling belong to the adapter. Future adapters must document these details and their config schema before registration. Format support is explicit; the manifest's string field does not promise arbitrary runtime support.

Core retains scores `>= confidence_threshold`. For `class_aware` NMS, sort by descending score, preserve adapter order on ties, and suppress lower-priority boxes of the same class when IoU is strictly greater than the configured threshold. Return survivors in that order. For `none`, retain adapter order after filtering; `iou_threshold` is present but ignored. If a model already performs NMS, select `none` and describe embedded behavior in adapter config to avoid accidental double suppression.

Trace contains dataset ID/version/hash and training/evaluation/export record IDs/hashes. Each run record carries implementation version/code revision, resolved configuration, immutable input/output artifact hashes, and numeric metrics when relevant. Evaluation MUST record the split, metric definitions, thresholds and sample count in config, and metric values in metrics. Production training inputs must identify the dataset; evaluation inputs must identify dataset and trained model; export inputs must identify trained model and export outputs must match the packaged model hash. Intermediate artifacts may live in a retained audit store and are not loaded by Deploy. Never hash a final manifest into a record included by that same manifest: that creates a hash cycle.

Deploy verifies audit record shapes and identities without importing training/evaluation code. Full provenance auditing additionally resolves run inputs/outputs and dataset content against the retained stores. Keep credentials and machine-specific paths out of packages.

## Detection Event contract

Schema: `schemas/detection-event.schema.json`.

One event means one successfully processed frame. `captured_at` is Source acquisition time; `emitted_at` is emission time and MUST be no earlier. Source assigns a new session ID on restart and monotonically increasing frame indices within a session (gaps allowed). Event IDs are unique; retries reuse the same ID and identical payload so sinks can deduplicate. Delivery and ordering guarantees are transport-specific, not implied by this contract.

`model` contains package ID and SHA-256 of the exact manifest bytes loaded. `image` describes the original oriented input, not resized tensor dimensions. Labels/classes match the package. Boxes follow common coordinate rules. Events contain postprocessed detections only. A valid empty `detections: []` means successful inference with no surviving detections. Decode failures, dropped frames, timeouts and inference errors MUST NOT emit empty success events; report operational errors through lifecycle/logging instead. Error wire protocols and statistics aggregation are outside these four contracts.

## Examples and validation

`examples/model-package/` is a linked contract fixture with real byte hashes for its included files. The `.fixture` image and model are text placeholders, not executable assets. Run records and metrics are illustrative; this test-only dataset is not training-ready. `detection-event.json`, `empty-event.json`, and `candidate-annotation.json` demonstrate runtime output and unreleased annotations.

Validation layers for implementation:

1. Parse strict JSON, reject duplicate keys/nonfinite values, validate schemas with format checking and local reference resolution.
2. Check cross-file identities, uniqueness, coordinates, states, time ordering, path containment and hashes.
3. Check provenance lineage, split leakage and adapter configuration/support.
4. Run a real supported model against a reviewed sample before claiming an executable end-to-end package.

The current fixtures cover layers 1 and linked-file integrity; they do not claim successful training or inference. See `../docs/IMPLEMENTATION_HANDOFF.md` for the next coding session.

## Implementation clarifications

The validator and deployment core now implement this baseline; see `../docs/CONTRACT_COVERAGE.md` for exact coverage. JSON document shapes are unchanged.

- Audited run-record inputs/outputs use role `dataset` for the traced dataset and `model` for trained/exported model artifacts. Training has exactly one model output identity; evaluation and export consume that identity. Additional roles remain permitted.
- Audited evaluation config names its required values `split`, `metric_definitions`, `thresholds`, `sample_count`. Metric definitions and numeric metrics have identical nonempty keys. Definitions are nonempty strings, thresholds a nonempty numeric map, and sample count a positive integer.
- Timestamp fractions retain arbitrary decimal precision. Leap-second civil timestamps are unsupported by this implementation and rejected explicitly.
- Annotation revisions resolve by immutable ID from available local annotation JSON records; unavailable required revisions produce `audit_incomplete`. External producer-system authenticity and model/prompt references require their own audit systems.
- The illustrative package is not a full-audit success case. Its shapes and hashes are useful examples; positive audit tests construct consistent synthetic records separately.
