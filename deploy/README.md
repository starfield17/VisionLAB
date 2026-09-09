# Deployment core

The core runs a synchronous single-model object-detection pipeline. Install with `python -m pip install '.[deploy]'` in an activated conda environment. The package includes no real model adapter or image decoder yet.

## Public surface

Import `Pipeline`, `RuntimeRegistry`, `Frame`, `SourceInfo`, `Detection`, `Transform`, `Source`, `InferenceAdapter` and `Sink` from `deploy`. Import concrete sinks from `deploy.sinks`. Adapter metadata types live in `contractcheck.package`.

| Port/value | Required behavior |
| --- | --- |
| Source | `open()`, `read() -> Frame or None`, `close()`; None means normal EOF only |
| Frame | Oriented RGB uint8 array `[height,width,3]`, SourceInfo, RFC 3339 UTC capture time |
| SourceInfo | Stable source ID/kind, new session ID on restart, nonnegative increasing frame index; gaps allowed |
| InferenceAdapter | `open(ValidatedPackage)`, `infer(float32 tensor, Transform) -> list[Detection]`, `close()` |
| Detection | Original-image pixel box `(xmin,ymin,xmax,ymax)`, stable string class ID, confidence `[0,1]` |
| Sink | `open()`, `write(event dict)`, `close()` |

Factories must be side-effect free. A port's `close()` must be safe after partial `open()` failure, release its owned resources, and not assume another port remains open. A failed `open()` is followed by its own cleanup as well as cleanup of previously opened ports.

`RuntimeRegistry.register(AdapterSpec(...), factory)` binds an exact ID/version, supported formats, closed config schema and implementation. No import-by-name, plugin discovery, or package-provided executable code is used. `Pipeline(source, sink, registry, package_dir, expected_manifest_hash=...)` performs preflight before calling any port's open method. Clock and event-ID factories can be injected for deterministic tests.

## Numerical boundary

Preprocessing converts RGB to the declared order, applies contract v1 bilinear stretch without intermediate byte rounding, normalizes, and returns a contiguous float32 tensor with batch size one. `Transform` records original/input dimensions; `original_box()` inverts stretch in tensor pixel coordinates, clips to the original image and returns None for zero-area results. Adapters convert normalized or center-size outputs to pixel corner boxes first.

Adapters own output decoding, background handling and label-index mapping. Core rejects nonfinite scores/coordinates, unknown classes and invalid original-image boxes before filtering. Confidence filtering is inclusive. Class-aware NMS is stable on equal scores, suppresses only when IoU is strictly above the threshold, and does not suppress across classes. `none` preserves adapter order after filtering.

## Lifecycle and output

Pipeline instances are single-use. Open order is Source → Adapter → Sink; close order is reversed. Normal EOF returns the number of successfully written events. Read, inference, write, schema, or cleanup errors fail the run. Every attempted open is paired with close, including interruption and partial initialization failures. Cleanup errors are logged; an existing primary failure is preserved, otherwise the first cleanup failure is raised after all cleanup attempts.

Exactly one event is written for each successful frame, including legitimate empty detections. Failures never create empty success events. Clock regressions, repeated frame indices, repeated event IDs and event/package mismatches fail before writing. Sources must maintain a meaningful acquisition clock; Core does not fabricate timestamps to hide clock skew. No automatic retries, concurrency, streaming queues, hot reload, or cross-run deduplication are provided. The current per-run sequence/ID bookkeeping is intended for bounded image batches; long-running stream adapters need an explicit retention policy before adoption.

`StdoutSink` writes one strict JSON object per line and flushes it. It does not close the caller's stream. A failed stream write can have partially written bytes; the core cannot make arbitrary sinks transactional or claim exactly-once delivery.

`deploy/tests/test_core.py` demonstrates the whole composition using two test adapters, a memory Source, and memory/JSONL sinks. Test doubles are never registered as production implementations.
