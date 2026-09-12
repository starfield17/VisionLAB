# Friction log

## Initial validator and deployment-core implementation

- Trigger: illustrative package records were being used as a full-audit success fixture.
- Constraint: the contract requires evaluation/model lineage that the illustrative records deliberately do not contain.
- Resolution: preserve the illustrative files; reject their full audit and build a separate consistent synthetic fixture for positive audit tests.
- Consequence: tests verify trace metadata without claiming actual training, accuracy or executable model support.
- Follow-up: supply a real reviewed dataset and model package when implementing the first runtime adapter.

## Audit metadata conventions

- Trigger: run-record config is an open object and I/O roles were strings without named lineage conventions.
- Constraint: checking whether evaluation/export consumed the trained model requires an unambiguous metadata edge.
- Resolution: use roles `dataset` and `model`; evaluation config has `split`, `metric_definitions`, `thresholds`, `sample_count`; metric keys must match definitions.
- Consequence: contract JSON shapes remain unchanged; prose and validator documentation now specify the convention.
- Follow-up: Model Lab producers must emit these fields and test against full audit.

## Guard memory floor vs quantized checkpoint load

- Trigger: the guarded probe stopped with `memory_limit` before the model produced any result.
- Constraint: the 4.7 GB quantized checkpoint loads in a single process that peaks near 7 GB RSS (weights plus Metal/accelerator buffers and decode cache), and system available memory transiently dips below the default 3 GiB guard floor during load.
- Resolution: measured peak RSS and available-memory floor, then reran the probe with an explicitly lowered floor recorded as an experiment (model load peak ~7 GiB, floor ~1 GiB). The smaller-image (448 longest side) retry did not change the outcome because the peak is dominated by the checkpoint, not the image.
- Consequence: the guard worked as designed; `workdir` run dirs preserve raw JSON, stdout/stderr, guard result and immutable labeling-run records for every image.
- Follow-up: a host intended for this labeling route must size available memory above the checkpoint peak plus the floor, or accept a documented floor override as an explicit experiment.

## UTC precision

- Trigger: direct string comparison misordered timestamps with fractional seconds, while datetime alone loses sub-microsecond precision.
- Constraint: RFC 3339 fractions have no fixed digit count.
- Resolution: compare validated calendar seconds and exact Decimal fractions; reject leap seconds because the core has no leap-second time source.
- Consequence: portable deterministic ordering without truncation; timestamp producers must use ordinary UTC civil seconds.
- Follow-up: version a leap-second policy only if an actual Source requires it.

## Two `model` outputs in an early training record

- Trigger: the first integration training record marked both `weights/best.pt` and `weights/last.pt` with role `model`, while the audited lineage convention expects training to name exactly one trained model that evaluation and export consume.
- Constraint: the record was already an immutable artifact, and quietly editing it would have destroyed the evidence it carries.
- Resolution: the original file is preserved untouched; `model_lab record-training` derives a new, schema-valid record from the same run bytes, naming `best.pt` as the model output and `last.pt` as a checkpoint.
- Consequence: packages built from that run trace a single trained-model identity, and every hash in the new record still resolves to the original bytes.
- Follow-up: write training records through the Model Lab command from the start of a run's life.

## Exported ONNX bytes are not reproducible

- Trigger: exporting the same checkpoint twice with identical arguments produced different `.onnx` bytes.
- Constraint: the contract hashes exact stored bytes, so byte identity cannot be assumed for a graph produced by a third-party exporter.
- Resolution: the package records the hash of the bytes actually exported and probes those same bytes; determinism is claimed for upstream artifacts (checkpoints, datasets), not for exporter output.
- Consequence: lineage stays verifiable, but `export` cannot be treated as a reproducible build step without pinning exporter internals.
- Follow-up: an export that must be reproducible needs a documented, pinned exporter configuration or a byte-stable compilation step.

## End-to-end export changes the score scale

- Trigger: probing a real end-to-end ONNX export showed that its confidence values differ from the same checkpoint's default PyTorch path on the same image, while agreeing with the framework's own ONNX reader to ~1e-7.
- Constraint: the adapter must not invent a calibration step, and the package threshold is data, not an adapter internal.
- Resolution: the probe records both references, the adapter decodes only what the graph emits, and the packaged confidence threshold is left at a production value; empty events are reported as empty. Measured on one validation image: framework ONNX reader maximum score 0.276 (matching the raw decode), PyTorch checkpoint path 0.589, with the same detection count.
- Consequence: a model exported with an end-to-end head may need its threshold chosen from measured behaviour on a validation split rather than inherited from the training-time default.
- Follow-up: pick the packaged threshold from validation evidence when the exported head's scores are calibrated differently.

## Letterbox and stretch differ for framework-trained models

- Trigger: training and the framework's own inference letterbox images, while the contract 1.0.0 preprocessing profile supports only bilinear stretch.
- Constraint: the contract documents stretch and its inverse as the only supported geometry; silently letterboxing inside an adapter would break the boundary that makes detections auditable.
- Resolution: the exported model is deployed through the contract's stretch path, the probe records what the model emits on a stretched input, and no adapter performs an undeclared resize.
- Consequence: deployments inherit a measured, documented difference rather than an implicit one. On a validation image the same graph produced a maximum score of 0.276 on a letterboxed input and 0.046 on the contract's stretched input, so a production threshold inherited from letterbox behaviour can legitimately yield empty events.
- Follow-up: supporting letterbox requires an explicit, versioned preprocessing contract change with its own inverse transform and tests.

## Numpy inputs hide channel order in reference comparisons

- Trigger: comparing a raw runtime decode against the training framework on the same image produced large, inconsistent score differences, while comparing against the framework loading the same exported graph produced exact agreement.
- Constraint: the framework's loader treats a numpy array as BGR (its documented convention) and a file path as an ordinary RGB image, so the two calls are not the same experiment.
- Resolution: the probe passes a file path for every framework reference and keeps numpy only for the runtime call that the adapter performs.
- Consequence: reference numbers are trustworthy; a numpy-based comparison would have blamed the adapter for a channel swap.
- Follow-up: when a comparison disagrees, first prove both sides received identical pixels.

## EXIF-oriented images cannot be ingested without box remapping

- Trigger: 153 of 1500 source images carry a non-trivial EXIF orientation while their published boxes refer to the stored pixel frame.
- Constraint: canonical images are the dataset's coordinate frame, and the contract requires boxes inside that frame.
- Resolution: ingestion applies the orientation for use as data and stops such images from entering the release, recording the count instead of guessing a transform.
- Consequence: the dataset is smaller than the source archive, and the reason is auditable rather than silent.
- Follow-up: rotating boxes together with the image would let these items be released, but only with a documented, tested transform.

## Type checking depends on the host's stub set

- Trigger: the acceptance gate's type-check step failed on a training host while the same command passed on the development machine, because that host's installed third-party stubs use syntax newer than the project's declared minimum Python target.
- Constraint: the project targets an older Python floor than some current stub packages assume, and the gate must not be weakened to make one host green.
- Resolution: boundary, lint, unit-test and packaging gates were run on the training host, and the type-check step is recorded as environment-blocked there rather than skipped in the repository; the authoritative full gate runs where the toolchain matches the project's target.
- Consequence: environment drift in type stubs is visible instead of silently tolerated.
- Follow-up: pin or vendor the type stub set used by the gate when reproducible cross-host type checking matters.
