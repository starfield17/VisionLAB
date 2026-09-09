# contractcheck 0.2

Offline validators for Core Contracts 1.0.0. Validation never executes model code or downloads schema references. Runtime dependencies are `jsonschema` and `referencing`; NumPy and model runtimes are not required.

## Installation and checks

```sh
conda activate <env>
python -m pip install .
contractcheck validate <document.json>
contractcheck package preflight <package-dir> --adapter-spec <trusted-adapter.json>
contractcheck package audit <package-dir> --adapter-spec <trusted-adapter.json> --store-root <retained-store>
```

Run installation from the repository root. Installed schemas are loaded using package resources; `--schemas <directory>` explicitly overrides them. `--expected-manifest-hash <hex>` checks the caller's trusted manifest digest. Repeated `--store-root` adds local retained roots in order, and repeated `--adapter-spec` registers distinct trusted descriptions.

Exit codes: `0` success, `1` invalid document/artifact, `2` invalid CLI/schema/adapter configuration. Errors include document, JSON Pointer, stable rule name and message. Invalid document structure prevents semantic traversal of that document. Independent package documents still produce their own diagnostics.

## Adapter descriptions

A caller-owned adapter description has exactly these fields:

```json
{
  "id": "example-adapter",
  "version": "1",
  "formats": ["example-format"],
  "config_schema": {
    "type": "object",
    "properties": {},
    "additionalProperties": false
  }
}
```

This is an illustration, not an installed inference implementation. The caller must obtain descriptions from the trusted deployment composition, never accept a package-supplied description as proof of support. All config object schemas must reject unknown properties. References must be internal and resolvable; schema IDs cannot change resolution scope.

The old `--adapter ID@VER` flag now returns a configuration error explaining the migration. Python registration is `register(id, version, config_schema, formats=(... ,))`; ID/version alone is insufficient. In Deploy, register an `AdapterSpec` and its factory together through `RuntimeRegistry`.

## Validation modes

`validate` detects the document type and applies schema and available standalone semantics. Dataset validation opens images for hashing and annotations for semantic checking; standalone annotations can check geometry ordering, uniqueness and timestamps but cannot verify image bounds without dataset context. A standalone manifest check is not a package preflight.

`preflight` checks included artifacts, manifest/trace identity, dataset metadata, labels, preprocessing, supported format and adapter config. It does not open raw dataset images, annotations, or intermediate training assets.

`audit` also resolves raw dataset content and run inputs/outputs, reuses dataset/annotation semantics, checks revision chains, and verifies dataset → training model → evaluation/export associations. Unavailable referenced annotation history is `audit_incomplete`. An approved machine annotation must retain its candidate revision. Dataset dimensions are metadata: image decoding and EXIF orientation are not certified by this tool. Audit cannot establish human identity, actual training behavior or model accuracy; see the coverage matrix.

Artifact paths must stay inside their primary or explicit retained root after symlink resolution. A missing primary file may be found under a retained root using the same relative path. A present but invalid or hash-mismatched primary file fails; it cannot be hidden by a fallback. Dataset paths resolve from the manifest directory, run-record paths from package root. Included package records must actually be inside the package.

Evaluation metadata uses `config.split`, `config.sample_count`, `config.metric_definitions` and `config.thresholds`; metrics and definitions have identical nonempty keys. Run I/O roles `dataset` and `model` identify lineage edges. Other roles are allowed and their referenced bytes are audited too.

`PackageValidator.load()` returns `ValidatedPackage` or raises `PackageValidationError` carrying the `ValidationResult`. Treat returned metadata and package files as immutable for the duration of a run. `DetectionEventValidator.validate_document(event, package=...)` checks event/package identity and label agreement as well as standalone event rules.

## Verification

```sh
conda activate <env>
python -m pip install -e '.[dev]'
python tools/check.py
```

The command runs boundary/test-integrity checks, lint, type checks, all tests, sdist/wheel builds, and installed CLI checks outside the source tree. Tests use temporary copies and synthetic provenance records; they never promote the illustrative `.fixture` package to a real trained model.
