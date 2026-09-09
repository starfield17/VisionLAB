"""Offline deployment preflight and full local provenance audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from referencing.exceptions import Unresolvable

from . import common
from .annotation import check_classes
from .dataset import DatasetValidator
from .errors import ValidationResult
from .loader import SchemaStore, pointer


@dataclass(frozen=True)
class AdapterSpec:
    id: str
    version: str
    formats: tuple[str, ...]
    config_schema: dict[str, Any]


class AdapterRegistry:
    """Trusted support descriptions, supplied by callers rather than packages."""
    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], AdapterSpec] = {}

    def register(self, adapter_id: str, version: str, config_schema: dict[str, Any],
                 *, formats: tuple[str, ...]) -> None:
        if not adapter_id or not version or not formats or any(not isinstance(f, str) or not f for f in formats):
            raise ValueError('adapter requires ID, version and supported formats')
        if not isinstance(config_schema, dict):
            raise ValueError('adapter config schema must be an object schema')
        try:
            Draft202012Validator.check_schema(config_schema)
        except SchemaError as exc:
            raise ValueError(f'invalid adapter config schema: {exc.message}') from exc
        if config_schema.get('type') != 'object' or config_schema.get('additionalProperties') is not False:
            raise ValueError('adapter config schema must be a closed object')
        registry = Registry().with_resource('urn:adapter-config', Resource.from_contents(
            config_schema, default_specification=DRAFT202012))
        def check(node):
            if isinstance(node, dict):
                if '$id' in node:
                    raise ValueError('adapter config schemas must not override reference scope with $id')
                for keyword in ('$ref', '$dynamicRef'):
                    if keyword in node:
                        if not node[keyword].startswith('#'):
                            raise ValueError('adapter config schema refs must be internal')
                        try:
                            registry.resolver('urn:adapter-config').lookup(node[keyword])
                        except Unresolvable as exc:
                            raise ValueError('adapter config schema reference cannot be resolved') from exc
                if ('properties' in node or node.get('type') == 'object') and node.get('additionalProperties') is not False:
                    raise ValueError('adapter config object schemas must reject unknown properties')
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(config_schema)
        key = (adapter_id, version)
        if key in self._specs:
            raise ValueError(f'duplicate adapter {adapter_id}@{version}')
        self._specs[key] = AdapterSpec(adapter_id, version, formats, config_schema)

    def get(self, adapter_id: str, version: str) -> AdapterSpec | None:
        return self._specs.get((adapter_id, version))


@dataclass(frozen=True)
class ValidatedPackage:
    root: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    labels: dict[str, Any]
    preprocessing: dict[str, Any]
    model_path: Path


class PackageValidationError(ValueError):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__('\n'.join(str(e) for e in result.errors))


class PackageValidator:
    def __init__(self, store: SchemaStore, adapters: AdapterRegistry | None = None):
        self.store = store
        self.adapters = adapters or AdapterRegistry()

    def validate(self, package_dir: Path, mode: str = 'preflight', store_roots: tuple[Path, ...] = (),
                 expected_manifest_hash: str | None = None) -> ValidationResult:
        return self._validate(package_dir, mode, store_roots, expected_manifest_hash)[0]

    def load(self, package_dir: Path, expected_manifest_hash: str | None = None) -> ValidatedPackage:
        result, package = self._validate(package_dir, 'preflight', (), expected_manifest_hash)
        if not result.ok or package is None:
            raise PackageValidationError(result)
        return package

    def _validate(self, package_dir, mode, store_roots, expected):
        result = ValidationResult()
        if mode not in ('preflight', 'audit'):
            result.add_error(str(package_dir), '', 'mode', 'expected preflight or audit')
            return result, None
        root = Path(package_dir)
        resolver = common.ArtifactResolver()
        manifest_path = resolver.resolve('manifest.json', root, str(root), '', result)
        if manifest_path is None:
            return result, None
        manifest = self.store.load(manifest_path, 'model-package.schema.json', result)
        if manifest is None:
            return result, None
        document = str(manifest_path)
        try:
            manifest_hash = common.sha256_file(manifest_path)
        except OSError as exc:
            result.add_error(document, '', 'artifact_io', str(exc))
            return result, None
        if expected is not None and expected != manifest_hash:
            result.add_error(document, '', 'manifest_hash_expected', 'manifest differs from trusted hash')
        common.check_utc_timestamp(result, manifest['created_at'], document, '/created_at')
        model = manifest['model']
        model_path = resolver.artifact(model['artifact'], root, document, '/model/artifact', result)
        adapter = model['adapter']
        spec = self.adapters.get(adapter['id'], adapter['version'])
        if spec is None:
            result.add_error(document, '/model/adapter', 'unsupported_adapter', 'adapter ID/version is not registered')
        else:
            if model['format'] not in spec.formats:
                result.add_error(document, '/model/format', 'unsupported_format', 'format is not supported by adapter')
            validator = Draft202012Validator(spec.config_schema, registry=Registry(), format_checker=FormatChecker())
            for error in validator.iter_errors(adapter['config']):
                result.add_error(document, '/model/adapter/config' + pointer(error.absolute_path),
                                 'unsupported_adapter_config', error.message)
        common.scan_open_config(adapter['config'], document, '/model/adapter/config', result)

        def included(ref, field, schema):
            path = resolver.artifact(ref, root, document, field, result)
            return (self.store.load(path, schema, result), path) if path is not None else (None, None)

        labels, _ = included(manifest['labels'], '/labels', 'labels.schema.json')
        preprocessing, _ = included(manifest['preprocessing'], '/preprocessing', 'preprocessing.schema.json')
        ds, ds_path = included(manifest['trace']['dataset']['artifact'], '/trace/dataset/artifact', 'dataset.schema.json')
        if labels is not None:
            check_classes(labels['classes'], document, result, '/labels/classes')
            common.check_unique(result, ((c['output_index'], f'/labels/classes/{i}/output_index') for i, c in enumerate(labels['classes'])),
                                document, 'labels', 'output index')
        if ds is not None:
            trace = manifest['trace']['dataset']
            if (ds['id'], ds['version']) != (trace['id'], trace['version']):
                result.add_error(document, '/trace/dataset', 'record_identity', 'dataset identity differs')
            DatasetValidator(self.store, common.ArtifactResolver(store_roots)).validate_document(
                ds, ds_path, result, content=mode == 'audit')
            if labels is not None:
                ontology = ds['ontology']
                mapping = {c['id']: c['label'] for c in ontology['classes']}
                if (labels['ontology_id'], labels['ontology_version']) != (ontology['id'], ontology['version']):
                    result.add_error(document, '/labels', 'labels_ontology_mismatch', 'ontology identity differs')
                for i, entry in enumerate(labels['classes']):
                    if mapping.get(entry['id']) != entry['label']:
                        result.add_error(document, f'/labels/classes/{i}', 'labels_ontology_mismatch', 'class ID/label differs from ontology')
        records = {}
        for stage in ('training', 'evaluation', 'export'):
            entry = manifest['trace'][stage]
            record, path = included(entry['artifact'], f'/trace/{stage}/artifact', 'run-record.schema.json')
            if record is None:
                continue
            records[stage] = record
            if record['id'] != entry['id'] or record['stage'] != stage:
                result.add_error(str(path), '/id', 'record_identity', 'run record ID/stage differs from trace')
            common.check_utc_timestamp(result, record['created_at'], str(path), '/created_at')
            common.scan_open_config(record['config'], str(path), '/config', result)
            for side in ('inputs', 'outputs'):
                for i, io in enumerate(record[side]):
                    ref = io['artifact']
                    field = f'/{side}/{i}/artifact'
                    if not common.validate_relative_posix_path(ref['path']):
                        result.add_error(str(path), field + '/path', 'invalid_path', 'expected a clean relative POSIX path')
                        continue
                    try:
                        cycle = (root / ref['path']).resolve() == manifest_path.resolve()
                    except (OSError, RuntimeError, ValueError) as exc:
                        result.add_error(str(path), field, 'artifact_io', str(exc))
                        continue
                    if cycle or ref['sha256'] == manifest_hash:
                        result.add_error(str(path), field, 'hash_cycle', 'run record references its containing manifest')
                    elif mode == 'audit':
                        audit_resolver = common.ArtifactResolver(store_roots)
                        n = len(result.errors)
                        audit_resolver.artifact(ref, root, str(path), field, result)
                        # audit callers distinguish unavailable retained artifacts from included files.
                        if len(result.errors) > n and result.errors[-1].rule == 'artifact_missing':
                            result.errors.pop()
                            result.add_error(str(path), field, 'artifact_unresolved', 'retained artifact not found')
        if mode == 'audit' and len(records) == 3:
            self._lineage(manifest, records, document, result)
        package = None
        if result.ok and model_path is not None:
            package = ValidatedPackage(root.resolve(), manifest_hash, manifest, labels, preprocessing, model_path)
        return result, package

    def _lineage(self, manifest, records, document, result):
        def hashes(stage, side, role):
            return {io['artifact']['sha256'] for io in records[stage][side] if io['role'] == role}
        dataset_hash = manifest['trace']['dataset']['artifact']['sha256']
        for stage in ('training', 'evaluation'):
            if hashes(stage, 'inputs', 'dataset') != {dataset_hash}:
                result.add_error(document, f'/trace/{stage}', 'lineage_mismatch', 'run must reference traced dataset')
        trained = hashes('training', 'outputs', 'model')
        for stage in ('evaluation', 'export'):
            if len(trained) != 1 or hashes(stage, 'inputs', 'model') != trained:
                result.add_error(document, f'/trace/{stage}', 'lineage_mismatch', 'run must reference the trained model')
        if manifest['model']['artifact']['sha256'] not in hashes('export', 'outputs', 'model'):
            result.add_error(document, '/trace/export', 'export_mismatch', 'export output does not match packaged model')
        evaluation = records['evaluation']
        schema = {'type': 'object', 'required': ['split', 'metric_definitions', 'thresholds', 'sample_count'],
                  'properties': {'split': {'enum': ['validation', 'test']},
                                 'metric_definitions': {'type': 'object', 'minProperties': 1, 'additionalProperties': {'type': 'string', 'minLength': 1}},
                                 'thresholds': {'type': 'object', 'minProperties': 1, 'additionalProperties': {'type': 'number'}},
                                 'sample_count': {'type': 'integer', 'minimum': 1}}}
        for error in Draft202012Validator(schema).iter_errors(evaluation['config']):
            result.add_error(document, '/trace/evaluation/config' + pointer(error.absolute_path), 'evaluation_config', error.message)
        definitions = evaluation['config'].get('metric_definitions')
        if not evaluation.get('metrics') or not isinstance(definitions, dict) or set(evaluation.get('metrics', {})) != set(definitions):
            result.add_error(document, '/trace/evaluation', 'evaluation_metrics', 'metric values and definitions must have identical nonempty keys')
