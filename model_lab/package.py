"""Assemble a Model Package from real run artifacts and validate it end to end.

Only real bytes are packaged: the training record's single model output, the
dataset manifest that training consumed, the evaluation metrics artifact, the
exported graph and its probe. Assembly is atomic, refuses to overwrite an
existing package, and finishes by running the offline preflight and full local
provenance audit; a package that does not audit is never published.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from contractcheck.common import sha256_file
from contractcheck.loader import SchemaStore, read_and_parse
from contractcheck.package import AdapterRegistry, PackageValidator

from .runrecords import utc_now, validate_run_record


def _spec_registry(adapter_spec: Path) -> AdapterRegistry:
    spec = read_and_parse(Path(adapter_spec))
    if set(spec) != {'id', 'version', 'formats', 'config_schema'}:
        raise ValueError('adapter spec requires exactly id, version, formats, config_schema')
    registry = AdapterRegistry()
    registry.register(spec['id'], spec['version'], spec['config_schema'], formats=tuple(spec['formats']))
    return registry


def _hash_role(record: dict, side: str, role: str) -> set:
    return {entry['artifact']['sha256'] for entry in record[side] if entry['role'] == role}


def assemble(*, destination: Path, package_id: str, dataset_manifest: Path, training_record: Path,
             evaluation_record: Path, export_record: Path, model_artifact: Path, checkpoint: Path,
             metrics_artifact: Path, probe_artifact: Path, adapter_spec: Path, adapter_id: str,
             adapter_version: str, adapter_config: dict, preprocessing: dict,
             confidence_threshold: float = 0.25,
             iou_threshold: float = 0.7, nms_mode: str = 'none', store_roots: tuple[Path, ...] = (),
             created_at: str | None = None) -> dict:
    """Publish one validated package directory built from retained run artifacts."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    dataset_manifest, training_record = Path(dataset_manifest), Path(training_record)
    evaluation_record, export_record = Path(evaluation_record), Path(export_record)
    model_artifact, checkpoint = Path(model_artifact), Path(checkpoint)
    metrics_artifact, probe_artifact = Path(metrics_artifact), Path(probe_artifact)
    for path in (dataset_manifest, training_record, evaluation_record, export_record,
                 model_artifact, checkpoint, adapter_spec, metrics_artifact, probe_artifact):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    training = validate_run_record(read_and_parse(training_record))
    evaluation = validate_run_record(read_and_parse(evaluation_record))
    export = validate_run_record(read_and_parse(export_record))
    dataset = read_and_parse(dataset_manifest)
    dataset_hash = sha256_file(dataset_manifest)
    if _hash_role(training, 'inputs', 'dataset') != {dataset_hash}:
        raise ValueError('training record does not consume the supplied dataset manifest')
    trained = _hash_role(training, 'outputs', 'model')
    if len(trained) != 1:
        raise ValueError('training record must name exactly one model output')
    for stage, record in (('evaluation', evaluation), ('export', export)):
        if _hash_role(record, 'inputs', 'model') != trained:
            raise ValueError(f'{stage} record does not consume the trained model')
    packaged = sha256_file(model_artifact)
    if _hash_role(export, 'outputs', 'model') != {packaged}:
        raise ValueError('packaged model bytes are not the export output')
    if sha256_file(checkpoint) not in trained:
        raise ValueError('packaged checkpoint bytes are not the training output')
    if evaluation['config'].get('sample_count', 0) < 1:
        raise ValueError('evaluation record must declare a positive sample count')
    _check_declared_output(evaluation, 'evaluation/metrics.json', metrics_artifact)
    _check_declared_output(export, 'export/probe.json', probe_artifact)
    registry = _spec_registry(adapter_spec)
    if registry.get(adapter_id, adapter_version) is None:
        raise ValueError('adapter spec does not declare the requested adapter ID/version')
    created = created_at or utc_now()
    labels = {'schema_version': '1.0.0', 'ontology_id': dataset['ontology']['id'],
              'ontology_version': dataset['ontology']['version'],
              'classes': [dict(entry, output_index=index)
                          for index, entry in enumerate(dataset['ontology']['classes'])]}
    manifest = {
        'schema_version': '1.0.0', 'package_id': package_id, 'created_at': created,
        'task': 'object_detection',
        'model': {
            'artifact': {'path': 'model.onnx', 'sha256': packaged},
            'format': 'onnx',
            'adapter': {'id': adapter_id, 'version': adapter_version, 'config': adapter_config},
        },
        'labels': {'path': 'labels.json', 'sha256': ''},
        'preprocessing': {'path': 'preprocessing.json', 'sha256': ''},
        'postprocessing': {'output_contract': 'adapter_detections_v1',
                           'confidence_threshold': float(confidence_threshold),
                           'nms': {'mode': nms_mode, 'iou_threshold': float(iou_threshold)}},
        'trace': {
            'dataset': {'id': dataset['id'], 'version': dataset['version'],
                        'artifact': {'path': 'config/dataset.json', 'sha256': dataset_hash}},
            'training': {'id': training['id'], 'artifact': {'path': 'records/training.json',
                                                            'sha256': sha256_file(training_record)}},
            'evaluation': {'id': evaluation['id'], 'artifact': {'path': 'records/evaluation.json',
                                                                'sha256': sha256_file(evaluation_record)}},
            'export': {'id': export['id'], 'artifact': {'path': 'records/export.json',
                                                        'sha256': sha256_file(export_record)}},
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.package-', dir=destination.parent) as temp:
        root = Path(temp) / 'package'
        (root / 'config').mkdir(parents=True)
        (root / 'records').mkdir()
        (root / 'evaluation').mkdir()
        (root / 'export').mkdir()
        (root / 'weights').mkdir()
        shutil.copyfile(dataset_manifest, root / 'config' / 'dataset.json')
        shutil.copyfile(training_record, root / 'records' / 'training.json')
        shutil.copyfile(evaluation_record, root / 'records' / 'evaluation.json')
        shutil.copyfile(export_record, root / 'records' / 'export.json')
        shutil.copyfile(checkpoint, root / 'weights' / 'best.pt')
        shutil.copyfile(model_artifact, root / 'model.onnx')
        shutil.copyfile(metrics_artifact, root / 'evaluation' / 'metrics.json')
        shutil.copyfile(probe_artifact, root / 'export' / 'probe.json')
        (root / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n', encoding='utf-8')
        (root / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2) + '\n', encoding='utf-8')
        manifest['labels']['sha256'] = sha256_file(root / 'labels.json')
        manifest['preprocessing']['sha256'] = sha256_file(root / 'preprocessing.json')
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        validator = PackageValidator(SchemaStore(), registry)
        preflight = validator.validate(root, 'preflight')
        if not preflight.ok:
            raise ValueError('package preflight failed:\n' + '\n'.join(str(e) for e in preflight.errors))
        audit = validator.validate(root, 'audit', tuple(Path(path) for path in store_roots))
        if not audit.ok:
            raise ValueError('package audit failed:\n' + '\n'.join(str(e) for e in audit.errors))
        root.rename(destination)
    return manifest


def _check_declared_output(record: dict, declared_path: str, artifact: Path) -> None:
    """The supplied bytes must be exactly what the record declares at that path."""
    matches = [entry for entry in record['outputs'] if entry['artifact']['path'] == declared_path]
    if len(matches) != 1 or matches[0]['artifact']['sha256'] != sha256_file(artifact):
        raise ValueError(f'{record["stage"]} record does not declare {declared_path} with these bytes')
