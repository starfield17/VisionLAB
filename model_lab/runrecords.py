"""Portable immutable run records for training, evaluation and export.

Artifact paths inside a run record are resolved from the Model Package root once
the record is packaged, so they are written package-relative
(`config/dataset.json`, `weights/best.pt`, `model.onnx`). Intermediates that stay
in a retained audit store are resolved by the validator through explicit store
roots; this module never rewrites an existing record. Values must stay portable:
absolute paths and machine-specific strings are rejected by validation.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from contractcheck.common import scan_open_config, sha256_file
from contractcheck.errors import ValidationResult
from contractcheck.loader import SchemaStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def implementation(name: str, version: str, code_revision: str = 'pinned') -> dict:
    for label, value in (('name', name), ('version', version), ('code_revision', code_revision)):
        if not isinstance(value, str) or not value:
            raise ValueError(f'implementation {label} must be a nonempty string')
    return {'name': name, 'version': version, 'code_revision': code_revision}


def validate_run_record(record: dict) -> dict:
    """Validate one run record against the contract plus portability rules."""
    store, result = SchemaStore(), ValidationResult()
    store.check(record, 'run-record.schema.json', '<record>', result)
    if result.ok:
        scan_open_config(record['config'], '<record>', '/config', result)
    if not result.ok:
        raise ValueError('\n'.join(str(error) for error in result.errors))
    return record


def read_results_csv(path: Path) -> list[dict[str, float]]:
    """Read an Ultralytics results CSV into finite per-epoch numeric rows."""
    try:
        with Path(path).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f'cannot read results csv: {exc}') from exc
    if not rows:
        raise ValueError('results csv has no epoch rows')
    parsed = []
    for row in rows:
        values = {}
        for key, raw in row.items():
            if key is None:
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                raise ValueError(f'results csv contains a nonfinite value in {key!r}')
            values[key] = number
        parsed.append(values)
    return parsed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(path: Path) -> str:
    try:
        return sha256_file(Path(path))
    except OSError as exc:
        raise ValueError(f'cannot hash {Path(path).name}: {exc}') from exc


def build_training_record(*, run_dir: Path, dataset_manifest: Path, record_id: str,
                          profile: str, config_file: str, pretrained_ref: str,
                          pretrained_path: Path, dataset_yolo_ref: str,
                          implementation_version: str, created_at: str | None = None,
                          output: Path | None = None) -> dict:
    """Write the schema-valid training record next to a finished Ultralytics run.

    The dataset manifest is copied byte-for-byte into `<run_dir>/config/dataset.json`
    so the record's lineage edge points at the exact bytes that were trained on.
    """
    run_dir, dataset_manifest = Path(run_dir), Path(dataset_manifest)
    _require(run_dir.is_dir(), f'training run directory does not exist: {run_dir}')
    for name in ('args.yaml', 'results.csv', 'weights/best.pt', 'weights/last.pt'):
        _require((run_dir / name).is_file(), f'training run is missing {name}')
    dataset_hash = _hash(dataset_manifest)
    arguments = yaml.safe_load((run_dir / 'args.yaml').read_text(encoding='utf-8'))
    _require(isinstance(arguments, dict), 'args.yaml must be a mapping')
    config_dir = run_dir / 'config'
    config_dir.mkdir(exist_ok=True)
    shipped = config_dir / 'dataset.json'
    shipped.write_bytes(dataset_manifest.read_bytes())
    _require(_hash(shipped) == dataset_hash, 'packaged dataset copy differs from the source manifest')

    rows = read_results_csv(run_dir / 'results.csv')
    metric_keys = ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)')
    _require(set(metric_keys) <= set(rows[0]), 'results csv is missing detection metrics')
    best = max(rows, key=lambda row: row.get('metrics/mAP50-95(B)', float('-inf')))
    best_epoch = int(best.get('epoch', 0))
    metrics = {
        'P': float(best['metrics/precision(B)']),
        'R': float(best['metrics/recall(B)']),
        'mAP50': float(best['metrics/mAP50(B)']),
        'mAP50-95': float(best['metrics/mAP50-95(B)']),
        'best_epoch': float(best_epoch),
        'epochs_recorded': float(len(rows)),
    }
    for key, value in best.items():
        if key.startswith('train/') or key.startswith('val/'):
            metrics[key.split('/', 1)[0] + '_' + key.split('/', 1)[1]] = float(value)

    record = {
        'schema_version': '1.0.0',
        'id': record_id,
        'stage': 'training',
        'created_at': created_at or utc_now(),
        'implementation': implementation('ultralytics', implementation_version),
        'config': {
            'profile': profile,
            'config_file': config_file,
            'pretrained_weights': {'path': pretrained_ref, 'sha256': _hash(pretrained_path)},
            'dataset_yolo': dataset_yolo_ref,
            'epochs': float(arguments['epochs']),
            'imgsz': float(arguments['imgsz']),
            'batch': float(arguments['batch']),
            'val': bool(arguments['val']),
            'best_epoch': float(best_epoch),
        },
        'inputs': [{'role': 'dataset', 'artifact': {'path': 'config/dataset.json', 'sha256': dataset_hash}}],
        'outputs': [
            {'role': 'model', 'artifact': {'path': 'weights/best.pt', 'sha256': _hash(run_dir / 'weights/best.pt')}},
            {'role': 'checkpoint', 'artifact': {'path': 'weights/last.pt', 'sha256': _hash(run_dir / 'weights/last.pt')}},
            {'role': 'results', 'artifact': {'path': 'results.csv', 'sha256': _hash(run_dir / 'results.csv')}},
        ],
        'metrics': metrics,
    }
    destination = Path(output) if output is not None else Path(run_dir) / 'training-record.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(validate_run_record(record), indent=2) + '\n', encoding='utf-8')
    return record


def build_evaluation_record(*, record_id: str, dataset_manifest: Path, checkpoint: Path,
                            metrics_artifact: Path, split: str, sample_count: int,
                            metric_definitions: dict, metrics: dict, thresholds: dict,
                            config: dict, implementation_version: str,
                            created_at: str | None = None) -> dict:
    """Evaluation of one trained model on a declared split with recorded thresholds."""
    _require(split in ('validation', 'test'), 'evaluation split must be validation or test')
    _require(isinstance(sample_count, int) and sample_count >= 1, 'sample_count must be a positive integer')
    _require(bool(metric_definitions) and set(metric_definitions) == set(metrics or {}),
             'metric values and definitions must have identical nonempty keys')
    _require(bool(thresholds) and all(isinstance(v, (int, float)) for v in thresholds.values()),
             'thresholds must be a nonempty numeric map')
    record = {
        'schema_version': '1.0.0',
        'id': record_id,
        'stage': 'evaluation',
        'created_at': created_at or utc_now(),
        'implementation': implementation('ultralytics', implementation_version),
        'config': {
            'split': split,
            'metric_definitions': dict(metric_definitions),
            'thresholds': {key: float(value) for key, value in thresholds.items()},
            'sample_count': int(sample_count),
            **config,
        },
        'inputs': [
            {'role': 'dataset', 'artifact': {'path': 'config/dataset.json', 'sha256': _hash(dataset_manifest)}},
            {'role': 'model', 'artifact': {'path': 'weights/best.pt', 'sha256': _hash(checkpoint)}},
        ],
        'outputs': [{'role': 'metrics', 'artifact': {'path': 'evaluation/metrics.json',
                                                     'sha256': _hash(metrics_artifact)}}],
        'metrics': {key: float(value) for key, value in metrics.items()},
    }
    return validate_run_record(record)


def build_export_record(*, record_id: str, checkpoint: Path, model_artifact: Path,
                        probe_artifact: Path, config: dict, implementation_version: str,
                        created_at: str | None = None) -> dict:
    """Export of one trained checkpoint whose output bytes are the packaged model."""
    record = {
        'schema_version': '1.0.0',
        'id': record_id,
        'stage': 'export',
        'created_at': created_at or utc_now(),
        'implementation': implementation('ultralytics', implementation_version),
        'config': {
            'model_output': {'path': 'model.onnx', 'sha256': _hash(model_artifact)},
            **config,
        },
        'inputs': [{'role': 'model', 'artifact': {'path': 'weights/best.pt', 'sha256': _hash(checkpoint)}}],
        'outputs': [
            {'role': 'model', 'artifact': {'path': 'model.onnx', 'sha256': _hash(model_artifact)}},
            {'role': 'probe', 'artifact': {'path': 'export/probe.json', 'sha256': _hash(probe_artifact)}},
        ],
    }
    return validate_run_record(record)
