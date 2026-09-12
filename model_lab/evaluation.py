"""Guarded validation runs that produce an audited evaluation record.

The record names its split, metric definitions, thresholds and sample count and
links the exact dataset manifest and trained checkpoint by hash. The training
framework runs only inside a guarded subprocess (`model_lab.runners.yolo_val`);
this module never imports it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from contractcheck.loader import read_and_parse

from . import guard
from .runrecords import build_evaluation_record

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
METRIC_DEFINITIONS = {
    'precision': 'Ultralytics detection precision at config.thresholds.confidence',
    'recall': 'Ultralytics detection recall at config.thresholds.confidence',
    'mAP50': 'Ultralytics mAP at IoU 0.50, single IoU',
    'mAP50-95': 'Ultralytics mAP averaged over IoU 0.50:0.05:0.95',
}


def build_val_command(*, checkpoint: Path, data: Path, project: Path, name: str, imgsz: int,
                      batch: int, device: str, json_out: Path, confidence: float, iou: float) -> list[str]:
    """Runner command for a guarded validation run; executed with this interpreter."""
    if not name:
        raise ValueError('run name must be a nonempty string')
    return [sys.executable, '-m', 'model_lab.runners.yolo_val',
            '--model', str(Path(checkpoint).resolve()),
            '--data', str(Path(data).resolve()),
            '--project', str(Path(project).resolve()),
            '--name', name,
            '--imgsz', str(int(imgsz)),
            '--batch', str(int(batch)),
            '--device', device,
            '--conf', str(float(confidence)),
            '--iou', str(float(iou)),
            '--json-out', str(Path(json_out).resolve())]


def count_split_images(data_yaml: Path, split: str = 'val') -> int:
    """Count real image files in the exported YOLO split named by data.yaml."""
    config = yaml.safe_load(Path(data_yaml).read_text(encoding='utf-8'))
    relative = config.get(split)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f'data.yaml has no {split!r} split path')
    folder = (Path(data_yaml).resolve().parent / relative).resolve()
    if not folder.is_dir():
        raise ValueError(f'YOLO split directory is missing: {relative}')
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def run_evaluation(*, checkpoint: Path, data: Path, dataset_manifest: Path, project: Path, name: str,
                   guard_output: Path, metrics_out: Path, output: Path, record_id: str, split: str,
                   imgsz: int, batch: int, device: str, implementation_version: str,
                   confidence: float = 0.001, iou: float = 0.7, timeout: float = 3600,
                   minimum_available: int = 3 * 1024**3) -> dict:
    """Run guarded validation, then write the evaluation record and metrics artifact."""
    dataset_manifest, metrics_out, output = Path(dataset_manifest), Path(metrics_out), Path(output)
    manifest = read_and_parse(dataset_manifest)
    expected = sum(1 for item in manifest['items'] if item['split'] == split)
    observed = count_split_images(Path(data), 'val' if split == 'validation' else split)
    if expected != observed:
        raise ValueError(f'YOLO export has {observed} {split} images but the dataset manifest has {expected}')
    if expected < 1:
        raise ValueError(f'dataset manifest has no {split} items')
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    command = build_val_command(checkpoint=checkpoint, data=data, project=project, name=name, imgsz=imgsz,
                                batch=batch, device=device, json_out=metrics_out,
                                confidence=confidence, iou=iou)
    status = guard.run(command, Path(guard_output), minimum_available=minimum_available, timeout=timeout)
    if status != 0:
        raise ValueError(f'guarded validation failed; see {Path(guard_output) / "result.json"}')
    report = read_and_parse(metrics_out)
    overall = report['overall']
    metrics = {key: float(overall[key]) for key in METRIC_DEFINITIONS}
    record = build_evaluation_record(
        record_id=record_id, dataset_manifest=dataset_manifest, checkpoint=Path(checkpoint),
        metrics_artifact=metrics_out, split=split, sample_count=expected,
        metric_definitions=METRIC_DEFINITIONS, metrics=metrics,
        thresholds={'confidence': confidence, 'iou': iou},
        config={'data_yaml': Path(data).name, 'imgsz': int(imgsz), 'batch': int(batch),
                'runner': 'model_lab.runners.yolo_val',
                'validated_images': int(report.get('validated_images') or expected)},
        implementation_version=implementation_version)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record
