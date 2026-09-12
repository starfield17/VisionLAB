"""Guarded ONNX export plus an executable probe, recorded as an export record.

The export input must be the trained checkpoint the training record names, and
the exported bytes are the ones the package ships. The probe records what the
exported graph actually is - input/output names, shapes, dtypes, score and class
semantics, whether NMS is embedded, and whether it agrees with the training
framework on the same image - instead of trusting upstream documentation.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from contractcheck.common import sha256_file
from contractcheck.loader import read_and_parse

from . import guard
from .runrecords import build_export_record


def build_export_command(*, checkpoint: Path, imgsz: int, nms: bool = False, device: str = 'cpu',
                         format: str = 'onnx', simplify: bool = False, half: bool = False,
                         dynamic: bool = False) -> list[str]:
    """Ultralytics export CLI command; the graph is written next to the checkpoint."""
    return ['yolo', 'export', f'model={Path(checkpoint).resolve()}', f'format={format}',
            f'imgsz={int(imgsz)}', f'nms={nms}', f'half={half}', f'dynamic={dynamic}',
            f'simplify={simplify}', f'device={device}', 'verbose=True']


def exported_model_path(staging: Path, checkpoint_name: str) -> Path:
    return Path(staging) / (Path(checkpoint_name).stem + '.onnx')


def build_probe_command(*, model: Path, checkpoint: Path, image: Path, imgsz: int, json_out: Path,
                        score_threshold: float = 0.001, device: str = 'cpu') -> list[str]:
    """Runner command that inspects the exported graph and exercises it."""
    return [sys.executable, '-m', 'model_lab.runners.onnx_probe',
            '--model', str(Path(model).resolve()),
            '--checkpoint', str(Path(checkpoint).resolve()),
            '--image', str(Path(image).resolve()),
            '--imgsz', str(int(imgsz)),
            '--score-threshold', str(float(score_threshold)),
            '--device', device,
            '--json-out', str(Path(json_out).resolve())]


def _trained_model_hash(training_record: Path) -> str:
    training = read_and_parse(Path(training_record))
    trained = {entry['artifact']['sha256'] for entry in training['outputs'] if entry['role'] == 'model'}
    if len(trained) != 1:
        raise ValueError('training record must name exactly one model output')
    return trained.pop()


def run_export(*, checkpoint: Path, dataset_manifest: Path, probe_image: Path, staging: Path,
               guard_output: Path, probe_output: Path, output: Path, record_id: str, imgsz: int,
               training_record: Path, implementation_version: str, nms: bool = False,
               simplify: bool = False, export_timeout: float = 1800, probe_timeout: float = 900,
               minimum_available: int = 3 * 1024**3) -> dict:
    """Export the trained checkpoint, probe the graph, and write the export record."""
    checkpoint, staging, output = Path(checkpoint), Path(staging), Path(output)
    probe_output = Path(probe_output)
    training = read_and_parse(Path(training_record))
    if sha256_file(checkpoint) != _trained_model_hash(Path(training_record)):
        raise ValueError('export input is not the trained model the training record names')
    dataset_hash = next(entry['artifact']['sha256'] for entry in training['inputs']
                        if entry['role'] == 'dataset')
    if sha256_file(dataset_manifest) != dataset_hash:
        raise ValueError('training record does not reference the supplied dataset manifest')
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    candidate = staging / checkpoint.name
    shutil.copyfile(checkpoint, candidate)
    if sha256_file(candidate) != sha256_file(checkpoint):
        raise ValueError('staged checkpoint bytes differ from the source checkpoint')
    status = guard.run(build_export_command(checkpoint=candidate, imgsz=imgsz, nms=nms, simplify=simplify),
                       Path(guard_output), minimum_available=minimum_available, timeout=export_timeout)
    if status != 0:
        raise ValueError(f'guarded export failed; see {Path(guard_output) / "result.json"}')
    graph = exported_model_path(staging, candidate.name)
    if not graph.is_file():
        raise ValueError(f'export did not produce {graph.name}')
    probe_output.parent.mkdir(parents=True, exist_ok=True)
    probe_guard = Path(str(guard_output) + '-probe')
    status = guard.run(build_probe_command(model=graph, checkpoint=candidate, image=probe_image,
                                           imgsz=imgsz, json_out=probe_output), probe_guard,
                       minimum_available=minimum_available, timeout=probe_timeout)
    if status != 0:
        raise ValueError(f'guarded probe failed; see {probe_guard / "result.json"}')
    probe = read_and_parse(probe_output)
    if probe.get('checker') != 'passed' or len(probe.get('outputs', [])) != 1:
        raise ValueError('exported graph failed ONNX checking or has an unsupported output count')
    declared = probe.get('declared', {})
    record = build_export_record(
        record_id=record_id, checkpoint=checkpoint, model_artifact=graph, probe_artifact=probe_output,
        config={
            'format': 'onnx', 'imgsz': int(imgsz), 'dynamic': False, 'half': False,
            'simplify': bool(simplify), 'embedded_nms': bool(nms),
            'decoder': declared.get('decoder'), 'box_format': declared.get('box_format'),
            'box_units': declared.get('box_units'), 'class_index_base': declared.get('class_index_base'),
            'input': probe['inputs'][0], 'output': probe['outputs'][0],
            'contract_stretch': probe.get('contract_stretch', {}),
            'probe_onnxruntime_version': probe.get('onnxruntime_version'),
            'probe_providers': probe.get('providers', []),
        },
        implementation_version=implementation_version)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record
