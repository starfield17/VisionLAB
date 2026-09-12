"""Model Package assembly: real hash lineage, atomic publish, audit enforcement."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.package import assemble
from model_lab.runrecords import build_evaluation_record, build_export_record, build_training_record
from model_lab.tests.helpers import make_dataset, sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SPEC = REPO_ROOT / 'deploy' / 'adapters' / 'specs' / 'yolo26_onnx.adapter.json'

RESULTS = ('epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),'
           'metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0,lr/pg1,lr/pg2\n'
           '1,1.0,0.5,0.4,0.3,0.10,0.20,0.30,0.25,0.6,0.5,0.4,0.1,0.1,0.1\n')

PREPROCESSING = {
    'schema_version': '1.0.0', 'input_name': 'images', 'dtype': 'float32', 'layout': 'NCHW',
    'color_space': 'RGB', 'width': 64, 'height': 64,
    'resize': {'mode': 'stretch', 'interpolation': 'bilinear'},
    'normalization': {'scale': 1 / 255, 'mean': [0, 0, 0], 'std': [1, 1, 1]},
}


def build_artifacts(tmp_path, *, model_bytes=b'onnx-graph') -> dict:
    dataset = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'validation', 'objects': [{'class_id': 'metal', 'bbox': [2, 2, 9, 9]}]},
    ])
    run = tmp_path / 'run'
    (run / 'weights').mkdir(parents=True)
    (run / 'weights' / 'best.pt').write_bytes(b'best-checkpoint')
    (run / 'weights' / 'last.pt').write_bytes(b'last-checkpoint')
    (run / 'args.yaml').write_text('epochs: 1\nimgsz: 64\nbatch: 2\nval: true\n', encoding='utf-8')
    (run / 'results.csv').write_text(RESULTS, encoding='utf-8')
    pretrained = tmp_path / 'yolo.pt'
    pretrained.write_bytes(b'pretrained')
    training = build_training_record(run_dir=run, dataset_manifest=dataset, record_id='training-fixture',
                                     profile='fixture', config_file='model_lab/configs/fixture.yaml',
                                     pretrained_ref='workdir/models/yolo.pt', pretrained_path=pretrained,
                                     dataset_yolo_ref='workdir/yolo',
                                     implementation_version='8.4.145',
                                     created_at='2026-01-02T00:00:00Z')
    metrics = tmp_path / 'metrics.json'
    write_json(metrics, {'overall': {'precision': 0.5}})
    evaluation = build_evaluation_record(
        record_id='evaluation-fixture', dataset_manifest=dataset,
        checkpoint=run / 'weights' / 'best.pt', metrics_artifact=metrics, split='validation',
        sample_count=1, metric_definitions={'precision': 'definition'}, metrics={'precision': 0.5},
        thresholds={'confidence': 0.001, 'iou': 0.7},
        config={'imgsz': 64, 'data_yaml': 'data.yaml'}, implementation_version='8.4.145',
        created_at='2026-01-03T00:00:00Z')
    model = tmp_path / 'model.onnx'
    model.write_bytes(model_bytes)
    probe = tmp_path / 'probe.json'
    write_json(probe, {'checker': 'passed'})
    export = build_export_record(record_id='export-fixture', checkpoint=run / 'weights' / 'best.pt',
                                 model_artifact=model, probe_artifact=probe,
                                 config={'format': 'onnx', 'imgsz': 64},
                                 implementation_version='8.4.145',
                                 created_at='2026-01-04T00:00:00Z')
    return {'dataset': dataset, 'run': run, 'training': run / 'training-record.json',
            'training_record': training, 'evaluation': tmp_path / 'evaluation.json',
            'evaluation_record': evaluation, 'export': tmp_path / 'export.json',
            'export_record': export, 'model': model, 'probe': probe, 'metrics': metrics}


def write_records(artifacts: dict) -> None:
    write_json(artifacts['evaluation'], artifacts['evaluation_record'])
    write_json(artifacts['export'], artifacts['export_record'])


def assemble_fixture(artifacts: dict, destination: Path, **overrides):
    write_records(artifacts)
    arguments = dict(
        destination=destination, package_id='fixture-package', dataset_manifest=artifacts['dataset'],
        training_record=artifacts['training'], evaluation_record=artifacts['evaluation'],
        export_record=artifacts['export'], model_artifact=artifacts['model'],
        checkpoint=artifacts['run'] / 'weights' / 'best.pt', metrics_artifact=artifacts['metrics'],
        probe_artifact=artifacts['probe'], adapter_spec=ADAPTER_SPEC,
        adapter_id='yolo26-onnx-detections', adapter_version='1',
        adapter_config={'decoder': 'end2end_detections_v1', 'max_detections': 300},
        preprocessing=PREPROCESSING, confidence_threshold=0.25, iou_threshold=0.7,
        store_roots=(artifacts['dataset'].parent, artifacts['run']))
    arguments.update(overrides)
    return assemble(**arguments)


def test_package_assembly_passes_preflight_and_full_audit(tmp_path):
    artifacts = build_artifacts(tmp_path)
    manifest = assemble_fixture(artifacts, tmp_path / 'package')
    package = tmp_path / 'package'
    assert manifest['trace']['dataset']['id'] == 'fixture-set'
    assert manifest['trace']['training']['id'] == 'training-fixture'
    assert manifest['model']['artifact']['sha256'] == sha256_file(artifacts['model'])
    assert manifest['trace']['export']['artifact']['sha256'] == sha256_file(artifacts['export'])
    assert (package / 'weights' / 'best.pt').read_bytes() == b'best-checkpoint'
    assert (package / 'config' / 'dataset.json').read_bytes() == artifacts['dataset'].read_bytes()
    assert (package / 'evaluation' / 'metrics.json').read_bytes() == artifacts['metrics'].read_bytes()
    assert (package / 'manifest.json').is_file()


def test_labels_follow_ontology_order(tmp_path):
    artifacts = build_artifacts(tmp_path)
    assemble_fixture(artifacts, tmp_path / 'package')
    labels = json.loads((tmp_path / 'package' / 'labels.json').read_text())
    assert [entry['output_index'] for entry in labels['classes']] == [0, 1]
    assert [entry['id'] for entry in labels['classes']] == ['plastic', 'metal']
    assert labels['ontology_id'] == 'fixture-litter'


def test_package_refuses_to_overwrite(tmp_path):
    artifacts = build_artifacts(tmp_path)
    assemble_fixture(artifacts, tmp_path / 'package')
    with pytest.raises(FileExistsError):
        assemble_fixture(artifacts, tmp_path / 'package')


def test_package_rejects_model_that_is_not_the_export_output(tmp_path):
    artifacts = build_artifacts(tmp_path)
    artifacts['model'].write_bytes(b'tampered-graph')
    with pytest.raises(ValueError, match='not the export output'):
        assemble_fixture(artifacts, tmp_path / 'package')


def test_package_requires_a_single_trained_model(tmp_path):
    artifacts = build_artifacts(tmp_path)
    artifacts['training_record']['outputs'].append(
        {'role': 'model', 'artifact': {'path': 'weights/last.pt',
                                       'sha256': sha256_file(artifacts['run'] / 'weights' / 'last.pt')}})
    write_json(artifacts['training'], artifacts['training_record'])
    with pytest.raises(ValueError, match='exactly one model output'):
        assemble_fixture(artifacts, tmp_path / 'package')


def test_package_rejects_evaluation_of_a_different_checkpoint(tmp_path):
    artifacts = build_artifacts(tmp_path)
    other = tmp_path / 'other.pt'
    other.write_bytes(b'other-checkpoint')
    artifacts['evaluation_record'] = build_evaluation_record(
        record_id='evaluation-other', dataset_manifest=artifacts['dataset'], checkpoint=other,
        metrics_artifact=artifacts['metrics'], split='validation', sample_count=1,
        metric_definitions={'precision': 'definition'}, metrics={'precision': 0.5},
        thresholds={'confidence': 0.001}, config={}, implementation_version='8.4.145')
    with pytest.raises(ValueError, match='does not consume the trained model'):
        assemble_fixture(artifacts, tmp_path / 'package')


def test_package_rejects_mismatched_metrics_bytes(tmp_path):
    artifacts = build_artifacts(tmp_path)
    tampered = tmp_path / 'tampered.json'
    write_json(tampered, {'overall': {'precision': 0.99}})
    with pytest.raises(ValueError, match='does not declare evaluation/metrics.json'):
        assemble_fixture(artifacts, tmp_path / 'package', metrics_artifact=tampered)


def test_package_requires_retained_store_for_intermediates(tmp_path):
    artifacts = build_artifacts(tmp_path)
    with pytest.raises(ValueError, match='package audit failed'):
        assemble_fixture(artifacts, tmp_path / 'package', store_roots=())
