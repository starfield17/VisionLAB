"""Training, evaluation and export records: lineage metadata and portability."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.runrecords import (build_evaluation_record, build_export_record,
                                  build_training_record, read_results_csv, validate_run_record)
from model_lab.tests.helpers import make_dataset, sha256_file

RESULTS_HEADER = ('epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),'
                  'metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,'
                  'lr/pg0,lr/pg1,lr/pg2')


def make_run(tmp_path, *, rows=None, metrics=True) -> Path:
    run = tmp_path / 'run'
    (run / 'weights').mkdir(parents=True)
    (run / 'weights' / 'best.pt').write_bytes(b'best-checkpoint')
    (run / 'weights' / 'last.pt').write_bytes(b'last-checkpoint')
    (run / 'args.yaml').write_text('epochs: 3\nimgsz: 64\nbatch: 2\nval: true\n', encoding='utf-8')
    if metrics:
        (run / 'results.csv').write_text('\n'.join([RESULTS_HEADER, *rows]) + '\n', encoding='utf-8')
    return run


def default_rows():
    return ['1,1.0,0.5,0.4,0.3,0.10,0.20,0.30,0.25,0.6,0.5,0.4,0.1,0.1,0.1',
            '2,2.0,0.4,0.3,0.2,0.50,0.60,0.70,0.65,0.5,0.4,0.3,0.1,0.1,0.1']


@pytest.fixture
def dataset(tmp_path) -> Path:
    return make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'validation', 'objects': [{'class_id': 'metal', 'bbox': [1, 1, 8, 8]}]},
    ])


def test_training_record_links_dataset_and_single_model(tmp_path, dataset):
    run = make_run(tmp_path, rows=default_rows())
    pretrained = tmp_path / 'yolo.pt'
    pretrained.write_bytes(b'pretrained')
    record = build_training_record(run_dir=run, dataset_manifest=dataset, record_id='training-x',
                                   profile='fixture', config_file='model_lab/configs/fixture.yaml',
                                   pretrained_ref='workdir/models/yolo.pt', pretrained_path=pretrained,
                                   dataset_yolo_ref='workdir/yolo', implementation_version='8.4.145',
                                   created_at='2026-01-02T00:00:00Z')
    assert record['stage'] == 'training'
    assert [entry['role'] for entry in record['outputs']] == ['model', 'checkpoint', 'results']
    assert record['outputs'][0]['artifact']['sha256'] == sha256_file(run / 'weights' / 'best.pt')
    assert record['inputs'][0]['artifact']['sha256'] == sha256_file(dataset)
    assert (run / 'config' / 'dataset.json').read_bytes() == dataset.read_bytes()
    assert record['metrics']['mAP50-95'] == 0.65
    assert record['metrics']['best_epoch'] == 2.0
    assert json.loads((run / 'training-record.json').read_text())['id'] == 'training-x'


def test_training_record_rejects_incomplete_runs(tmp_path, dataset):
    run = make_run(tmp_path, rows=default_rows(), metrics=False)
    pretrained = tmp_path / 'yolo.pt'
    pretrained.write_bytes(b'pretrained')
    with pytest.raises(ValueError, match='missing results.csv'):
        build_training_record(run_dir=run, dataset_manifest=dataset, record_id='training-x',
                              profile='fixture', config_file='cfg.yaml', pretrained_ref='workdir/models/yolo.pt',
                              pretrained_path=pretrained, dataset_yolo_ref='workdir/yolo',
                              implementation_version='8.4.145')
    (run / 'results.csv').write_text(RESULTS_HEADER + '\n1,1.0,0.5,0.4,0.3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='missing detection metrics'):
        build_training_record(run_dir=run, dataset_manifest=dataset, record_id='training-x',
                              profile='fixture', config_file='cfg.yaml', pretrained_ref='workdir/models/yolo.pt',
                              pretrained_path=pretrained, dataset_yolo_ref='workdir/yolo',
                              implementation_version='8.4.145')


def test_results_csv_requires_finite_numbers(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('epoch,metrics/mAP50(B)\n1,nan\n', encoding='utf-8')
    with pytest.raises(ValueError, match='nonfinite'):
        read_results_csv(path)


def artifact(tmp_path, name, payload=b'bytes') -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_evaluation_record_requires_matching_metric_keys(tmp_path, dataset):
    checkpoint = artifact(tmp_path, 'best.pt')
    metrics = artifact(tmp_path, 'metrics.json', b'{"overall": {}}')
    record = build_evaluation_record(
        record_id='evaluation-x', dataset_manifest=dataset, checkpoint=checkpoint,
        metrics_artifact=metrics, split='validation', sample_count=2,
        metric_definitions={'precision': 'definition'}, metrics={'precision': 0.5},
        thresholds={'confidence': 0.001, 'iou': 0.7}, config={'imgsz': 64},
        implementation_version='8.4.145', created_at='2026-01-03T00:00:00Z')
    assert record['config']['sample_count'] == 2
    assert record['metrics'] == {'precision': 0.5}
    assert record['outputs'][0]['artifact']['path'] == 'evaluation/metrics.json'
    with pytest.raises(ValueError, match='identical nonempty keys'):
        build_evaluation_record(
            record_id='evaluation-y', dataset_manifest=dataset, checkpoint=checkpoint,
            metrics_artifact=metrics, split='validation', sample_count=2,
            metric_definitions={'precision': 'definition', 'recall': 'definition'},
            metrics={'precision': 0.5}, thresholds={'confidence': 0.5}, config={},
            implementation_version='8.4.145')


def test_evaluation_record_rejects_bad_split_and_sample_count(tmp_path, dataset):
    checkpoint = artifact(tmp_path, 'best.pt')
    metrics = artifact(tmp_path, 'metrics.json')
    for overrides in ({'split': 'train'}, {'split': 'validation', 'sample_count': 0}):
        arguments = dict(record_id='evaluation-x', dataset_manifest=dataset, checkpoint=checkpoint,
                         metrics_artifact=metrics, split='validation', sample_count=1,
                         metric_definitions={'precision': 'definition'}, metrics={'precision': 0.5},
                         thresholds={'confidence': 0.5}, config={}, implementation_version='8.4.145')
        arguments.update(overrides)
        with pytest.raises(ValueError):
            build_evaluation_record(**arguments)


def test_export_record_rejects_non_portable_config(tmp_path):
    checkpoint = artifact(tmp_path, 'best.pt')
    model = artifact(tmp_path, 'model.onnx', b'graph')
    probe = artifact(tmp_path, 'probe.json', b'{"checker": "passed"}')
    with pytest.raises(ValueError, match='non_portable_config'):
        build_export_record(record_id='export-x', checkpoint=checkpoint, model_artifact=model,
                            probe_artifact=probe, config={'workdir': '/absolute/path'},
                            implementation_version='8.4.145')
    record = build_export_record(record_id='export-x', checkpoint=checkpoint, model_artifact=model,
                                 probe_artifact=probe, config={'format': 'onnx', 'imgsz': 64},
                                 implementation_version='8.4.145')
    assert record['outputs'][0]['artifact']['sha256'] == sha256_file(model)
    assert record['inputs'][0]['artifact']['sha256'] == sha256_file(checkpoint)


def test_validate_run_record_rejects_non_portable_and_incomplete_documents():
    document = {'schema_version': '1.0.0', 'id': 'x', 'stage': 'training',
                'created_at': '2026-01-01T00:00:00Z', 'implementation': {'name': 'n', 'version': '1',
                                                                        'code_revision': 'pinned'},
                'config': {'weights': '/absolute/best.pt'},
                'inputs': [{'role': 'dataset', 'artifact': {'path': 'a.json', 'sha256': 'a' * 64}}],
                'outputs': [{'role': 'model', 'artifact': {'path': 'm.onnx', 'sha256': 'b' * 64}}]}
    with pytest.raises(ValueError, match='non_portable_config'):
        validate_run_record(document)
    document['config'] = {}
    del document['outputs']
    with pytest.raises(ValueError, match='outputs'):
        validate_run_record(document)
