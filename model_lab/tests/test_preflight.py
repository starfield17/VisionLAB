"""Preflight distribution checks and the training gate."""
from __future__ import annotations

import json
from pathlib import Path

from model_lab import preflight
from model_lab.__main__ import main
from model_lab.tests.helpers import ONTOLOGY, make_dataset, write_json, png


def blocking_rules(issues):
    return {issue['rule'] for issue in issues if issue['severity'] == 'blocking'}


def warning_rules(issues):
    return {issue['rule'] for issue in issues if issue['severity'] == 'warning'}


def test_healthy_dataset_has_no_blocking_findings(tmp_path):
    path = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'train', 'objects': [{'class_id': 'metal', 'bbox': [2, 2, 9, 9]}]},
        {'split': 'validation', 'objects': [{'class_id': 'plastic', 'bbox': [1, 2, 7, 9]}]},
        {'split': 'validation', 'objects': []},
    ])
    report = preflight.analyze(path)
    issues = preflight.findings(report, min_train_instances=1, min_train_images=1)
    assert blocking_rules(issues) == set()
    assert report['splits']['train']['objects'] == 2
    assert report['splits']['validation']['negative_images'] == 1


def test_all_negative_train_split_is_blocking(tmp_path):
    path = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': []},
        {'split': 'train', 'objects': []},
        {'split': 'validation', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
    ])
    issues = preflight.findings(preflight.analyze(path))
    assert 'train_without_objects' in blocking_rules(issues)
    assert 'class_without_train_instances' in blocking_rules(issues)


def test_validation_only_class_is_blocking(tmp_path):
    path = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'validation', 'objects': [{'class_id': 'metal', 'bbox': [1, 1, 8, 8]}]},
    ])
    issues = preflight.findings(preflight.analyze(path))
    assert 'class_without_train_instances' in blocking_rules(issues)
    assert 'class_not_evaluated' in warning_rules(issues)


def test_unused_class_and_missing_negatives_are_warnings(tmp_path):
    path = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'validation', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
    ])
    issues = preflight.findings(preflight.analyze(path), min_train_instances=1, min_train_images=1)
    rules = warning_rules(issues)
    assert 'unused_class' in rules
    assert 'no_negative_train_images' in rules


def test_small_object_dominance_warns(tmp_path):
    path = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'size': (64, 64), 'objects': [
            {'class_id': 'plastic', 'bbox': [1, 1, 5, 5]},
            {'class_id': 'metal', 'bbox': [10, 10, 14, 14]}]},
        {'split': 'validation', 'size': (64, 64), 'objects': [
            {'class_id': 'metal', 'bbox': [2, 2, 6, 6]}]},
    ])
    issues = preflight.findings(preflight.analyze(path), min_train_images=1)
    assert 'small_object_dominant' in warning_rules(issues)


def test_dominant_class_warns(tmp_path):
    boxes = [{'class_id': 'plastic', 'bbox': [2, 2, 40, 40]} for _ in range(19)]
    boxes.append({'class_id': 'metal', 'bbox': [3, 3, 40, 40]})
    items = [{'split': 'train', 'size': (64, 64), 'objects': boxes}]
    items.append({'split': 'validation', 'size': (64, 64), 'objects': [
        {'class_id': 'metal', 'bbox': [2, 2, 40, 40]}]})
    issues = preflight.findings(preflight.analyze(make_dataset(tmp_path / 'dataset', items)),
                                min_train_images=1)
    assert 'dominant_class' in warning_rules(issues)


def test_training_gate_blocks_and_records_override(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': []},
        {'split': 'validation', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
    ])
    args = ['train', '--project', 'workdir/training', '--name', 'gate-run',
            '--cfg', 'model_lab/configs/smoke.yaml', '--model', 'workdir/models/yolo26n.pt',
            '--data', str(dataset.parent / 'data.yaml'), '--guard-output', str(tmp_path / 'guard'),
            '--dataset', str(dataset)]
    write_json(dataset.parent / 'data.yaml', {'train': 'images/train', 'val': 'images/val',
                                              'names': ['plastic', 'metal']})
    assert main(args) == 2
    assert not (tmp_path / 'guard').exists()
    calls = []
    monkeypatch.setattr('model_lab.__main__.run', lambda *a, **k: calls.append(a) or 0)
    assert main([*args, '--override-blocking', 'integration exercise only']) == 0
    report = json.loads((tmp_path / 'guard-preflight.json').read_text())
    assert report['override_reason'] == 'integration exercise only'
    assert calls and calls[0][0][0] == 'yolo'


def test_cli_preflight_json_exit_code(tmp_path, capsys):
    dataset = make_dataset(tmp_path / 'dataset', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'validation', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
    ])
    report_path = tmp_path / 'report.json'
    assert main(['preflight', '--dataset', str(dataset), '--json',
                 '--report', str(report_path)]) == 0
    document = json.loads(report_path.read_text())
    assert document['report']['dataset']['id'] == 'fixture-set'
    assert 'fixture-set' in capsys.readouterr().out


def test_preflight_reads_annotations_through_store_roots(tmp_path):
    dataset = make_dataset(tmp_path / 'store', [
        {'split': 'train', 'objects': [{'class_id': 'plastic', 'bbox': [1, 1, 8, 8]}]},
        {'split': 'validation', 'objects': []},
    ])
    staged = tmp_path / 'package' / 'config'
    staged.mkdir(parents=True)
    (staged / 'dataset.json').write_bytes(dataset.read_bytes())
    report = preflight.analyze(staged / 'dataset.json', store_roots=(dataset.parent,))
    assert report['splits']['train']['objects'] == 1
    assert report['ontology']['id'] == ONTOLOGY['id']


def test_analyze_rejects_an_invalid_manifest(tmp_path):
    path = tmp_path / 'dataset.json'
    write_json(path, {'schema_version': '1.0.0', 'id': 'x', 'version': '1',
                      'created_at': '2026-01-01T00:00:00Z', 'ontology': ONTOLOGY, 'items': []})
    try:
        preflight.analyze(path)
    except ValueError as exc:
        assert 'invalid' in str(exc)
    else:  # pragma: no cover - the manifest above must not be accepted
        raise AssertionError('empty item lists must be rejected')


def test_png_helper_writes_real_images(tmp_path):
    path = png(Path(tmp_path) / 'image.png')
    assert path.is_file() and path.stat().st_size > 0
