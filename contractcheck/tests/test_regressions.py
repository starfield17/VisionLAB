"""Review regressions: invalid inputs must fail safely, valid inputs must pass."""
import json

import pytest

from contractcheck.dataset import DatasetValidator
from contractcheck.detect_event import DetectionEventValidator
from contractcheck.loader import StrictJSONError, loads_strict
from contractcheck.package import AdapterRegistry, PackageValidator
from conftest import read_json, rehash_annotation, rehash_package, write_json


def registry():
    result = AdapterRegistry()
    result.register('fixture', '1', formats=('fixture',), config_schema={
        'type': 'object', 'properties': {}, 'additionalProperties': False,
    })
    return result


@pytest.mark.parametrize('blob', ['1e999', '-1e999', '{"config":{"x":1e999}}'])
def test_overflow_is_not_json(blob):
    with pytest.raises(StrictJSONError):
        loads_strict(blob)


def test_missing_hash_returns_error(store, valid_dataset):
    ds = read_json(valid_dataset)
    del ds['items'][0]['image']['sha256']
    write_json(valid_dataset, ds)
    result = DatasetValidator(store).validate(valid_dataset)
    assert not result.ok
    assert any(e.rule == 'schema' for e in result.errors)


@pytest.mark.parametrize('captured,emitted,ok', [
    ('2026-01-01T00:00:00Z', '2026-01-01T00:00:00.1Z', True),
    ('2026-01-01T00:00:00.1Z', '2026-01-01T00:00:00Z', False),
    ('2026-01-01T00:00:00.10Z', '2026-01-01T00:00:00.1Z', True),
    ('2026-01-01T00:00:00.00000002Z', '2026-01-01T00:00:00.00000001Z', False),
])
def test_fractional_time_order(store, tmp_path, captured, emitted, ok):
    event = read_json(__import__('pathlib').Path('contracts/examples/empty-event.json'))
    event.update(captured_at=captured, emitted_at=emitted)
    path = tmp_path / 'event.json'
    write_json(path, event)
    assert DetectionEventValidator(store).validate(path).ok == ok


def test_audit_rejects_candidate_even_with_correct_hashes(store, valid_package):
    path = valid_package / 'data/annotation.json'
    ann = read_json(path)
    ann['status'] = 'candidate'
    ann.pop('review')
    write_json(path, ann)
    rehash_annotation(valid_package / 'data/dataset.json', path)
    rehash_package(valid_package)
    result = PackageValidator(store, registry()).validate(valid_package, 'audit')
    assert any(e.rule == 'annotation_not_approved' for e in result.errors)


def test_label_text_must_match(store, valid_package):
    path = valid_package / 'labels.json'
    labels = read_json(path)
    labels['classes'][0]['label'] = 'Wrong'
    write_json(path, labels)
    rehash_package(valid_package)
    result = PackageValidator(store, registry()).validate(valid_package)
    assert any(e.rule == 'labels_ontology_mismatch' for e in result.errors)


@pytest.mark.parametrize('change,rule', [('format', 'unsupported_format'), ('config', 'unsupported_adapter_config')])
def test_adapter_contract(store, valid_package, change, rule):
    path = valid_package / 'manifest.json'
    manifest = read_json(path)
    if change == 'format':
        manifest['model']['format'] = 'unknown'
    else:
        manifest['model']['adapter']['config'] = {'unknown': True}
    write_json(path, manifest)
    result = PackageValidator(store, registry()).validate(valid_package)
    assert any(e.rule == rule for e in result.errors)


def test_malformed_bbox_is_error_not_exception(store, tmp_path):
    from pathlib import Path
    event = read_json(Path('contracts/examples/detection-event.json'))
    event['detections'][0]['bbox'] = ['bad', 0, 1, 2]
    path = tmp_path / 'event.json'
    path.write_text(json.dumps(event))
    assert not DetectionEventValidator(store).validate(path).ok
