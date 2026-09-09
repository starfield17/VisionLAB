"""Cross-entrypoint, corrupted-input and retained-store scenarios."""
import copy
import json
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from contractcheck.cli import main, run_validate
from contractcheck.dataset import DatasetValidator
from contractcheck.detect_event import DetectionEventValidator
from contractcheck.errors import ValidationResult
from contractcheck.loader import loads_strict
from contractcheck.package import PackageValidator
from conftest import read_json, rehash_annotation, rehash_package, rules_of, sha256, write_json
from test_regressions import registry


def rehash_chain(pkg):
    ds = pkg / 'data/dataset.json'
    rehash_annotation(ds, pkg / 'data/annotation.json')
    for stage in ('training', 'evaluation', 'export'):
        p = pkg / 'records' / f'{stage}.json'
        doc = read_json(p)
        for item in doc['inputs']:
            if item['role'] == 'dataset':
                item['artifact']['sha256'] = sha256(ds)
        write_json(p, doc)
    rehash_package(pkg)


@pytest.mark.parametrize('change,rule', [('class', 'unknown_class'), ('bbox', 'box_out_of_bounds'),
                                        ('candidate', 'annotation_not_approved'), ('identity', 'annotation_mismatch')])
def test_audit_and_dataset_share_semantics(store, valid_package, change, rule):
    p = valid_package / 'data/annotation.json'
    ann = read_json(p)
    if change == 'class':
        ann['objects'][0]['class_id'] = 'missing'
    elif change == 'bbox':
        ann['objects'][0]['bbox'][2] = 999
    elif change == 'identity':
        ann['id'] = 'wrong'
    else:
        ann['status'] = 'candidate'
    write_json(p, ann)
    rehash_chain(valid_package)
    standalone = DatasetValidator(store).validate(valid_package / 'data/dataset.json')
    audit = PackageValidator(store, registry()).validate(valid_package, 'audit')
    assert rule in rules_of(standalone)
    assert rule in rules_of(audit)


@pytest.mark.parametrize('docname', ['manifest.json', 'labels.json', 'preprocessing.json', 'data/dataset.json', 'records/export.json'])
def test_structural_corruption_is_error(store, valid_package, docname):
    path = valid_package / docname
    original = read_json(path)
    for key in original:
        malformed = copy.deepcopy(original)
        malformed[key] = None
        write_json(path, malformed)
        rehash_package(valid_package) if docname != 'manifest.json' else None
        assert not PackageValidator(store, registry()).validate(valid_package, 'audit').ok
    write_json(path, original)


def test_invalid_utf8(store, valid_dataset):
    valid_dataset.write_bytes(b'\xff\xfe')
    assert rules_of(DatasetValidator(store).validate(valid_dataset)) == {'strict_json'}
    assert main(['validate', str(valid_dataset)]) == 1


def test_symlink_cycle_returns_error(store, valid_package):
    path = valid_package / 'model.fixture'
    path.unlink()
    path.symlink_to('model.fixture')
    assert not PackageValidator(store, registry()).validate(valid_package).ok


def test_utc_checked_on_dataset_package_and_acquisition(store, valid_package):
    for name, key in [('manifest.json', 'created_at'), ('data/dataset.json', 'created_at')]:
        path = valid_package / name
        doc = read_json(path)
        doc[key] = '2026-01-01T00:00:00+00:00'
        write_json(path, doc)
        if name != 'manifest.json':
            rehash_package(valid_package)
        assert 'utc_timestamp' in rules_of(PackageValidator(store, registry()).validate(valid_package))


def test_event_package_context(store, valid_package, tmp_path):
    package = PackageValidator(store, registry()).load(valid_package)
    event = read_json(Path('contracts/examples/detection-event.json'))
    event['model']['manifest_sha256'] = package.manifest_sha256
    validator = DetectionEventValidator(store)
    assert validator.validate_document(event, package=package).ok
    event['detections'][0]['label'] = 'Wrong'
    assert 'unknown_class' in rules_of(validator.validate_document(event, package=package))
    event['model']['package_id'] = 'wrong'
    assert 'model_mismatch' in rules_of(validator.validate_document(event, package=package))


def test_external_dataset_content_is_semantically_checked(store, valid_package, tmp_path):
    retained = tmp_path / 'store'
    retained.mkdir()
    for name in ('annotation.json', 'image.fixture'):
        (valid_package / 'data' / name).rename(retained / name)
    assert PackageValidator(store, registry()).validate(valid_package).ok
    assert not PackageValidator(store, registry()).validate(valid_package, 'audit').ok
    assert PackageValidator(store, registry()).validate(valid_package, 'audit', (retained,)).ok
    annotation = retained / 'annotation.json'
    doc = read_json(annotation)
    doc['status'] = 'candidate'
    write_json(annotation, doc)
    ds_path = valid_package / 'data/dataset.json'
    rehash_annotation(ds_path, annotation)
    for stage in ('training', 'evaluation', 'export'):
        path = valid_package / 'records' / f'{stage}.json'
        doc = read_json(path)
        for ref in doc['inputs']:
            if ref['role'] == 'dataset':
                ref['artifact']['sha256'] = sha256(ds_path)
        write_json(path, doc)
    rehash_package(valid_package)
    assert 'annotation_not_approved' in rules_of(PackageValidator(store, registry()).validate(valid_package, 'audit', (retained,)))


def test_revision_history_missing_and_cycle(store, valid_dataset):
    p = valid_dataset.parent / 'annotation.json'
    ann = read_json(p)
    ann['supersedes'] = 'previous'
    write_json(p, ann)
    rehash_annotation(valid_dataset, p)
    assert 'audit_incomplete' in rules_of(DatasetValidator(store).validate(valid_dataset))
    previous = copy.deepcopy(ann)
    previous.update(id='previous', supersedes=ann['id'])
    write_json(p.parent / 'previous.json', previous)
    assert 'annotation_cycle' in rules_of(DatasetValidator(store).validate(valid_dataset))


def test_broken_run_lineage(store, valid_package):
    path = valid_package / 'records/evaluation.json'
    doc = read_json(path)
    doc['inputs'] = [ref for ref in doc['inputs'] if ref['role'] != 'model']
    write_json(path, doc)
    rehash_package(valid_package)
    assert 'lineage_mismatch' in rules_of(PackageValidator(store, registry()).validate(valid_package, 'audit'))


def test_illustrative_package_is_not_full_audit(store):
    result = PackageValidator(store, registry()).validate(Path('contracts/examples/model-package'), 'audit')
    assert {'lineage_mismatch', 'evaluation_config'} <= rules_of(result)


def test_empty_schema_store_is_configuration_error(tmp_path):
    assert main(['validate', 'irrelevant', '--schemas', str(tmp_path)]) == 2


def test_schema_error_json_pointer_escape(store):
    store.schemas['temporary'] = {'type': 'object', 'properties': {'a/b~c': {'type': 'integer'}}}
    result = ValidationResult()
    assert not store.check({'a/b~c': 'bad'}, 'temporary', 'doc', result)
    assert result.errors[0].field == '/a~1b~0c'
    del store.schemas['temporary']


@given(st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
                   lambda child: st.lists(child, max_size=5) | st.dictionaries(st.text(), child, max_size=5), max_leaves=20))
def test_strict_json_roundtrip(value):
    assert loads_strict(json.dumps(value)) == value


def test_standalone_inverted_annotation(store, tmp_path):
    path = tmp_path / 'annotation.json'
    ann = read_json(Path('contracts/examples/candidate-annotation.json'))
    ann['objects'][0]['bbox'] = [10, 0, 1, 2]
    write_json(path, ann)
    assert 'box_inverted' in rules_of(run_validate(store, path))


@pytest.mark.parametrize('keyword', ['$ref', '$dynamicRef'])
def test_adapter_schema_cannot_fetch_remote(keyword):
    from contractcheck.package import AdapterRegistry
    schema = {'type': 'object', 'properties': {'option': {keyword: 'https://invalid.example/schema'}},
              'additionalProperties': False}
    with pytest.raises(ValueError, match='internal'):
        AdapterRegistry().register('example', '1', schema, formats=('example',))


def test_adapter_schema_invalid_local_reference_is_configuration_error():
    from contractcheck.package import AdapterRegistry
    schema = {'type': 'object', 'properties': {'option': {'$ref': '#/$defs/missing'}},
              'additionalProperties': False}
    with pytest.raises(ValueError, match='resolved'):
        AdapterRegistry().register('example', '1', schema, formats=('example',))


def test_programmatic_event_rejects_nonfinite_score(store):
    event = read_json(Path('contracts/examples/detection-event.json'))
    event['detections'][0]['confidence'] = float('nan')
    assert 'non_finite' in rules_of(DetectionEventValidator(store).validate_document(event))


def test_invalid_schema_reference_is_configuration_error(tmp_path):
    folder = tmp_path / 'schemas'
    folder.mkdir()
    (folder / 'bad.schema.json').write_text(json.dumps({
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        '$id': 'bad.schema.json', '$ref': '#/missing',
    }))
    assert main(['validate', 'irrelevant', '--schemas', str(folder)]) == 2


def test_audit_duplicate_group_split(store, valid_package):
    path = valid_package / 'data/dataset.json'
    ds = read_json(path)
    duplicate = copy.deepcopy(ds['items'][0])
    duplicate.update(id='item-2', split='train')
    ds['items'].append(duplicate)
    write_json(path, ds)
    rehash_package(valid_package)
    assert 'split_leakage' in rules_of(PackageValidator(store, registry()).validate(valid_package, 'audit'))
