import copy
import hashlib
import json
from pathlib import Path

import pytest

from contractcheck.dataset import DatasetValidator
from contractcheck.loader import SchemaStore
from model_lab.conversion import approve, build_dataset, candidate, export_yolo


ONTOLOGY = {'id': 'people', 'version': '1', 'classes': [{'id': 'person', 'label': 'person'}]}
SOURCE = {'kind': 'locate_anything', 'actor_id': 'label-worker', 'run_id': 'label-run', 'tool_version': 'pinned-cpp'}


def make_candidate(raw):
    return candidate(raw, annotation_id='candidate-1', item_id='image-1', image_hash='a' * 64,
                     width=100, height=80, ontology=ONTOLOGY, source=SOURCE, created_at='2026-01-01T00:00:00Z')


def test_candidate_preserves_pixels_and_omits_confidence():
    doc = make_candidate({'detections': [{'label': 'person', 'box': [10, 20, 50, 60]}]})
    assert doc['status'] == 'candidate'
    assert doc['objects'] == [{'id': 'object-0000', 'class_id': 'person', 'bbox': [10, 20, 50, 60]}]
    assert doc['source'] == SOURCE


@pytest.mark.parametrize('entry', [{'label': 'unknown', 'box': [0, 0, 1, 1]},
    {'label': 'person', 'box': [20, 0, 1, 1]}, {'label': 'person', 'box': [0, 0, 101, 80]},
    {'label': 'person', 'box': [0, 0, float('nan'), 80]}])
def test_invalid_candidates_are_not_silently_dropped(entry):
    with pytest.raises(ValueError):
        make_candidate({'detections': [entry]})


def test_empty_response_is_still_candidate():
    doc = make_candidate({'detections': []})
    assert doc['objects'] == []
    assert doc['status'] == 'candidate'


def write(path, doc):
    path.write_text(json.dumps(doc))


def png_bytes(value):
    import struct
    import zlib
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + bytes([value] * 6)) * 2)) + chunk(b'IEND', b''))


def item_spec(image, i, split, tmp_path):
    cand = candidate({'detections': [{'label': 'person', 'box': [0, 0, 1, 2]}] if i % 2 == 0 else []},
                     annotation_id=f'candidate-{i}', item_id=f'item-{i}',
                     image_hash=artifact(image)['sha256'], width=2, height=2,
                     ontology=ONTOLOGY, source=SOURCE, created_at='2026-01-01T00:00:00Z')
    approved = approve(cand, approval_id=f'approved-{i}', reviewer_id='human-1',
                       reviewed_at='2026-01-01T00:00:00Z', reason='Synthetic test review')
    return {'id': f'item-{i}', 'image': image, 'width': 2, 'height': 2, 'group_id': f'group-{i}',
            'split': split, 'provenance': {'source_ref': f'test:{i}', 'acquired_at': '2026-01-01T00:00:00Z'},
            'annotation': cand, 'approved': approved}


def artifact(path):
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def dataset(tmp_path):
    # Real tiny RGB PNGs, distinct hashes. Decoder tests belong to acquisition.
    import struct
    import zlib
    def png(value):
        def chunk(kind, payload):
            return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
        return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress((b'\0' + bytes([value]*6))*2)) + chunk(b'IEND', b'')
    items = []
    for i, split in enumerate(('train', 'validation')):
        image = tmp_path / f'{i}.png'
        image.write_bytes(png(i))
        ann = candidate({'detections': [{'label': 'person', 'box': [0, 0, 1, 2]}] if i == 0 else []},
                        annotation_id=f'candidate-{i}', item_id=f'item-{i}', image_hash=artifact(image)['sha256'],
                        width=2, height=2, ontology=ONTOLOGY, source=SOURCE, created_at='2026-01-01T00:00:00Z')
        write(tmp_path / f'candidate-{i}.json', ann)
        approved = copy.deepcopy(ann)
        approved.update(id=f'approved-{i}', supersedes=ann['id'], status='approved',
                        review={'reviewer_id': 'test-human', 'reviewed_at': '2026-01-01T00:00:00Z', 'reason': 'Synthetic test review'})
        ann_path = tmp_path / f'approved-{i}.json'
        write(ann_path, approved)
        items.append({'id': f'item-{i}', 'image': artifact(image), 'width': 2, 'height': 2,
                      'group_id': f'group-{i}', 'split': split,
                      'provenance': {'source_ref': f'test:{i}', 'acquired_at': '2026-01-01T00:00:00Z'},
                      'annotation': {'id': approved['id'], 'artifact': artifact(ann_path)}})
    path = tmp_path / 'dataset.json'
    write(path, {'schema_version': '1.0.0', 'id': 'test', 'version': '1', 'created_at': '2026-01-01T00:00:00Z',
                 'ontology': ONTOLOGY, 'items': items})
    return path


def test_export_normalization_mapping_negative_and_no_absolute_paths(dataset, tmp_path):
    out = tmp_path / 'output'
    export_yolo(dataset, out)
    assert (out / 'labels/train/image-000000.txt').read_text() == '0 0.25 0.5 0.5 1\n'
    assert (out / 'labels/val/image-000001.txt').read_text() == ''
    mapping = json.loads((out / 'mapping.json').read_text())
    assert mapping['classes'] == [{'index': 0, 'class_id': 'person', 'label': 'person'}]
    assert json.loads((out / 'data.yaml').read_text()) == {'train': 'images/train', 'val': 'images/val', 'names': ['person']}
    assert str(tmp_path) not in (out / 'mapping.json').read_text()
    with pytest.raises(FileExistsError):
        export_yolo(dataset, out)


def test_export_rejects_candidate(dataset, tmp_path):
    ds = json.loads(dataset.read_text())
    ann = tmp_path / 'candidate-0.json'
    ds['items'][0]['annotation'] = {'id': 'candidate-0', 'artifact': artifact(ann)}
    write(dataset, ds)
    with pytest.raises(ValueError, match='annotation_not_approved'):
        export_yolo(dataset, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_export_requires_validation_split(dataset, tmp_path):
    ds = json.loads(dataset.read_text())
    ds['items'][1]['split'] = 'train'
    write(dataset, ds)
    with pytest.raises(ValueError, match='validation'):
        export_yolo(dataset, tmp_path / 'output')


def test_approve_creates_immutable_revision():
    cand = make_candidate({'detections': [{'label': 'person', 'box': [0, 0, 1, 2]}]})
    approved = approve(cand, approval_id='approved-1', reviewer_id='human-1',
                       reviewed_at='2026-01-01T00:00:00Z', reason='verified')
    assert approved['id'] == 'approved-1'
    assert approved['supersedes'] == cand['id']
    assert approved['status'] == 'approved'
    assert approved['source'] == SOURCE
    assert approved['review']['reason'] == 'verified'
    with pytest.raises(ValueError):
        approve(approved, approval_id='x', reviewer_id='r', reviewed_at='2026-01-01T00:00:00Z', reason='no')


def test_approve_rejects_review_before_creation():
    cand = make_candidate({'detections': []})
    with pytest.raises(ValueError, match='time_order'):
        approve(cand, approval_id='approved-1', reviewer_id='human-1',
                reviewed_at='2025-12-31T00:00:00Z', reason='late')


def test_build_dataset_publishes_validated_snapshot(tmp_path):
    image0, image1 = tmp_path / '0.png', tmp_path / '1.png'
    image0.write_bytes(png_bytes(10))
    image1.write_bytes(png_bytes(20))
    items = [item_spec(image0, 0, 'train', tmp_path), item_spec(image1, 1, 'validation', tmp_path)]
    dest = tmp_path / 'dataset'
    build_dataset(dest, dataset_id='people', version='1', created_at='2026-01-01T00:00:00Z',
                  ontology=ONTOLOGY, items=items)
    manifest = json.loads((dest / 'dataset.json').read_text())
    assert manifest['items'][0]['annotation']['id'] == 'approved-0'
    assert (dest / manifest['items'][0]['image']['path']).is_file()
    assert (dest / manifest['items'][1]['annotation']['artifact']['path']).is_file()
    assert DatasetValidator(SchemaStore()).validate(dest / 'dataset.json').ok
    with pytest.raises(FileExistsError):
        build_dataset(dest, dataset_id='people', version='1', created_at='2026-01-01T00:00:00Z',
                      ontology=ONTOLOGY, items=items)


def test_build_dataset_rejects_unknown_split(tmp_path):
    image = tmp_path / '0.png'
    image.write_bytes(png_bytes(10))
    spec = item_spec(image, 0, 'train', tmp_path)
    spec['split'] = 'trainning'
    with pytest.raises(ValueError, match='split'):
        build_dataset(tmp_path / 'dataset', dataset_id='d', version='1',
                      created_at='2026-01-01T00:00:00Z', ontology=ONTOLOGY, items=[spec])


def test_low_memory_configs():
    for name, batch, size, epochs in [('smoke', 1, 320, 1), ('initial', 2, 416, 30)]:
        config = json.loads(Path(f'model_lab/configs/{name}.yaml').read_text())
        assert (config['batch'], config['imgsz'], config['epochs']) == (batch, size, epochs)
        assert config['model'] == 'yolo26n.pt'
        assert config['cache'] is False and config['workers'] == 0
        assert config['amp'] is False and config['compile'] is False
        assert config['val'] is True and config['mosaic'] == 0
        assert 'device' not in config and 'data' not in config
