"""Public COCO ingestion: curation, grouping, splits, provenance."""
from __future__ import annotations

import json
import shutil

import pytest
from PIL import Image, ImageDraw

from contractcheck.dataset import DatasetValidator
from contractcheck.loader import SchemaStore

from model_lab.ingest import assign_splits, ingest_coco, load_ontology_config
from model_lab.tests.helpers import write_json

ONTOLOGY_CONFIG = {
    'ontology': {'id': 'fixture-litter', 'version': '1',
                 'classes': [{'id': 'plastic', 'label': 'plastic'},
                             {'id': 'metal', 'label': 'metal'}]},
    'categories': {'Film': 'plastic', 'Can': 'metal'},
    'ignore': ['Unlabelled'],
}


def _pattern(path, kind, size=(40, 30)):
    image = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    if kind == 'vertical':
        draw.rectangle([0, 0, size[0] // 2, size[1]], fill=(255, 255, 255))
    elif kind == 'vertical-shifted':
        draw.rectangle([0, 0, size[0] // 2 + 1, size[1]], fill=(255, 255, 255))
    elif kind == 'horizontal':
        draw.rectangle([0, 0, size[0], size[1] // 2], fill=(255, 255, 255))
    else:
        for x in range(0, size[0], 4):
            draw.line([(x, 0), (x, size[1])], fill=(120, 120, 120))
    image.save(path)


def build_source(tmp_path, *, duplicates=False, exif=False):
    images = tmp_path / 'source'
    images.mkdir(parents=True)
    entries, annotations = [], []
    kinds = ['vertical', 'vertical-shifted', 'horizontal', 'diagonal']
    for index in range(4):
        path = images / f'{index:03d}.png'
        if duplicates and index == 1:
            shutil.copyfile(images / '000.png', path)
        else:
            _pattern(path, kinds[index])
        size = (40, 30)
        entries.append({'id': index + 1, 'file_name': path.name, 'width': size[0], 'height': size[1],
                        'license': 'CC', 'flickr_url': f'https://example.invalid/{index}'})
        annotations.append({'id': 100 + index, 'image_id': index + 1, 'category_id': 1,
                            'bbox': [2, 2, 10, 10]})
        if index == 1:
            annotations.append({'id': 200, 'image_id': index + 1, 'category_id': 2,
                                'bbox': [0, 0, 5, 5]})
        if index == 2:
            annotations.append({'id': 201, 'image_id': index + 1, 'category_id': 1,
                                'bbox': [30, 20, 20, 20]})
            annotations.append({'id': 202, 'image_id': index + 1, 'category_id': 3,
                                'bbox': [1, 1, 2, 2]})
    if exif:
        oriented = Image.new('RGB', (30, 20), (5, 5, 5))
        exif_bytes = Image.Exif()
        exif_bytes[274] = 6
        path = images / 'oriented.jpg'
        oriented.save(path, format='JPEG', exif=exif_bytes.tobytes())
        entries.append({'id': 99, 'file_name': path.name, 'width': 20, 'height': 30})
        annotations.append({'id': 300, 'image_id': 99, 'category_id': 1, 'bbox': [1, 1, 4, 4]})
    document = {'images': entries, 'annotations': annotations,
                'categories': [{'id': 1, 'name': 'Film'}, {'id': 2, 'name': 'Can'},
                               {'id': 3, 'name': 'Unlabelled'}]}
    annotations_path = tmp_path / 'annotations.json'
    write_json(annotations_path, document)
    return images, annotations_path


def ingest(tmp_path, *, duplicates=False, exif=False, max_side=64):
    images, annotations = build_source(tmp_path, duplicates=duplicates, exif=exif)
    config = tmp_path / 'ontology.json'
    write_json(config, ONTOLOGY_CONFIG)
    destination = tmp_path / 'dataset'
    report = ingest_coco(images_dir=images, annotations_path=annotations, ontology_config=config,
                         destination=destination, dataset_id='fixture-litter', version='1',
                         actor_id='fixture-dataset@1', run_id='zenodo-fixture', tool_version='ingest-1',
                         acquired_at='2026-01-01T00:00:00Z', created_at='2026-01-02T00:00:00Z',
                         longest_side=max_side, seed=7)
    return destination, report


def test_ingest_publishes_validated_dataset_with_scaled_boxes(tmp_path):
    destination, report = ingest(tmp_path)
    result = DatasetValidator(SchemaStore()).validate(destination / 'dataset.json')
    assert result.ok, result.errors
    assert report['counts']['items'] == 4
    assert report['counts']['objects'] == 5
    assert report['counts']['objects_dropped'] == {'unrepresentable_box': 1, 'ignored_category': 1}
    manifest = json.loads((destination / 'dataset.json').read_text())
    assert {item['split'] for item in manifest['items']} == {'train', 'validation'}


def test_ingest_records_attribution_and_review_provenance(tmp_path):
    destination, _ = ingest(tmp_path)
    attribution = json.loads((destination / 'attribution.json').read_text())
    assert {entry['license'] for entry in attribution} == {'CC'}
    annotation = json.loads(next((destination / 'annotations').iterdir()).read_text())
    assert annotation['status'] == 'approved'
    assert annotation['source']['actor_id'] == 'fixture-dataset@1'
    assert 'not independently' in annotation['review']['reason']
    assert annotation['review']['reviewed_at'] >= annotation['created_at']


def test_ingest_keeps_every_ontology_class_in_train(tmp_path):
    destination, report = ingest(tmp_path)
    assert set(report['splits']['train']['classes']) == {'plastic', 'metal'}
    assert report['splits']['validation']['images'] >= 1


def test_exact_duplicate_bytes_are_dropped_and_counted(tmp_path):
    destination, report = ingest(tmp_path, duplicates=True)
    assert report['counts']['images_skipped'] == {'duplicate_image': 1}
    assert report['counts']['items'] == 3


def test_near_duplicate_frames_share_a_group_and_a_split(tmp_path):
    destination, report = ingest(tmp_path)
    manifest = json.loads((destination / 'dataset.json').read_text())
    assert report['groups']['multi_image'] == 1
    by_group = {}
    for item in manifest['items']:
        by_group.setdefault(item['group_id'], []).append(item)
    grouped = [items for items in by_group.values() if len(items) > 1]
    assert len(grouped) == 1
    assert len({item['split'] for item in grouped[0]}) == 1


def test_ingest_is_deterministic_for_a_fixed_seed(tmp_path):
    first, _ = ingest(tmp_path / 'first')
    second, _ = ingest(tmp_path / 'second')
    left = json.loads((first / 'dataset.json').read_text())
    right = json.loads((second / 'dataset.json').read_text())
    assert [item['split'] for item in left['items']] == [item['split'] for item in right['items']]


def test_exif_oriented_images_are_skipped_not_mislabelled(tmp_path):
    destination, report = ingest(tmp_path, exif=True)
    assert report['counts']['images_skipped'] == {'exif_orientation': 1}
    manifest = json.loads((destination / 'dataset.json').read_text())
    assert len(manifest['items']) == 4


def test_ingest_refuses_to_overwrite_and_unmapped_categories(tmp_path):
    destination, _ = ingest(tmp_path)
    images, annotations = build_source(tmp_path / 'again')
    config = tmp_path / 'again' / 'ontology.json'
    document = json.loads(json.dumps(ONTOLOGY_CONFIG))
    document['categories'] = {'Film': 'plastic'}
    write_json(config, document)
    with pytest.raises(ValueError, match='does not classify'):
        ingest_coco(images_dir=images, annotations_path=annotations, ontology_config=config,
                    destination=tmp_path / 'other', dataset_id='x', version='1', actor_id='a',
                    run_id='r', tool_version='t', acquired_at='2026-01-01T00:00:00Z')
    assert (destination / 'dataset.json').is_file()


def test_ontology_config_rejects_unknown_classes(tmp_path):
    config = tmp_path / 'ontology.json'
    document = json.loads(json.dumps(ONTOLOGY_CONFIG))
    document['categories'] = {'Film': 'unknown-class'}
    write_json(config, document)
    with pytest.raises(ValueError, match='unknown ontology classes'):
        load_ontology_config(config)


def test_assign_splits_requires_two_groups():
    items = [{'id': 'a', 'group_id': 'g', 'objects': [{'class_id': 'plastic'}]}]
    with pytest.raises(ValueError, match='two independent groups'):
        assign_splits(items, dataset_id='x', seed=1, train_fraction=0.8)
