import pytest
from PIL import Image

from model_lab.overlay import render_overlay

ONTOLOGY = {'id': 'o', 'version': '1', 'classes': [{'id': 'person', 'label': 'Person'}, {'id': 'dog', 'label': 'Dog'}]}


def annotation(objects):
    return {'schema_version': '1.0.0', 'id': 'a', 'item_id': 'i', 'image_sha256': 'a' * 64,
            'ontology_id': 'o', 'ontology_version': '1', 'created_at': '2026-01-01T00:00:00Z',
            'status': 'candidate', 'source': {'kind': 'locate_anything', 'actor_id': 'w', 'run_id': 'r', 'tool_version': 'v'},
            'objects': objects}


def test_overlay_draws_boxes_and_labels(tmp_path):
    image = tmp_path / 'img.png'
    Image.new('RGB', (100, 80), (255, 0, 0)).save(image)
    objects = [{'id': 'o1', 'class_id': 'person', 'bbox': [10, 10, 50, 60]},
               {'id': 'o2', 'class_id': 'dog', 'bbox': [60, 10, 90, 40]}]
    out = tmp_path / 'overlay.png'
    render_overlay(image, annotation(objects), ONTOLOGY, out)
    overlay = Image.open(out)
    assert overlay.size == (100, 80)
    assert overlay.getpixel((10, 10)) != (255, 0, 0)


def test_overlay_empty_annotation_is_negative_review(tmp_path):
    image = tmp_path / 'img.png'
    Image.new('RGB', (40, 30), (0, 255, 0)).save(image)
    out = tmp_path / 'negative.png'
    render_overlay(image, annotation([]), ONTOLOGY, out)
    overlay = Image.open(out)
    assert overlay.size == (40, 30)
    assert overlay.getpixel((20, 15)) == (0, 255, 0)


def test_overlay_output_immutable_and_colors_stable(tmp_path):
    image = tmp_path / 'img.png'
    Image.new('RGB', (100, 80), (255, 255, 255)).save(image)
    objects = [{'id': 'o1', 'class_id': 'person', 'bbox': [10, 10, 50, 60]}]
    render_overlay(image, annotation(objects), ONTOLOGY, tmp_path / 'a.png')
    render_overlay(image, annotation(objects), ONTOLOGY, tmp_path / 'b.png')
    assert (tmp_path / 'a.png').read_bytes() == (tmp_path / 'b.png').read_bytes()
    with pytest.raises(FileExistsError):
        render_overlay(image, annotation(objects), ONTOLOGY, tmp_path / 'a.png')


@pytest.mark.parametrize('objects', [
    [{'id': 'o1', 'class_id': 'alien', 'bbox': [0, 0, 1, 1]}],
    [{'id': 'o1', 'class_id': 'person', 'bbox': [20, 0, 1, 1]}],
    [{'id': 'o1', 'class_id': 'person', 'bbox': [0, 0, 101, 80]}],
])
def test_overlay_rejects_invalid_objects(tmp_path, objects):
    image = tmp_path / 'img.png'
    Image.new('RGB', (100, 80), (0, 0, 0)).save(image)
    with pytest.raises(ValueError):
        render_overlay(image, annotation(objects), ONTOLOGY, tmp_path / 'o.png')
