"""Fixture builders shared by Model Lab tests (not a test module)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ONTOLOGY = {
    'id': 'fixture-litter', 'version': '1',
    'classes': [{'id': 'plastic', 'label': 'plastic'}, {'id': 'metal', 'label': 'metal'}],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, document) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')


def png(path: Path, color=(20, 40, 60), size=(16, 12)) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path, format='PNG')
    return path


def make_dataset(root: Path, items: list[dict], *, dataset_id: str = 'fixture-set',
                 version: str = '1', ontology: dict | None = None) -> Path:
    """Write a validated-shape dataset snapshot; each item gives split/objects/color."""
    root = Path(root)
    ontology = ontology or ONTOLOGY
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(items):
        stem = f'image-{index:06d}'
        image = root / 'images' / (stem + '.png')
        # Distinct default colours keep image hashes unique, as the contract requires.
        color = item.get('color') or (10 + index * 7, 40 + index * 3, 60 + index * 11)
        Image.new('RGB', item.get('size', (16, 12)), color).save(image)
        objects = [{'id': f'object-{position:04d}', 'class_id': obj['class_id'], 'bbox': list(obj['bbox'])}
                   for position, obj in enumerate(item.get('objects', []))]
        annotation = {
            'schema_version': '1.0.0', 'id': f'{dataset_id}-approved-{index:06d}',
            'item_id': f'{dataset_id}-{index:06d}', 'image_sha256': sha256_file(image),
            'ontology_id': ontology['id'], 'ontology_version': ontology['version'],
            'created_at': '2026-01-01T00:00:00Z', 'status': 'approved',
            'source': {'kind': 'human', 'actor_id': 'fixture', 'run_id': 'fixture-run',
                       'tool_version': 'fixture-1'},
            'review': {'reviewer_id': 'fixture', 'reviewed_at': '2026-01-01T00:00:00Z',
                       'reason': 'synthetic fixture'},
            'objects': objects,
        }
        annotation_path = root / 'annotations' / (stem + '-approved.json')
        write_json(annotation_path, annotation)
        records.append({
            'id': annotation['item_id'],
            'image': {'path': f'images/{stem}.png', 'sha256': annotation['image_sha256']},
            'width': item.get('size', (16, 12))[0], 'height': item.get('size', (16, 12))[1],
            'group_id': item.get('group_id', f'group-{index}'), 'split': item['split'],
            'provenance': {'source_ref': f'fixture:{index}', 'acquired_at': '2026-01-01T00:00:00Z'},
            'annotation': {'id': annotation['id'],
                           'artifact': {'path': f'annotations/{stem}-approved.json',
                                        'sha256': sha256_file(annotation_path)}},
        })
    manifest = {'schema_version': '1.0.0', 'id': dataset_id, 'version': version,
                'created_at': '2026-01-01T00:00:00Z', 'ontology': ontology, 'items': records}
    path = root / 'dataset.json'
    write_json(path, manifest)
    return path
