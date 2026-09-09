"""Convert pixel-space LocateAnything results and reviewed datasets."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from contractcheck.common import check_box, check_time_order, check_utc_timestamp, sha256_file
from contractcheck.dataset import DatasetValidator
from contractcheck.errors import ValidationResult
from contractcheck.loader import SchemaStore, read_and_parse


def candidate(raw: dict, *, annotation_id: str, item_id: str, image_hash: str,
              width: int, height: int, ontology: dict, source: dict, created_at: str) -> dict:
    """The pinned C++ CLI returns pixel boxes and no confidence score."""
    store = SchemaStore()
    result = ValidationResult()
    store.check(ontology, 'ontology.schema.json', '<ontology>', result)
    if not result.ok:
        raise ValueError(str(result.errors))
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('positive integer image dimensions required')
    if not isinstance(raw, dict) or not isinstance(raw.get('detections'), list):
        raise ValueError('expected a detections array')
    labels = {entry['label']: entry['id'] for entry in ontology['classes']}
    if len(labels) != len(ontology['classes']) or len(set(labels.values())) != len(labels):
        raise ValueError('ontology IDs and labels must be unique')
    objects = []
    for i, detection in enumerate(raw['detections']):
        if not isinstance(detection, dict) or detection.get('label') not in labels:
            raise ValueError('unknown label; explicit ontology mapping required')
        bbox = detection.get('box')
        if not isinstance(bbox, list):
            raise ValueError('expected pixel-space box array')
        check_box(result, bbox, width, height, '<response>', f'/detections/{i}/box')
        objects.append({'id': f'object-{i:04d}', 'class_id': labels[detection['label']], 'bbox': bbox})
    if source.get('kind') != 'locate_anything':
        raise ValueError('source must identify locate_anything')
    doc = {'schema_version': '1.0.0', 'id': annotation_id, 'item_id': item_id,
           'image_sha256': image_hash, 'ontology_id': ontology['id'], 'ontology_version': ontology['version'],
           'created_at': created_at, 'status': 'candidate', 'source': source, 'objects': objects}
    check_utc_timestamp(result, created_at, '<annotation>', '/created_at')
    store.check(doc, 'annotation.schema.json', '<annotation>', result)
    if not result.ok:
        raise ValueError('\n'.join(str(e) for e in result.errors))
    return doc


def export_yolo(manifest_path: Path, destination: Path) -> None:
    """Validate all content first; atomically publish into a new directory.

    Labels use ontology declaration order. Opaque item/class IDs are never paths.
    data.yaml deliberately has no path key: Ultralytics resolves it at the file.
    """
    manifest_path, destination = Path(manifest_path), Path(destination)
    result = DatasetValidator(SchemaStore()).validate(manifest_path)
    if not result.ok:
        raise ValueError('\n'.join(str(e) for e in result.errors))
    ds = read_and_parse(manifest_path)
    splits = {item['split'] for item in ds['items']}
    if not {'train', 'validation'} <= splits:
        raise ValueError('training export requires nonempty train and validation splits')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    classes = ds['ontology']['classes']
    indices = {entry['id']: i for i, entry in enumerate(classes)}
    split_names = {'train': 'train', 'validation': 'val', 'test': 'test'}
    with tempfile.TemporaryDirectory(prefix='.yolo-', dir=destination.parent) as temp:
        root = Path(temp) / 'dataset'
        root.mkdir()
        mapping = []
        for i, item in enumerate(ds['items']):
            split = split_names[item['split']]
            src = manifest_path.parent / item['image']['path']
            suffix = src.suffix.lower()
            if suffix not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'):
                raise ValueError('expected a canonical image file, not a fixture or arbitrary binary')
            stem = f'image-{i:06d}'
            image_dir, label_dir = root / 'images' / split, root / 'labels' / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            image_out = image_dir / (stem + suffix)
            shutil.copyfile(src, image_out)
            if sha256_file(image_out) != item['image']['sha256']:
                raise ValueError('source image changed during export')
            ann_path = manifest_path.parent / item['annotation']['artifact']['path']
            if sha256_file(ann_path) != item['annotation']['artifact']['sha256']:
                raise ValueError('annotation changed during export')
            ann = read_and_parse(ann_path)
            lines = []
            for obj in ann['objects']:
                x, y, X, Y = obj['bbox']
                values = ((x + X) / (2 * item['width']), (y + Y) / (2 * item['height']),
                          (X - x) / item['width'], (Y - y) / item['height'])
                lines.append(str(indices[obj['class_id']]) + ' ' + ' '.join(format(v, '.17g') for v in values))
            (label_dir / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            mapping.append({'item_id': item['id'], 'image': image_out.relative_to(root).as_posix(),
                            'annotation_id': ann['id']})
        # JSON is valid YAML and avoids quoting ambiguities in ontology labels.
        config = {'train': 'images/train', 'val': 'images/val', 'names': [c['label'] for c in classes]}
        if 'test' in splits:
            config['test'] = 'images/test'
        (root / 'data.yaml').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        metadata = {'dataset_id': ds['id'], 'dataset_version': ds['version'],
                    'manifest_sha256': sha256_file(manifest_path),
                    'classes': [{'index': i, 'class_id': c['id'], 'label': c['label']} for i, c in enumerate(classes)],
                    'items': mapping}
        (root / 'mapping.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        root.rename(destination)


def approve(annotation: dict, *, approval_id: str, reviewer_id: str, reviewed_at: str, reason: str) -> dict:
    """Create an immutable approved revision of a candidate annotation.

    The original candidate stays unchanged; the new record keeps the original
    source provenance and points back via `supersedes`. The review time must
    be at or after the candidate creation time.
    """
    if annotation.get('status') != 'candidate':
        raise ValueError('only candidate annotations can be approved')
    doc = {**annotation, 'id': approval_id, 'status': 'approved',
           'supersedes': annotation['id'],
           'review': {'reviewer_id': reviewer_id, 'reviewed_at': reviewed_at, 'reason': reason}}
    store, result = SchemaStore(), ValidationResult()
    check_utc_timestamp(result, doc['created_at'], '<approval>', '/created_at')
    check_utc_timestamp(result, reviewed_at, '<approval>', '/review/reviewed_at')
    check_time_order(result, doc['created_at'], reviewed_at, '<approval>', '/review/reviewed_at')
    store.check(doc, 'annotation.schema.json', '<approval>', result)
    if not result.ok:
        raise ValueError('\n'.join(str(e) for e in result.errors))
    return doc


def build_dataset(destination: Path, *, dataset_id: str, version: str, created_at: str,
                  ontology: dict, items: list) -> None:
    """Publish a validated versioned Dataset manifest from canonical images and reviewed annotations.

    Each item: {'id', 'image' (canonical PNG path), 'width', 'height', 'group_id', 'split',
                'provenance': {'source_ref', 'acquired_at'},
                'annotation' (candidate dict), 'approved' (approved dict)}.
    The approved record must cite the candidate; both are retained next to the
    manifest so DatasetValidator can resolve the revision chain. Canonical
    image bytes are rehashed and must equal the annotation image hash. The
    snapshot is fully validated before the directory is published.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    store, result = SchemaStore(), ValidationResult()
    store.check(ontology, 'ontology.schema.json', '<ontology>', result)
    labels = {c['id']: c['label'] for c in ontology['classes']}
    if len(labels) != len(ontology['classes']) or len(set(labels.values())) != len(labels):
        raise ValueError('ontology IDs and labels must be unique')
    check_utc_timestamp(result, created_at, '<dataset>', '/created_at')
    for i, item in enumerate(items):
        check_utc_timestamp(result, item['provenance']['acquired_at'], '<dataset>', f'/items/{i}/provenance/acquired_at')
    if not result.ok:
        raise ValueError('\n'.join(str(e) for e in result.errors))
    for split in {item['split'] for item in items} - {'train', 'validation', 'test'}:
        raise ValueError(f'unknown split {split!r}')
    with tempfile.TemporaryDirectory(prefix='.dataset-', dir=destination.parent) as temp:
        root = Path(temp) / 'dataset'
        (root / 'images').mkdir(parents=True)
        (root / 'annotations').mkdir()
        records = []
        for i, item in enumerate(items):
            if item['approved'].get('status') != 'approved':
                raise ValueError('released annotations must be approved')
            stem = f'image-{i:06d}'
            image_out = root / 'images' / (stem + '.png')
            shutil.copyfile(item['image'], image_out)
            if sha256_file(image_out) != item['annotation']['image_sha256']:
                raise ValueError('canonical image hash does not match the annotation')
            ann_out = root / 'annotations' / (stem + '-approved.json')
            (root / 'annotations' / (stem + '-candidate.json')).write_text(
                json.dumps(item['annotation'], indent=2) + '\n', encoding='utf-8')
            ann_out.write_text(json.dumps(item['approved'], indent=2) + '\n', encoding='utf-8')
            records.append({'id': item['id'], 'image': {'path': f'images/{stem}.png',
                                                        'sha256': item['annotation']['image_sha256']},
                            'width': item['width'], 'height': item['height'], 'group_id': item['group_id'],
                            'split': item['split'], 'provenance': item['provenance'],
                            'annotation': {'id': item['approved']['id'],
                                           'artifact': {'path': f'annotations/{stem}-approved.json',
                                                        'sha256': sha256_file(ann_out)}}})
        manifest = {'schema_version': '1.0.0', 'id': dataset_id, 'version': version,
                    'created_at': created_at, 'ontology': ontology, 'items': records}
        manifest_path = root / 'dataset.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        check = DatasetValidator(SchemaStore()).validate(manifest_path)
        if not check.ok:
            raise ValueError('\n'.join(str(e) for e in check.errors))
        root.rename(destination)