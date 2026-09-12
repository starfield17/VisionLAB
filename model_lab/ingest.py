"""Ingest an already-annotated public detection dataset into the Dataset contract.

Published labels are adopted, never re-invented: every released annotation keeps
the upstream identity as its `source.actor_id`/`review.reviewer_id` and states in
its review reason that this repository did not perform an independent human
review. The converter decodes and re-encodes images into a canonical RGB PNG
frame, rescales boxes into that frame, drops boxes that cannot be represented
honestly (non-finite, zero area, outside the frame) with recorded counts, groups
duplicate families and assigns deterministic splits with class coverage.

Nothing here runs inference or downloads anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from contractcheck.dataset import DatasetValidator
from contractcheck.errors import ValidationResult
from contractcheck.loader import SchemaStore, read_and_parse

from .acquisition import ImageDecodeError, canonicalize
from .runrecords import utc_now

EXIF_BOX_FRAME_UNSUPPORTED = 'exif_orientation'
SMALL_OBJECT_PX = 32
DUPLICATE_HAMMING = 4


def load_ontology_config(path: Path) -> tuple[dict, dict, set]:
    """Return (ontology, category-name -> class-id mapping, ignored category names)."""
    config = read_and_parse(Path(path))
    if not isinstance(config, dict) or set(config) - {'ontology', 'categories', 'ignore'}:
        raise ValueError('ontology config accepts exactly ontology, categories and ignore')
    ontology = config.get('ontology')
    mapping = config.get('categories')
    ignore = set(config.get('ignore') or [])
    result = ValidationResult()
    SchemaStore().check(ontology, 'ontology.schema.json', str(path), result)
    if not result.ok:
        raise ValueError('\n'.join(str(error) for error in result.errors))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('ontology config needs a nonempty categories mapping')
    class_ids = {entry['id'] for entry in ontology['classes']}
    unknown = sorted({value for value in mapping.values() if value not in class_ids})
    if unknown:
        raise ValueError(f'categories map to unknown ontology classes: {unknown}')
    if ignore & set(mapping):
        raise ValueError('a category cannot be both mapped and ignored')
    return ontology, {name: value for name, value in mapping.items()}, ignore


def _average_hash(image) -> int:
    small = image.convert('L').resize((8, 8))
    pixels = list(small.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


class _Groups:
    """Union-find over images that share bytes or a near-duplicate perceptual hash."""

    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _duplicate_groups(items: list[dict]) -> dict[str, str]:
    ids = [item['id'] for item in items]
    union = _Groups(ids)
    by_bytes: dict[str, str] = {}
    for item in items:
        previous = by_bytes.setdefault(item['image']['sha256'], item['id'])
        union.union(previous, item['id'])
    buckets: dict[int, list[str]] = defaultdict(list)
    for item in items:
        buckets[item['perceptual_hash']].append(item['id'])
    keys = sorted(buckets)
    for members in buckets.values():
        for member in members[1:]:
            union.union(members[0], member)
    for index, key in enumerate(keys):
        for other in keys[index + 1:]:
            if bin(key ^ other).count('1') <= DUPLICATE_HAMMING:
                union.union(buckets[key][0], buckets[other][0])
    return {item['id']: f'group-{union.find(item["id"])}' for item in items}


def _stable_order(values: list[str], *, dataset_id: str, seed: int) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f'{dataset_id}:{seed}:{value}'.encode()).hexdigest())


def assign_splits(items: list[dict], *, dataset_id: str, seed: int, train_fraction: float) -> dict[str, str]:
    """Deterministic split assignment that keeps groups intact and train classes covered."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be strictly between 0 and 1')
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item['group_id']].append(item)
    names = _stable_order(sorted(groups), dataset_id=dataset_id, seed=seed)
    if len(names) < 2:
        raise ValueError('at least two independent groups are required to build a split')
    classes_of = {name: {obj['class_id'] for item in groups[name] for obj in item['objects']} for name in names}
    assignment: dict[str, str] = {}
    for class_id in sorted({value for values in classes_of.values() for value in values}):
        for name in names:
            if class_id in classes_of[name]:
                assignment.setdefault(name, 'train')
                break
    target_train = max(len(assignment), min(len(names) - 1, round(len(names) * train_fraction)))
    for name in names:
        if name in assignment:
            continue
        assignment[name] = 'train' if sum(1 for value in assignment.values() if value == 'train') < target_train else 'validation'
    if not any(value == 'validation' for value in assignment.values()):
        def shareable(name: str) -> bool:
            return all(sum(1 for other in names if assignment[other] == 'train' and class_id in classes_of[other]) > 1
                       for class_id in classes_of[name])
        movable = [name for name in reversed(names) if assignment[name] == 'train' and shareable(name)]
        assignment[(movable or [names[-1]])[0]] = 'validation'
    return assignment


def _scale_box(bbox, scale: float, width: int, height: int) -> list[float] | None:
    x, y, w, h = (float(value) for value in bbox)
    x1, y1, x2, y2 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    tolerance = 0.5
    if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
        return None
    return [min(max(x1, 0.0), float(width)), min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)), min(max(y2, 0.0), float(height))]


def ingest_coco(*, images_dir: Path, annotations_path: Path, ontology_config: Path,
                destination: Path, dataset_id: str, version: str, actor_id: str, run_id: str,
                tool_version: str, acquired_at: str, created_at: str | None = None,
                longest_side: int = 1280, seed: int = 42, train_fraction: float = 0.8,
                source_prefix: str | None = None, max_skipped_fraction: float = 0.05) -> dict:
    """Convert a COCO-format image folder into a versioned Dataset snapshot."""
    images_dir, annotations_path, destination = Path(images_dir), Path(annotations_path), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    ontology, mapping, ignore = load_ontology_config(ontology_config)
    document = read_and_parse(annotations_path)
    for field in ('images', 'annotations', 'categories'):
        if not isinstance(document.get(field), list):
            raise ValueError(f'COCO document needs a {field} array')
    category_names = {entry['id']: entry['name'] for entry in document['categories']}
    unmapped = sorted({name for name in category_names.values() if name not in mapping and name not in ignore})
    if unmapped:
        raise ValueError(f'ontology config does not classify these categories: {unmapped}')
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in document['annotations']:
        by_image[annotation['image_id']].append(annotation)
    created = created_at or utc_now()

    dropped = Counter()
    skipped = Counter()
    small_objects = 0
    items: list[dict] = []
    attribution: list[dict] = []
    seen_hashes: set[str] = set()
    images = sorted(document['images'], key=lambda entry: entry['id'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ingest-', dir=destination.parent) as temp:
        root = Path(temp) / 'dataset'
        (root / 'images').mkdir(parents=True)
        (root / 'annotations').mkdir()
        for index, image in enumerate(images):
            source = images_dir / image['file_name']
            if not source.is_file():
                skipped['missing_source'] += 1
                continue
            stem = f'image-{index:06d}'
            try:
                record = canonicalize(source, root / 'images' / f'{stem}.png', longest_side=longest_side)
            except (ImageDecodeError, ValueError, FileExistsError):
                skipped['decode_failed'] += 1
                continue
            if record['transform']['exif_orientation_applied']:
                (root / 'images' / f'{stem}.png').unlink()
                skipped[EXIF_BOX_FRAME_UNSUPPORTED] += 1
                continue
            if record['canonical_sha256'] in seen_hashes:
                # The contract requires unique image hashes per dataset version.
                (root / 'images' / f'{stem}.png').unlink()
                skipped['duplicate_image'] += 1
                continue
            seen_hashes.add(record['canonical_sha256'])
            scale = record['transform']['scale']
            width, height = record['canonical_width'], record['canonical_height']
            objects = []
            for annotation in sorted(by_image.get(image['id'], []), key=lambda entry: entry['id']):
                name = category_names[annotation['category_id']]
                if name in ignore:
                    dropped['ignored_category'] += 1
                    continue
                box = _scale_box(annotation['bbox'], scale, width, height)
                if box is None:
                    dropped['unrepresentable_box'] += 1
                    continue
                if min(box[2] - box[0], box[3] - box[1]) < SMALL_OBJECT_PX:
                    small_objects += 1
                objects.append({'id': f'object-{len(objects):04d}', 'class_id': mapping[name], 'bbox': box})
            item_id = f'{dataset_id}-{image["id"]:012d}'
            annotation_id = f'{dataset_id}-approved-{image["id"]:012d}'
            annotation_document = {
                'schema_version': '1.0.0', 'id': annotation_id, 'item_id': item_id,
                'image_sha256': record['canonical_sha256'], 'ontology_id': ontology['id'],
                'ontology_version': ontology['version'], 'created_at': created,
                'status': 'approved',
                'source': {'kind': 'human', 'actor_id': actor_id, 'run_id': run_id,
                           'tool_version': tool_version},
                'review': {'reviewer_id': actor_id, 'reviewed_at': created,
                           'reason': 'adopted published dataset annotations; not independently '
                                     're-reviewed by this repository'},
                'objects': objects,
            }
            annotation_path = root / 'annotations' / f'{stem}-approved.json'
            annotation_path.write_text(json.dumps(annotation_document, indent=2) + '\n', encoding='utf-8')
            items.append({
                'id': item_id,
                'image': {'path': f'images/{stem}.png', 'sha256': record['canonical_sha256']},
                'perceptual_hash': _average_hash(Image.open(root / 'images' / f'{stem}.png')),
                'width': width, 'height': height,
                'provenance': {'source_ref': f'{source_prefix or dataset_id}:{image["file_name"]}',
                               'acquired_at': acquired_at},
                'annotation': {'id': annotation_id,
                               'artifact': {'path': f'annotations/{stem}-approved.json',
                                            'sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest()}},
                'objects': objects,
                'source_file': image['file_name'],
                'license': image.get('license'),
                'flickr_url': image.get('flickr_url'),
                'coco_url': image.get('coco_url'),
            })
        total_seen = len(images)
        deliberate = {'duplicate_image', EXIF_BOX_FRAME_UNSUPPORTED}
        unusable = sum(count for reason, count in skipped.items() if reason not in deliberate)
        if total_seen and unusable / total_seen > max_skipped_fraction:
            raise ValueError(f'too many images were unusable: {dict(skipped)} of {total_seen}')
        groups = _duplicate_groups(items)
        for item in items:
            item['group_id'] = groups[item['id']]
        splits = assign_splits(items, dataset_id=dataset_id, seed=seed, train_fraction=train_fraction)
        manifest_items = []
        for item in items:
            manifest_items.append({
                'id': item['id'], 'image': item['image'], 'width': item['width'], 'height': item['height'],
                'group_id': item['group_id'], 'split': splits[item['group_id']],
                'provenance': item['provenance'], 'annotation': item['annotation'],
            })
            attribution.append({'item_id': item['id'], 'source_file': item['source_file'],
                                'license': item['license'], 'flickr_url': item['flickr_url'],
                                'coco_url': item['coco_url'], 'group_id': item['group_id']})
        manifest = {'schema_version': '1.0.0', 'id': dataset_id, 'version': version,
                    'created_at': created, 'ontology': ontology, 'items': manifest_items}
        (root / 'dataset.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        check = DatasetValidator(SchemaStore()).validate(root / 'dataset.json')
        if not check.ok:
            raise ValueError('\n'.join(str(error) for error in check.errors))
        split_report: dict[str, dict] = {}
        for item, manifest_item in zip(items, manifest_items):
            bucket = split_report.setdefault(manifest_item['split'],
                                             {'images': 0, 'objects': 0, 'negative_images': 0, 'classes': {}})
            bucket['images'] += 1
            bucket['objects'] += len(item['objects'])
            if not item['objects']:
                bucket['negative_images'] += 1
            for obj in item['objects']:
                bucket['classes'][obj['class_id']] = bucket['classes'].get(obj['class_id'], 0) + 1
        report = {
            'dataset': {'id': dataset_id, 'version': version, 'created_at': created,
                        'manifest_sha256': hashlib.sha256((root / 'dataset.json').read_bytes()).hexdigest(),
                        'annotations_sha256': hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
                        'ontology_id': ontology['id'], 'ontology_version': ontology['version']},
            'counts': {'source_images': len(images), 'items': len(items),
                       'objects': sum(len(item['objects']) for item in items),
                       'objects_dropped': dict(dropped), 'images_skipped': dict(skipped),
                       'small_objects': small_objects},
            'splits': split_report,
            'groups': {'total': len({item['group_id'] for item in items}),
                       'multi_image': sum(1 for _, size in Counter(item['group_id'] for item in items).items()
                                          if size > 1)},
            'policy': {'longest_side': longest_side, 'seed': seed, 'train_fraction': train_fraction,
                       'duplicate_hamming': DUPLICATE_HAMMING, 'small_object_px': SMALL_OBJECT_PX,
                       'alpha_policy': 'composite_white', 'approval': 'adopted published annotations'},
        }
        (root / 'ingest-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (root / 'attribution.json').write_text(json.dumps(attribution, indent=2) + '\n', encoding='utf-8')
        root.rename(destination)
    return report
