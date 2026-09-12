"""Dataset and training sanity checks that never weaken the contract.

The contract stays permissive about legitimate data. This module reports what a
training job is actually about to learn from, so that a production run is not
started on a split that cannot teach anything: class distributions per split,
negative-image coverage, small objects, duplicate families and ontology entries
with no training instances. Findings are classified as `blocking` (refuse to
train unless an explicit reason overrides it) or `warning` (record and continue).
"""
from __future__ import annotations

import json
from pathlib import Path

from contractcheck.common import ArtifactResolver, sha256_file
from contractcheck.errors import ValidationResult
from contractcheck.loader import SchemaStore, read_and_parse

SMALL_OBJECT_PX = 32
MIN_TRAIN_INSTANCES = 10
MIN_TRAIN_IMAGES = 100


def analyze(manifest_path: Path, store_roots: tuple[Path, ...] = ()) -> dict:
    """Count images, objects and classes per split by reading real annotations."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_and_parse(manifest_path)
    errors = SchemaStore().validate(manifest, 'dataset.schema.json', str(manifest_path))
    if errors:
        raise ValueError('dataset manifest is invalid: ' + '; '.join(errors))
    resolver = ArtifactResolver(store_roots)
    result = ValidationResult()
    ontology = manifest['ontology']
    splits: dict[str, dict] = {}
    per_class: dict[str, dict] = {entry['id']: {'label': entry['label'], 'total': 0} for entry in ontology['classes']}
    images_without_objects = 0
    small_objects = 0
    groups: dict[str, str] = {}
    for item in manifest['items']:
        split = item['split']
        bucket = splits.setdefault(split, {'images': 0, 'objects': 0, 'negative_images': 0, 'classes': {}})
        bucket['images'] += 1
        groups[item['group_id']] = split
        resolved = resolver.artifact(item['annotation']['artifact'], manifest_path.parent, str(manifest_path),
                                     '/items/annotation/artifact', result)
        if resolved is None:
            raise ValueError('annotation artifact is unavailable: ' +
                             '; '.join(str(error) for error in result.errors))
        objects = read_and_parse(resolved)['objects']
        if not objects:
            images_without_objects += 1
            bucket['negative_images'] += 1
        for obj in objects:
            bucket['objects'] += 1
            bucket['classes'][obj['class_id']] = bucket['classes'].get(obj['class_id'], 0) + 1
            entry = per_class.setdefault(obj['class_id'], {'label': obj['class_id'], 'total': 0})
            entry['total'] += 1
            entry.setdefault(split, 0)
            entry[split] += 1
            x, y, X, Y = obj['bbox']
            if min(X - x, Y - y) < SMALL_OBJECT_PX:
                small_objects += 1
    return {
        'dataset': {'id': manifest['id'], 'version': manifest['version'], 'path': manifest_path.name,
                    'sha256': sha256_file(manifest_path)},
        'ontology': ontology,
        'splits': splits,
        'classes': per_class,
        'totals': {'images': len(manifest['items']), 'images_without_objects': images_without_objects,
                   'small_objects': small_objects, 'groups': len(groups)},
    }


def findings(report: dict, *, min_train_instances: int = MIN_TRAIN_INSTANCES,
             min_train_images: int = MIN_TRAIN_IMAGES, small_object_px: int = SMALL_OBJECT_PX) -> list[dict]:
    """Classify preflight observations; blocking findings stop a production run."""
    issues: list[dict] = []

    def add(severity: str, rule: str, message: str) -> None:
        issues.append({'severity': severity, 'rule': rule, 'message': message})

    splits = report['splits']
    train = splits.get('train', {'images': 0, 'objects': 0, 'negative_images': 0, 'classes': {}})
    validation = splits.get('validation', {'images': 0, 'objects': 0, 'negative_images': 0, 'classes': {}})
    if train['images'] == 0:
        add('blocking', 'missing_train_split', 'the dataset has no train images')
    if validation['images'] == 0:
        add('blocking', 'missing_validation_split', 'the dataset has no validation images')
    if train['images'] and train['objects'] == 0:
        add('blocking', 'train_without_objects',
            'every train image is negative, so training cannot learn the target classes')
    if train['images'] and train['images'] < min_train_images:
        add('warning', 'small_train_split', f'train split has only {train["images"]} images')
    if train['images'] and validation['images'] and train['negative_images'] == 0:
        add('warning', 'no_negative_train_images',
            'no train image is a confirmed negative, so false-positive behaviour is untested')
    for class_id, entry in report['classes'].items():
        total = entry['total']
        in_train = entry.get('train', 0)
        if total == 0:
            add('warning', 'unused_class', f'ontology class {class_id!r} has no instances anywhere')
            continue
        if in_train == 0:
            add('blocking', 'class_without_train_instances',
                f'class {class_id!r} has {total} instances but none in the train split')
        elif in_train < min_train_instances:
            add('warning', 'class_below_min_train_instances',
                f'class {class_id!r} has only {in_train} train instances')
        if in_train and validation.get('classes', {}).get(class_id, 0) == 0:
            add('warning', 'class_not_evaluated',
                f'class {class_id!r} has no validation instances, so its metrics are not measured')
    total_objects = sum(bucket['objects'] for bucket in splits.values())
    if total_objects and report['totals']['small_objects'] / total_objects > 0.5:
        add('warning', 'small_object_dominant',
            f'more than half of the objects are smaller than {small_object_px}px on their short side')
    if total_objects:
        counts = {class_id: entry['total'] for class_id, entry in report['classes'].items()}
        top = max(counts.values())
        if top / total_objects > 0.9 and len(counts) > 1:
            add('warning', 'dominant_class', 'one class holds more than 90% of the objects')
    return issues


def summary_lines(report: dict, issues: list[dict]) -> list[str]:
    lines = [f"dataset {report['dataset']['id']}@{report['dataset']['version']} sha256={report['dataset']['sha256']}"]
    for split in sorted(report['splits']):
        bucket = report['splits'][split]
        classes = ', '.join(f'{name}={count}' for name, count in sorted(bucket['classes'].items())) or 'none'
        lines.append(f'  {split}: images={bucket["images"]} objects={bucket["objects"]} '
                     f'negatives={bucket["negative_images"]} classes=[{classes}]')
    for issue in issues:
        lines.append(f'  {issue["severity"].upper()} {issue["rule"]}: {issue["message"]}')
    return lines


def blocking(issues: list[dict]) -> list[dict]:
    return [issue for issue in issues if issue['severity'] == 'blocking']


def write_report(path: Path, report: dict, issues: list[dict], override_reason: str | None = None) -> dict:
    document = {'report': report, 'findings': issues}
    if override_reason:
        document['override_reason'] = override_reason
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document
