"""Dataset and annotation validation with injected retained-store resolution."""
from pathlib import Path

from . import common
from .annotation import check_classes, validate_annotation
from .errors import ValidationResult
from .loader import SchemaStore, StrictJSONError, read_and_parse


class DatasetValidator:
    def __init__(self, store: SchemaStore, resolver: common.ArtifactResolver | None = None):
        self.store = store
        self.resolver = resolver or common.ArtifactResolver()

    def validate(self, manifest_path: Path) -> ValidationResult:
        result = ValidationResult()
        manifest = self.store.load(manifest_path, 'dataset.schema.json', result)
        if manifest is not None:
            self.validate_document(manifest, manifest_path, result)
        return result

    def validate_document(self, manifest, path, result, *, content=True):
        document = str(path)
        common.check_utc_timestamp(result, manifest['created_at'], document, '/created_at')
        check_classes(manifest['ontology']['classes'], document, result, '/ontology/classes')
        items = manifest['items']
        for key, values in [('item id', ((item['id'], f'/items/{i}/id') for i, item in enumerate(items))),
                            ('image hash', ((item['image']['sha256'], f'/items/{i}/image/sha256') for i, item in enumerate(items))),
                            ('annotation id', ((item['annotation']['id'], f'/items/{i}/annotation/id') for i, item in enumerate(items)))]:
            common.check_unique(result, values, document, 'dataset', key)
        groups: dict[str, str] = {}
        for i, item in enumerate(items):
            if item['group_id'] in groups and groups[item['group_id']] != item['split']:
                result.add_error(document, f'/items/{i}/group_id', 'split_leakage', 'group spans multiple splits')
            groups[item['group_id']] = item['split']
            common.check_utc_timestamp(result, item['provenance']['acquired_at'], document, f'/items/{i}/provenance/acquired_at')
            if not content:
                continue
            self.resolver.artifact(item['image'], path.parent, document, f'/items/{i}/image', result)
            ann_path = self.resolver.artifact(item['annotation']['artifact'], path.parent, document,
                                              f'/items/{i}/annotation/artifact', result)
            if ann_path is None:
                continue
            ann = self.store.load(ann_path, 'annotation.schema.json', result)
            if ann is None:
                # Specific rule retained alongside schema error for callers.
                try:
                    raw = read_and_parse(ann_path)
                    if isinstance(raw, dict) and raw.get('status') in ('approved', 'rejected') and 'review' not in raw:
                        result.add_error(str(ann_path), '/review', 'missing_review', 'review record is required')
                except (OSError, StrictJSONError):
                    pass
                continue
            validate_annotation(ann, str(ann_path), result, item=item, ontology=manifest['ontology'])
            self._history(ann, ann_path, result)

    def _history(self, annotation, path, result):
        if 'supersedes' not in annotation:
            if annotation['source']['kind'] != 'human' and annotation['status'] in ('approved', 'rejected'):
                result.add_error(str(path), '/supersedes', 'audit_incomplete', 'machine review must reference original candidate')
            return
        # Index only explicitly available local annotation records; never fetch opaque IDs.
        index = {annotation['id']: (annotation, path)}
        for root in (path.parent, *self.resolver.roots):
            try:
                for candidate in root.rglob('*.json'):
                    if not candidate.resolve().is_relative_to(root.resolve()):
                        continue
                    try:
                        doc = read_and_parse(candidate)
                    except (OSError, StrictJSONError):
                        continue
                    if isinstance(doc, dict) and 'item_id' in doc and 'id' in doc and isinstance(doc['id'], str):
                        if doc['id'] in index and index[doc['id']][0] != doc:
                            result.add_error(str(candidate), '/id', 'duplicate_id', 'conflicting annotation revisions')
                        index[doc['id']] = (doc, candidate)
            except (OSError, RuntimeError) as exc:
                result.add_error(str(path), '/supersedes', 'artifact_io', str(exc))
                return
        seen = {annotation['id']}
        current = annotation
        while 'supersedes' in current:
            target = current['supersedes']
            if target in seen:
                result.add_error(str(path), '/supersedes', 'annotation_cycle', 'revision history is cyclic')
                return
            if target not in index:
                result.add_error(str(path), '/supersedes', 'audit_incomplete', f'annotation revision {target!r} is unavailable')
                return
            seen.add(target)
            previous, previous_path = index[target]
            if not self.store.check(previous, 'annotation.schema.json', str(previous_path), result):
                return
            validate_annotation(previous, str(previous_path), result)
            for field in ('item_id', 'image_sha256', 'ontology_id', 'ontology_version', 'source'):
                if previous[field] != current[field]:
                    result.add_error(str(path), '/supersedes', 'annotation_mismatch', f'revision changes {field}')
            common.check_time_order(result, previous['created_at'], current['created_at'], str(path), '/created_at')
            current = previous
        if current['source']['kind'] != 'human' and current['status'] != 'candidate':
            result.add_error(str(path), '/supersedes', 'audit_incomplete', 'machine history must start with a candidate')
