"""Single-event semantics, optionally checked against the loaded package."""
from pathlib import Path

from . import common
from .errors import ValidationResult
from .loader import SchemaStore


class DetectionEventValidator:
    def __init__(self, store: SchemaStore):
        self.store = store

    def validate(self, path: Path, *, package=None) -> ValidationResult:
        result = ValidationResult()
        doc = self.store.load(path, 'detection-event.schema.json', result)
        if doc is not None:
            self._semantics(doc, str(path), result, package)
        return result

    def validate_document(self, doc, *, package=None) -> ValidationResult:
        result = ValidationResult()
        if self.store.check(doc, 'detection-event.schema.json', '<event>', result):
            self._semantics(doc, '<event>', result, package)
        return result

    def _semantics(self, doc, document, result, package):
        for field in ('captured_at', 'emitted_at'):
            common.check_utc_timestamp(result, doc[field], document, '/' + field)
        common.check_time_order(result, doc['captured_at'], doc['emitted_at'], document, '/emitted_at')
        for i, det in enumerate(doc['detections']):
            common.check_box(result, det['bbox'], doc['image']['width'], doc['image']['height'], document,
                             f'/detections/{i}/bbox')
        if package is not None:
            if doc['model'] != {'package_id': package.manifest['package_id'], 'manifest_sha256': package.manifest_sha256}:
                result.add_error(document, '/model', 'model_mismatch', 'event does not identify the loaded package')
            labels = {c['id']: c['label'] for c in package.labels['classes']}
            for i, det in enumerate(doc['detections']):
                if labels.get(det['class_id']) != det['label']:
                    result.add_error(document, f'/detections/{i}/class_id', 'unknown_class', 'event class/label differs from package')
