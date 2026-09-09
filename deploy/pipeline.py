"""Synchronous fail-fast lifecycle and contract-validated event emission."""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from contractcheck.detect_event import DetectionEventValidator
from contractcheck.loader import SchemaStore
from contractcheck.package import PackageValidator

from .postprocessing import postprocess
from .preprocessing import preprocess
from .registry import RuntimeRegistry
from .types import Sink, Source


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class Pipeline:
    def __init__(self, source: Source, sink: Sink, registry: RuntimeRegistry,
                 package_dir: Path, *, schemas: SchemaStore | None = None,
                 expected_manifest_hash: str | None = None,
                 clock: Callable[[], str] = utc_now,
                 event_id: Callable[[], str] = lambda: str(uuid4())) -> None:
        self.source, self.sink, self.registry = source, sink, registry
        self.package_dir = package_dir
        self.schemas = schemas or SchemaStore()
        self.expected_manifest_hash = expected_manifest_hash
        self.clock, self.event_id = clock, event_id
        self._started = False

    def run(self) -> int:
        if self._started:
            raise RuntimeError('Pipeline instances are single-use; create a new session to restart')
        self._started = True
        package = PackageValidator(self.schemas, self.registry.descriptions).load(
            self.package_dir, self.expected_manifest_hash)
        descriptor = package.manifest['model']['adapter']
        adapter = self.registry.create(descriptor['id'], descriptor['version'])
        labels = {c['id']: c['label'] for c in package.labels['classes']}
        validator = DetectionEventValidator(self.schemas)
        cleanup: list[Callable[[], None]] = []
        last_indices: dict[tuple[str, str], int] = {}
        ids: set[str] = set()
        count = 0
        try:
            # close() must be safe after a partially failed open(), by port contract.
            cleanup.append(self.source.close)
            self.source.open()
            cleanup.append(adapter.close)
            adapter.open(package)
            cleanup.append(self.sink.close)
            self.sink.open()
            while True:
                frame = self.source.read()
                if frame is None:
                    return count
                key = (frame.source.id, frame.source.session_id)
                if frame.source.frame_index <= last_indices.get(key, -1):
                    raise ValueError('frame indices must increase within a source session')
                last_indices[key] = frame.source.frame_index
                tensor, transform = preprocess(frame.image, package.preprocessing)
                detections = postprocess(adapter.infer(tensor, transform), transform.original_width,
                                         transform.original_height, labels, package.manifest['postprocessing'])
                event: dict[str, Any] = {
                    'schema_version': '1.0.0', 'event_id': self.event_id(),
                    'captured_at': frame.captured_at, 'emitted_at': self.clock(),
                    'source': asdict(frame.source),
                    'model': {'package_id': package.manifest['package_id'], 'manifest_sha256': package.manifest_sha256},
                    'image': {'width': transform.original_width, 'height': transform.original_height},
                    'detections': [{'bbox': [float(v) for v in d.bbox], 'class_id': d.class_id,
                                    'label': labels[d.class_id], 'confidence': float(d.confidence)} for d in detections],
                }
                result = validator.validate_document(event, package=package)
                if not result.ok:
                    raise ValueError('\n'.join(str(error) for error in result.errors))
                if event['event_id'] in ids:
                    raise ValueError('event IDs must be unique')
                ids.add(event['event_id'])
                self.sink.write(event)
                count += 1
        finally:
            primary = sys.exc_info()[1]
            first_cleanup: BaseException | None = None
            for close in reversed(cleanup):
                try:
                    close()
                except BaseException as exc:
                    if first_cleanup is None:
                        first_cleanup = exc
                    logging.getLogger(__name__).error('resource cleanup failed', exc_info=True)
            if primary is None and first_cleanup is not None:
                raise first_cleanup
