"""Shared semantic rules and contained local artifact resolution."""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .errors import ValidationResult
from .loader import pointer

UTC_TIMESTAMP_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?Z$')


def utc_key(value: str) -> tuple[datetime, Decimal]:
    match = UTC_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValueError('timestamp must be RFC 3339 UTC ending in Z')
    # Keep arbitrary fractional precision; datetime alone truncates microseconds.
    return datetime.strptime(match[1], '%Y-%m-%dT%H:%M:%S'), Decimal(match[2] or '0')


def check_utc_timestamp(result, value, document, field):
    try:
        utc_key(value)
    except ValueError as exc:
        result.add_error(document, field, 'utc_timestamp', str(exc))


def check_time_order(result, before, after, document, field):
    try:
        if utc_key(after) < utc_key(before):
            result.add_error(document, field, 'time_order', 'timestamp is before creation/acquisition')
    except ValueError:
        pass  # Individual UTC checks report invalid timestamps.


def finite(value):
    return type(value) in (int, float) and (not isinstance(value, float) or math.isfinite(value))


def check_box(result, bbox, width, height, document, field):
    if len(bbox) != 4 or not all(finite(v) for v in bbox):
        result.add_error(document, field, 'box_shape', 'bbox must contain four finite numbers')
        return
    x, y, X, Y = bbox
    if x >= X or y >= Y:
        result.add_error(document, field, 'box_inverted', 'bbox must have positive area')
    if x < 0 or y < 0 or X > width or Y > height:
        result.add_error(document, field, 'box_out_of_bounds', 'bbox is outside the original image')


def check_unique(result, values, document, owner, kind):
    seen = set()
    for value, field in values:
        if value in seen:
            result.add_error(document, field, 'duplicate_id', f'duplicate {kind} {value!r} in {owner}')
        seen.add(value)


def validate_relative_posix_path(path: str) -> bool:
    return bool(path) and not any(c in path for c in ('\\', ':', '\x00')) and all(
        part not in ('', '.', '..') for part in path.split('/'))


class ArtifactResolver:
    """Search package/dataset root first, then explicit retained roots in order.

    A present but invalid/mismatched primary artifact is an error, not a fallback.
    All hashes are verified by consumers, never used to guess a different file.
    """
    def __init__(self, roots: tuple[Path, ...] = ()):
        self.roots = roots

    def resolve(self, path: str, root: Path, document: str, field: str,
                result: ValidationResult) -> Path | None:
        if not validate_relative_posix_path(path):
            result.add_error(document, field, 'invalid_path', 'expected a clean relative POSIX path')
            return None
        for base in (root, *self.roots):
            try:
                base = base.resolve()
                candidate = (base / path).resolve()
                if not candidate.is_relative_to(base):
                    result.add_error(document, field, 'path_escape', 'artifact resolves outside its root')
                    return None
                if candidate.is_file():
                    return candidate
            except (OSError, ValueError, RuntimeError) as exc:
                result.add_error(document, field, 'artifact_io', str(exc))
                return None
        result.add_error(document, field, 'artifact_unresolved' if self.roots else 'artifact_missing',
                         f'artifact {path!r} was not found')
        return None

    def artifact(self, ref, root, document, field, result):
        path = self.resolve(ref['path'], root, document, field + '/path', result)
        if path is not None and not check_artifact_hash(path, ref['sha256'], document, field + '/sha256', result):
            return None
        return path


def resolve_artifact(path, root, document, field, result):
    return ArtifactResolver().resolve(path, root, document, field, result)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_artifact_hash(path, expected, document, field, result):
    try:
        actual = sha256_file(path)
    except OSError as exc:
        result.add_error(document, field, 'artifact_io', str(exc))
        return False
    if actual != expected:
        result.add_error(document, field, 'hash_mismatch', 'artifact bytes do not match declared sha256')
        return False
    return True


def scan_open_config(config, document, field, result):
    def walk(node, parts):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, [*parts, key])
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, [*parts, i])
        elif isinstance(node, float) and not math.isfinite(node):
            result.add_error(document, field + pointer(parts), 'non_finite', 'number must be finite')
        elif isinstance(node, str) and (node.startswith(('/', '~')) or re.match(r'^[A-Za-z]:[\\/]', node)):
            result.add_error(document, field + pointer(parts), 'non_portable_config', 'use portable artifact references')
    walk(config, [])
