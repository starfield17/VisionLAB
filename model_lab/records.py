"""Portable immutable labeling-run audit records.

Candidate/LocateAnything run metadata is the producer's own audit store: the
run-record schema (training/evaluation/export) deliberately does not cover
labeling runs. This module validates a portable `labeling_run` record shape
containing the fields required by docs/AUTOLABEL_TRAIN_INFORMATION.md:
checkpoint revision/hash, prompt, canonical image hash, backend, decode mode,
command options and raw-result hash.

Records are immutable snapshots referencing artifacts by relative POSIX path
and hash. Machine-specific transient command paths are rejected; config and
command must carry logical identifiers and relative references only.
"""
from __future__ import annotations

import copy
import re

from contractcheck.common import check_utc_timestamp, scan_open_config, validate_relative_posix_path
from contractcheck.errors import ValidationResult

_SHA256 = re.compile(r'^[a-f0-9]{64}$')

_LABELING_REQUIRED = ('schema_version', 'kind', 'id', 'created_at', 'tool', 'checkpoint',
                      'prompt', 'backend', 'decode_mode', 'command', 'inputs', 'outputs')


def _need(value, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{field} must be a nonempty string')


def labeling_record(record: dict) -> dict:
    """Validate an immutable labeling-run audit record and return an independent copy."""
    doc = copy.deepcopy(record)
    for field in _LABELING_REQUIRED:
        if field not in doc:
            raise ValueError(f'missing required field {field!r}')
    if doc['schema_version'] != '1.0.0':
        raise ValueError('schema_version must be "1.0.0"')
    if doc['kind'] != 'labeling_run':
        raise ValueError('kind must be "labeling_run"')
    result = ValidationResult()
    _need(doc['id'], 'id')
    check_utc_timestamp(result, doc['created_at'], '<record>', '/created_at')
    if not result.ok:
        raise ValueError('\n'.join(str(e) for e in result.errors))
    _need(doc['prompt'], 'prompt')
    _need(doc['backend'], 'backend')
    _need(doc['decode_mode'], 'decode_mode')

    tool = doc['tool']
    for field in ('name', 'revision', 'version'):
        _need(tool.get(field), f'tool/{field}')
    checkpoint = doc['checkpoint']
    for field in ('revision', 'reference', 'license'):
        _need(checkpoint.get(field), f'checkpoint/{field}')
    if not isinstance(checkpoint.get('sha256'), str) or not _SHA256.fullmatch(checkpoint['sha256']):
        raise ValueError('checkpoint/sha256 must be lowercase hex SHA-256')
    if type(checkpoint.get('size_bytes')) is not int or checkpoint['size_bytes'] <= 0:
        raise ValueError('checkpoint/size_bytes must be a positive integer')

    command = doc['command']
    if not isinstance(command.get('args'), list) or not all(isinstance(a, str) and a for a in command['args']):
        raise ValueError('command/args must be nonempty strings')
    for field, minimum in (('threads', 1), ('generation_token_limit', 1)):
        if type(command.get(field)) is not int or command[field] < minimum:
            raise ValueError(f'command/{field} must be a positive integer')
    scan_open_config(command, '<record>', '/command', result)
    if not result.ok:
        raise ValueError('\n'.join(str(e) for e in result.errors))

    for kind in ('inputs', 'outputs'):
        refs = doc[kind]
        if not isinstance(refs, list) or not refs:
            raise ValueError(f'{kind} must be a nonempty list')
        for i, ref in enumerate(refs):
            if not isinstance(ref, dict) or not ref.get('role'):
                raise ValueError(f'{kind}/{i} must declare a nonempty role')
            artifact = ref.get('artifact')
            if not isinstance(artifact, dict):
                raise ValueError(f'{kind}/{i}/artifact must be an object')
            path, digest = artifact.get('path'), artifact.get('sha256')
            if not isinstance(path, str) or not validate_relative_posix_path(path):
                raise ValueError(f'{kind}/{i}/artifact/path must be a clean relative POSIX path')
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError(f'{kind}/{i}/artifact/sha256 must be lowercase hex SHA-256')
    return doc
