"""Strict JSON and offline Draft 2020-12 validation."""
from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable

from .errors import ValidationResult


class StrictJSONError(ValueError):
    """Invalid JSON, encoding, duplicate key or nonfinite number."""


def loads_strict(text: str) -> Any:
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise StrictJSONError(f'duplicate key {key!r}')
            result[key] = value
        return result

    def constant(value):
        raise StrictJSONError(f'non-finite number {value} is not allowed')

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            return constant(value)
        return parsed

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except (ValueError, RecursionError) as exc:
        raise StrictJSONError(str(exc)) from exc


def read_and_parse(path: Path) -> Any:
    try:
        return loads_strict(path.read_text(encoding='utf-8'))
    except UnicodeError as exc:
        raise StrictJSONError('document is not valid UTF-8') from exc


def pointer(parts) -> str:
    return ''.join('/' + str(p).replace('~', '~0').replace('/', '~1') for p in parts)


class SchemaStore:
    def __init__(self, schema_dir=None) -> None:
        self.schema_dir = schema_dir if schema_dir is not None else default_schema_dir()
        self.schemas: dict[str, dict[str, Any]] = {}
        self.registry = Registry()
        for path in sorted(self.schema_dir.iterdir(), key=lambda p: p.name):
            if not path.name.endswith('.schema.json'):
                continue
            schema = loads_strict(path.read_text(encoding='utf-8'))
            if not isinstance(schema, dict):
                raise ValueError('contract schemas must be JSON objects')
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise ValueError(f'invalid contract schema: {exc.message}') from exc
            sid = schema.get('$id')
            if not sid or sid in self.schemas:
                raise ValueError(f'missing or duplicate schema ID: {sid!r}')
            self.schemas[sid] = schema
            self.registry = self.registry.with_resource(sid, Resource.from_contents(schema))
        if not self.schemas:
            raise ValueError('schema directory contains no contract schemas')
        # Reject nonlocal/unresolved refs at configuration time, never retrieve them.
        def refs(node, sid):
            if isinstance(node, dict):
                for keyword in ('$ref', '$dynamicRef'):
                    if keyword in node:
                        target = node[keyword].split('#')[0]
                        if target and target not in self.schemas:
                            raise ValueError(f'unknown or nonlocal schema reference: {target}')
                        try:
                            self.registry.resolver(sid).lookup(node[keyword])
                        except Unresolvable as exc:
                            raise ValueError(f'unresolved schema reference: {node[keyword]}') from exc
                for value in node.values():
                    refs(value, sid)
            elif isinstance(node, list):
                for value in node:
                    refs(value, sid)
        for sid, schema in self.schemas.items():
            refs(schema, sid)

    def check(self, document: Any, schema_id: str, source: str, result: ValidationResult) -> bool:
        if schema_id not in self.schemas:
            raise ValueError(f'unknown schema {schema_id!r}')
        nonfinite = []
        def scan(node, parts):
            if isinstance(node, float) and not math.isfinite(node):
                nonfinite.append(pointer(parts))
            elif isinstance(node, dict):
                for key, value in node.items():
                    scan(value, [*parts, key])
            elif isinstance(node, (list, tuple)):
                for index, value in enumerate(node):
                    scan(value, [*parts, index])
        scan(document, [])
        for field in nonfinite:
            result.add_error(source, field, 'non_finite', 'number must be finite')
        if nonfinite:
            return False
        validator = Draft202012Validator(self.schemas[schema_id], registry=self.registry,
                                        format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(document), key=lambda e: pointer(e.absolute_path))
        for error in errors:
            result.add_error(source, pointer(error.absolute_path), 'schema',
                             f'{pointer(error.absolute_path)}: {error.message}')
        return not errors

    def validate(self, document: Any, schema_id: str, source: str) -> list[str]:
        result = ValidationResult()
        self.check(document, schema_id, source, result)
        return [str(error) for error in result.errors]

    def load(self, path: Path, schema_id: str, result: ValidationResult):
        try:
            doc = read_and_parse(path)
        except (StrictJSONError, OSError) as exc:
            result.add_error(str(path), '', 'strict_json', str(exc))
            return None
        return doc if self.check(doc, schema_id, str(path), result) else None


def default_schema_dir():
    return files('contracts').joinpath('schemas')
