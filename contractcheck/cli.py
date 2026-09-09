"""Offline validation CLI. Exit codes: success 0, document errors 1, usage/config 2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, common
from .annotation import check_classes, validate_annotation
from .dataset import DatasetValidator
from .detect_event import DetectionEventValidator
from .errors import ValidationResult
from .loader import SchemaStore, StrictJSONError, read_and_parse
from .package import AdapterRegistry, PackageValidator


def detect_type(doc):
    if not isinstance(doc, dict):
        return None
    for key, kind in [('event_id', 'detection-event'), ('package_id', 'model-package'),
                      ('items', 'dataset'), ('objects', 'annotation'), ('stage', 'run-record'),
                      ('layout', 'preprocessing')]:
        if key in doc:
            return kind
    if 'classes' in doc:
        return 'labels' if 'ontology_id' in doc else 'ontology'
    return None


def run_validate(store: SchemaStore, path: Path) -> ValidationResult:
    result = ValidationResult()
    try:
        doc = read_and_parse(path)
    except (OSError, StrictJSONError) as exc:
        result.add_error(str(path), '', 'strict_json', str(exc))
        return result
    kind = detect_type(doc)
    if kind is None:
        result.add_error(str(path), '', 'unknown_document', 'cannot identify contract document')
        return result
    if kind == 'dataset':
        return DatasetValidator(store).validate(path)
    if kind == 'detection-event':
        return DetectionEventValidator(store).validate(path)
    if not store.check(doc, kind + '.schema.json', str(path), result):
        return result
    if kind == 'annotation':
        validate_annotation(doc, str(path), result)
    if kind in ('labels', 'ontology'):
        check_classes(doc['classes'], str(path), result)
        if kind == 'labels':
            common.check_unique(result, ((c['output_index'], f'/classes/{i}/output_index') for i, c in enumerate(doc['classes'])),
                                str(path), 'labels', 'output index')
    if kind in ('run-record', 'model-package'):
        common.check_utc_timestamp(result, doc['created_at'], str(path), '/created_at')
    if kind == 'run-record':
        common.scan_open_config(doc['config'], str(path), '/config', result)
    return result


def load_adapter_specs(paths: list[Path]) -> AdapterRegistry:
    registry = AdapterRegistry()
    for path in paths:
        spec = read_and_parse(path)
        if not isinstance(spec, dict) or set(spec) != {'id', 'version', 'formats', 'config_schema'}:
            raise ValueError('adapter spec requires exactly id, version, formats, config_schema')
        if not isinstance(spec['id'], str) or not isinstance(spec['version'], str) or not isinstance(spec['formats'], list):
            raise ValueError('invalid adapter spec ID/version/formats')
        registry.register(spec['id'], spec['version'], spec['config_schema'], formats=tuple(spec['formats']))
    return registry


def build_parser():
    parser = argparse.ArgumentParser(prog='contractcheck')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    single = sub.add_parser('validate')
    single.add_argument('file', type=Path)
    single.add_argument('--schemas', type=Path)
    package = sub.add_parser('package').add_subparsers(dest='mode', required=True)
    for mode in ('preflight', 'audit'):
        p = package.add_parser(mode)
        p.add_argument('dir', type=Path)
        p.add_argument('--schemas', type=Path)
        p.add_argument('--adapter-spec', type=Path, action='append', default=[])
        p.add_argument('--adapter', action='append', default=[], help='removed: use --adapter-spec')
        p.add_argument('--expected-manifest-hash')
        if mode == 'audit':
            p.add_argument('--store-root', action='append', type=Path, default=[])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = SchemaStore(args.schemas)
        if args.command == 'package':
            if args.adapter:
                raise ValueError('--adapter ID@VER cannot validate support; use --adapter-spec FILE')
            adapters = load_adapter_specs(args.adapter_spec)
        # Fail as configuration errors if required schemas were omitted.
        expected = {'annotation', 'dataset', 'ontology', 'labels', 'preprocessing', 'run-record', 'model-package', 'detection-event'}
        if not {name + '.schema.json' for name in expected} <= store.schemas.keys():
            raise ValueError('schema store is missing required contracts')
    except (OSError, ValueError) as exc:
        print(f'ERROR configuration: {exc}', file=sys.stderr)
        return 2
    if args.command == 'validate':
        result = run_validate(store, args.file)
    else:
        result = PackageValidator(store, adapters).validate(args.dir, args.mode,
                    tuple(getattr(args, 'store_root', [])), args.expected_manifest_hash)
    for error in result.errors:
        print(f'ERROR {error}', file=sys.stderr)
    for warning in result.warnings:
        print(f'WARN {warning}', file=sys.stderr)
    return 0 if result.ok else 1
