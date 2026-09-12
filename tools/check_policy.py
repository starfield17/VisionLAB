"""Enforce dependency boundaries and test integrity, including untracked files."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

LAYERS = {
    'contractcheck.errors': set(),
    'contractcheck.loader': {'contractcheck.errors'},
    'contractcheck.common': {'contractcheck.loader', 'contractcheck.errors'},
    'contractcheck.annotation': {'contractcheck.common'},
    'contractcheck.dataset': {'contractcheck.common', 'contractcheck.annotation', 'contractcheck.errors', 'contractcheck.loader'},
    'contractcheck.detect_event': {'contractcheck.common', 'contractcheck.errors', 'contractcheck.loader'},
    'contractcheck.package': {'contractcheck.common', 'contractcheck.annotation', 'contractcheck.dataset', 'contractcheck.errors', 'contractcheck.loader'},
    'contractcheck.cli': {'contractcheck', 'contractcheck.common', 'contractcheck.annotation', 'contractcheck.dataset', 'contractcheck.detect_event', 'contractcheck.errors', 'contractcheck.loader', 'contractcheck.package'},
    'contractcheck.__main__': {'contractcheck.cli'},
    'contractcheck': set(),
    'deploy.types': set(),
    'deploy.preprocessing': {'deploy.types'},
    'deploy.postprocessing': {'deploy.types'},
    'deploy.registry': {'deploy.types'},
    'deploy.pipeline': {'deploy.types', 'deploy.registry', 'deploy.preprocessing', 'deploy.postprocessing'},
    'deploy': {'deploy.types', 'deploy.registry', 'deploy.pipeline'},
    'deploy.sinks': {'deploy.sinks.stdout'},
    'deploy.sinks.stdout': set(),
    'deploy.adapters': {'deploy.adapters.onnx_detector'},
    'deploy.adapters.onnx_detector': {'deploy.types'},
    'deploy.sources': {'deploy.sources.image_file'},
    'deploy.sources.image_file': {'deploy.types'},
    'deploy.compose': {'deploy.adapters.onnx_detector', 'deploy.registry'},
    'deploy.cli': {'deploy.compose', 'deploy.pipeline', 'deploy.sinks.stdout', 'deploy.sources.image_file'},
    'deploy.__main__': {'deploy.cli'},
    'model_lab': set(),
    'model_lab.acquisition': set(),
    'model_lab.conversion': set(),
    'model_lab.evaluation': {'model_lab.guard', 'model_lab.runrecords'},
    'model_lab.export': {'model_lab.guard', 'model_lab.runrecords'},
    'model_lab.guard': set(),
    'model_lab.ingest': {'model_lab.acquisition', 'model_lab.runrecords'},
    'model_lab.overlay': set(),
    'model_lab.package': {'model_lab.runrecords'},
    'model_lab.preflight': set(),
    'model_lab.records': set(),
    'model_lab.runrecords': set(),
    'model_lab.runners': {'model_lab.runners.onnx_probe', 'model_lab.runners.yolo_val'},
    'model_lab.runners.onnx_probe': set(),
    'model_lab.runners.yolo_val': set(),
    'model_lab.train': set(),
    'model_lab.__main__': {'model_lab.conversion', 'model_lab.overlay', 'model_lab.guard', 'model_lab.train',
                           'model_lab.ingest', 'model_lab.preflight', 'model_lab.runrecords',
                           'model_lab.evaluation', 'model_lab.export', 'model_lab.package'},
    'contracts': set(),
    'contracts.schemas': set(),
}
EXTERNAL = {'model_lab': {'contractcheck', 'PIL', 'psutil'}, 'contractcheck': {'jsonschema', 'referencing'}, 'deploy': {'numpy', 'contractcheck'}, 'contracts': set()}
# Runtime frameworks stay inside concrete boundary modules; Core, the validator
# and Model Lab orchestration never import them.
MODULE_EXTERNAL = {
    'deploy.adapters.onnx_detector': {'numpy', 'onnxruntime', 'contractcheck'},
    'deploy.sources.image_file': {'numpy', 'PIL', 'contractcheck'},
    'model_lab.evaluation': {'contractcheck', 'yaml'},
    'model_lab.export': {'contractcheck'},
    'model_lab.ingest': {'contractcheck', 'PIL'},
    'model_lab.package': {'contractcheck'},
    'model_lab.preflight': {'contractcheck'},
    'model_lab.runrecords': {'contractcheck', 'yaml'},
    'model_lab.runners.onnx_probe': {'numpy', 'cv2', 'onnx', 'onnxruntime', 'PIL', 'ultralytics'},
    'model_lab.runners.yolo_val': {'ultralytics'},
}


def imports(tree, module, is_package):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = module.split('.') if is_package else module.split('.')[:-1]
                base = '.'.join(parent[:len(parent) - node.level + 1])
                if node.module:
                    yield base + '.' + node.module
                else:
                    for alias in node.names:
                        yield base if alias.name.startswith('__') else base + '.' + alias.name
            elif node.module:
                yield node.module


def boundary_errors(root: Path) -> list[str]:
    errors = []
    for package, allowed in EXTERNAL.items():
        for path in (root / package).rglob('*.py'):
            if 'tests' in path.parts:
                continue
            rel = path.relative_to(root).with_suffix('')
            module = '.'.join(rel.parts).removesuffix('.__init__')
            if module not in LAYERS:
                errors.append(f'{rel}: module needs an explicit dependency policy')
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for target in imports(tree, module, path.name == '__init__.py'):
                top = target.split('.')[0]
                if top == package:
                    if target not in LAYERS[module]:
                        errors.append(f'{module}: forbidden internal import {target}')
                elif top not in sys.stdlib_module_names and top not in MODULE_EXTERNAL.get(module, allowed):
                    errors.append(f'{module}: forbidden external import {target}')
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and (getattr(node.func, 'id', '') in ('__import__', 'eval', 'exec') or getattr(node.func, 'attr', '') == 'import_module'):
                    errors.append(f'{module}: dynamic code loading bypasses static composition')
    return errors


def test_inventory(root: Path):
    inventory = {}
    errors = []
    for pattern in ('contractcheck/tests/test_*.py', 'deploy/tests/test_*.py', 'model_lab/tests/test_*.py', 'tests/test_*.py'):
        for path in root.glob(pattern):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    key = f'{path.relative_to(root).as_posix()}::{node.name}'
                    inventory[key] = sum(isinstance(n, ast.Assert) for n in ast.walk(node))
                if isinstance(node, ast.Attribute) and node.attr in ('skip', 'skipif', 'xfail'):
                    errors.append(f'{path.relative_to(root)}: disabled test marker is forbidden')
    return inventory, errors


def check(root: Path) -> list[str]:
    errors = boundary_errors(root)
    current, disabled = test_inventory(root)
    errors.extend(disabled)
    baseline = json.loads((root / 'tools/test_baseline.json').read_text())
    for name, assertions in baseline.items():
        if name not in current or current[name] < assertions:
            errors.append(f'{name}: baseline test missing or assertions reduced')
    return errors


if __name__ == '__main__':
    failures = check(Path(__file__).resolve().parents[1])
    print('\n'.join(failures) if failures else 'Dependency boundaries and test integrity passed')
    raise SystemExit(bool(failures))
