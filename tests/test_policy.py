"""Prove architectural checks detect violations rather than merely passing."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('check_policy', Path('tools/check_policy.py'))
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def test_forbidden_import_is_detected(tmp_path):
    folder = tmp_path / 'contractcheck'
    folder.mkdir()
    (folder / 'loader.py').write_text('import deploy\n')
    errors = policy.boundary_errors(tmp_path)
    assert errors == ['contractcheck.loader: forbidden external import deploy']


def test_core_cannot_import_sink_even_indirectly(tmp_path):
    folder = tmp_path / 'deploy'
    folder.mkdir()
    (folder / 'types.py').write_text('from .sinks import StdoutSink\n')
    assert policy.boundary_errors(tmp_path) == ['deploy.types: forbidden internal import deploy.sinks']


def test_disabled_tests_detected(tmp_path):
    folder = tmp_path / 'tests'
    folder.mkdir()
    (folder / 'test_disabled.py').write_text('import pytest\n@pytest.mark.skip\ndef test_bad():\n    assert True\n')
    inventory, errors = policy.test_inventory(tmp_path)
    assert inventory == {'tests/test_disabled.py::test_bad': 1}
    assert errors == ['tests/test_disabled.py: disabled test marker is forbidden']
