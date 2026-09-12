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


def test_model_lab_cannot_depend_on_deploy(tmp_path):
    folder = tmp_path / 'model_lab'
    folder.mkdir()
    (folder / 'conversion.py').write_text('import deploy\n')
    assert policy.boundary_errors(tmp_path) == ['model_lab.conversion: forbidden external import deploy']


def test_core_cannot_import_the_runtime(tmp_path):
    folder = tmp_path / 'deploy'
    folder.mkdir()
    (folder / 'pipeline.py').write_text('import onnxruntime\n')
    assert policy.boundary_errors(tmp_path) == ['deploy.pipeline: forbidden external import onnxruntime']


def test_core_sinks_cannot_import_an_image_library(tmp_path):
    folder = tmp_path / 'deploy' / 'sinks'
    folder.mkdir(parents=True)
    (folder / 'stdout.py').write_text('from PIL import Image\n')
    assert policy.boundary_errors(tmp_path) == ['deploy.sinks.stdout: forbidden external import PIL']


def test_model_lab_orchestration_cannot_import_a_training_framework(tmp_path):
    folder = tmp_path / 'model_lab'
    folder.mkdir()
    (folder / 'evaluation.py').write_text('from ultralytics import YOLO\n')
    assert policy.boundary_errors(tmp_path) == ['model_lab.evaluation: forbidden external import ultralytics']


def test_guarded_runner_may_import_the_training_framework(tmp_path):
    folder = tmp_path / 'model_lab' / 'runners'
    folder.mkdir(parents=True)
    (folder / 'yolo_val.py').write_text('from ultralytics import YOLO\n')
    assert policy.boundary_errors(tmp_path) == []


def test_contractcheck_cannot_import_numpy(tmp_path):
    folder = tmp_path / 'contractcheck'
    folder.mkdir()
    (folder / 'dataset.py').write_text('import numpy\n')
    assert policy.boundary_errors(tmp_path) == ['contractcheck.dataset: forbidden external import numpy']
