"""Deploy CLI wiring: one image becomes one validated Detection Event."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from contractcheck.detect_event import DetectionEventValidator
from contractcheck.loader import SchemaStore
from contractcheck.package import AdapterSpec

from deploy import cli
from deploy.registry import RuntimeRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PACKAGE = REPO_ROOT / 'contracts' / 'examples' / 'model-package'


class FixtureAdapter:
    """Test-only double; production support lives in deploy.adapters."""

    def __init__(self, detections=None):
        self.detections = detections or []

    def open(self, package):
        self.package = package
        self.opened = True

    def infer(self, tensor, transform):
        return list(self.detections)

    def close(self):
        self.opened = False


def registry_with(detections=None) -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.register(AdapterSpec(id='fixture', version='1', formats=('fixture',),
                                  config_schema={'type': 'object', 'properties': {},
                                                 'additionalProperties': False}),
                      lambda: FixtureAdapter(detections))
    return registry


@pytest.fixture
def package_dir(tmp_path) -> Path:
    destination = tmp_path / 'package'
    shutil.copytree(EXAMPLE_PACKAGE, destination)
    return destination


def image_file(tmp_path, size=(40, 30)) -> Path:
    path = tmp_path / 'input.png'
    Image.new('RGB', size, (12, 34, 56)).save(path)
    return path


def test_detect_writes_one_validated_event(tmp_path, package_dir, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_registry', registry_with)
    status = cli.main(['detect', '--package', str(package_dir), '--image', str(image_file(tmp_path))])
    captured = capsys.readouterr()
    assert status == 0
    event = json.loads(captured.out.strip())
    result = DetectionEventValidator(SchemaStore()).validate_document(event, package=None)
    assert result.ok, result.errors
    assert event['detections'] == []
    assert event['image'] == {'width': 40, 'height': 30}
    assert event['model']['package_id'] == 'shapes-package-001'
    assert event['source']['kind'] == 'image_file'


def test_detect_reports_detections_in_original_coordinates(tmp_path, package_dir, capsys, monkeypatch):
    from deploy.types import Detection

    detections = [Detection(bbox=(1.0, 2.0, 20.0, 25.0), class_id='circle', confidence=0.75)]
    monkeypatch.setattr(cli, 'build_registry', lambda: registry_with(detections))
    status = cli.main(['detect', '--package', str(package_dir), '--image', str(image_file(tmp_path))])
    event = json.loads(capsys.readouterr().out.strip())
    assert status == 0
    assert event['detections'][0]['bbox'] == [1.0, 2.0, 20.0, 25.0]
    assert event['detections'][0]['label'] == 'Circle'
    assert DetectionEventValidator(SchemaStore()).validate_document(event, package=None).ok


def test_corrupt_image_is_an_error_not_an_empty_event(tmp_path, package_dir, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_registry', registry_with)
    broken = tmp_path / 'broken.png'
    broken.write_bytes(b'not an image')
    status = cli.main(['detect', '--package', str(package_dir), '--image', str(broken)])
    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ''
    assert 'ERROR' in captured.err


def test_unknown_adapter_is_reported(tmp_path, package_dir, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_registry', RuntimeRegistry)
    status = cli.main(['detect', '--package', str(package_dir), '--image', str(image_file(tmp_path))])
    captured = capsys.readouterr()
    assert status == 1
    assert 'unsupported_adapter' in captured.err


def test_expected_manifest_hash_is_enforced(tmp_path, package_dir, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_registry', registry_with)
    status = cli.main(['detect', '--package', str(package_dir), '--image', str(image_file(tmp_path)),
                       '--expected-manifest-hash', 'f' * 64])
    assert status == 1
    assert 'manifest_hash_expected' in capsys.readouterr().err
