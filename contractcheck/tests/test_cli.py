"""End-to-end CLI behavior: exit codes and output routing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT, copy_tree

PYTHON = sys.executable
MODULE = ["-m", "contractcheck"]


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, *MODULE, *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validate_valid_dataset_exit_zero():
    proc = run_cli(
        "validate",
        "contracts/examples/model-package/data/dataset.json",
    )
    assert proc.returncode == 0, proc.stderr


def test_validate_valid_event_exit_zero():
    proc = run_cli("validate", "contracts/examples/empty-event.json")
    assert proc.returncode == 0, proc.stderr


def test_validate_labels_is_not_detected_as_ontology():
    proc = run_cli("validate", "contracts/examples/model-package/labels.json")
    assert proc.returncode == 0, proc.stderr


def test_validate_invalid_document_exit_one(tmp_path: Path):
    dataset = tmp_path / "bad"
    copy_tree(REPO_ROOT / "contracts/examples/model-package/data", dataset)
    path = dataset / "dataset.json"
    import json
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["schema_version"] = "0.9.0"
    path.write_text(json.dumps(doc), encoding="utf-8")
    proc = run_cli("validate", str(path))
    assert proc.returncode == 1
    assert "schema" in proc.stderr


def test_validate_unknown_document_exit_one(tmp_path: Path):
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"random": true}', encoding="utf-8")
    proc = run_cli("validate", str(unknown))
    assert proc.returncode == 1
    assert "unknown_document" in proc.stderr


def test_validate_duplicate_keys_exit_one(tmp_path: Path):
    bad = tmp_path / "dup.json"
    bad.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    proc = run_cli("validate", str(bad))
    assert proc.returncode == 1
    assert "strict_json" in proc.stderr


def test_package_preflight_requires_adapter(tmp_path: Path):
    pkg = tmp_path / "package"
    copy_tree(REPO_ROOT / "contracts/examples/model-package", pkg)
    proc = run_cli("package", "preflight", str(pkg))
    assert proc.returncode == 1
    assert "unsupported_adapter" in proc.stderr
    proc_ok = run_cli("package", "preflight", str(pkg), "--adapter-spec", str(adapter_spec(tmp_path)))
    assert proc_ok.returncode == 0, proc_ok.stderr


def test_package_audit_with_store_root(tmp_path: Path, valid_package):
    pkg = tmp_path / "package"
    store_dir = tmp_path / "store"
    pkg = valid_package
    store_dir.mkdir()
    intermediate = store_dir / "extra.bin"
    intermediate.write_bytes(b"x")
    import hashlib
    digest = hashlib.sha256(intermediate.read_bytes()).hexdigest()
    import json as _json
    training_path = pkg / "records" / "training.json"
    training = _json.loads(training_path.read_text(encoding="utf-8"))
    training["inputs"].append(
        {"role": "intermediate", "artifact": {"path": "extra.bin", "sha256": digest}}
    )
    training_path.write_text(_json.dumps(training), encoding="utf-8")
    from conftest import rehash_package
    rehash_package(pkg)
    proc = run_cli("package", "audit", str(pkg), "--adapter-spec", str(adapter_spec(tmp_path)),
                   "--store-root", str(store_dir))
    assert proc.returncode == 0, proc.stderr
    proc_missing = run_cli("package", "audit", str(pkg), "--adapter-spec", str(adapter_spec(tmp_path)))
    assert proc_missing.returncode == 1
    assert "artifact_unresolved" in proc_missing.stderr


def adapter_spec(tmp_path):
    import json
    path = tmp_path / 'adapter.json'
    path.write_text(json.dumps({'id': 'fixture', 'version': '1', 'formats': ['fixture'],
                               'config_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}}))
    return path


def test_legacy_adapter_is_configuration_error(tmp_path):
    proc = run_cli('package', 'preflight', 'contracts/examples/model-package', '--adapter', 'fixture@1')
    assert proc.returncode == 2
    assert '--adapter-spec' in proc.stderr
