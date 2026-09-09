"""Shared fixtures and mutation helpers for contractcheck tests.

The contract fixture under contracts/examples/model-package is the valid
baseline; negative tests copy it into a tmp path and mutate it. Tests never
touch the original fixture files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from contractcheck.loader import SchemaStore, default_schema_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PKG = REPO_ROOT / "contracts" / "examples" / "model-package"
EXAMPLE_DATA = EXAMPLE_PKG / "data"
EXAMPLE_EVENTS = REPO_ROOT / "contracts" / "examples"


@pytest.fixture(scope="session")
def store() -> SchemaStore:
    return SchemaStore(default_schema_dir())


@pytest.fixture(scope="session")
def example_package() -> Path:
    return EXAMPLE_PKG


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """A standalone copy of the example dataset (dataset.json + annotation + image)."""
    dst = tmp_path / "dataset"
    copy_tree(EXAMPLE_DATA, dst)
    return dst / "dataset.json"


@pytest.fixture
def valid_package(tmp_path: Path) -> Path:
    """A copy of the example model package that preflight passes when the
    fixture adapter is declared."""
    dst = tmp_path / "package"
    copy_tree(EXAMPLE_PKG, dst)
    # Synthetic provenance fixture, structurally auditable but not a trained model.
    for stage in ('evaluation', 'export'):
        record_path = dst / 'records' / f'{stage}.json'
        record = read_json(record_path)
        record['inputs'].append({'role': 'model', 'artifact': {'path': 'model.fixture', 'sha256': sha256(dst / 'model.fixture')}})
        if stage == 'evaluation':
            record['config'] = {'split': 'test', 'sample_count': 1,
                                'metric_definitions': {'example_metric': 'Synthetic fixture metric'},
                                'thresholds': {'confidence': 0.25}}
        write_json(record_path, record)
    rehash_package(dst)
    return dst


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in Path(src).rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def rehash_annotation(dataset_path: Path, annotation_path: Path) -> None:
    """Recompute the annotation's hash inside the dataset manifest."""
    ds = read_json(dataset_path)
    ds["items"][0]["annotation"]["artifact"]["sha256"] = sha256(annotation_path)
    write_json(dataset_path, ds)


def rehash_package(pkg_dir: Path) -> None:
    """Recompute every directly referenced artifact hash inside manifest.json."""
    manifest_path = pkg_dir / "manifest.json"
    manifest = read_json(manifest_path)
    refs = [
        manifest["model"]["artifact"],
        manifest["labels"],
        manifest["preprocessing"],
        manifest["trace"]["dataset"]["artifact"],
        manifest["trace"]["training"]["artifact"],
        manifest["trace"]["evaluation"]["artifact"],
        manifest["trace"]["export"]["artifact"],
    ]
    for ref in refs:
        ref["sha256"] = sha256(pkg_dir / ref["path"])
    write_json(manifest_path, manifest)


def rules_of(result) -> set[str]:
    return {error.rule for error in result.errors}
