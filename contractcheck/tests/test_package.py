"""Model Package contract: preflight and audit semantic negative cases."""

from __future__ import annotations

from pathlib import Path

from contractcheck.loader import SchemaStore
from contractcheck.package import AdapterRegistry, PackageValidator

from conftest import read_json, rehash_package, rules_of, sha256, write_json

FIXTURE_ADAPTER = "fixture@1"


def adapters() -> AdapterRegistry:
    reg = AdapterRegistry()
    adapter_id, version = FIXTURE_ADAPTER.split("@")
    reg.register(adapter_id, version, formats=('fixture',), config_schema={'type': 'object', 'properties': {}, 'additionalProperties': False})
    return reg


def test_valid_preflight_passes(store: SchemaStore, valid_package: Path):
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert result.ok, result.errors


def test_valid_audit_passes(store: SchemaStore, valid_package: Path):
    result = PackageValidator(store, adapters()).validate(valid_package, mode="audit")
    assert result.ok, result.errors


def test_unsupported_adapter_fails_preflight(store: SchemaStore, valid_package: Path):
    result = PackageValidator(store, AdapterRegistry()).validate(
        valid_package, mode="preflight")
    assert "unsupported_adapter" in rules_of(result)


def test_altered_model_artifact_hash(store: SchemaStore, valid_package: Path):
    (valid_package / "model.fixture").write_bytes(b"tampered")
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "hash_mismatch" in rules_of(result)


def test_altered_labels_schema_and_hash(store: SchemaStore, valid_package: Path):
    labels = read_json(valid_package / "labels.json")
    labels["classes"][0]["label"] = "Wrong label"
    write_json(valid_package / "labels.json", labels)
    # manifest still holds the old hash -> altered artifact hash
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "hash_mismatch" in rules_of(result)


def test_labels_ontology_mismatch(store: SchemaStore, valid_package: Path):
    labels = read_json(valid_package / "labels.json")
    labels["classes"][0]["id"] = "not-in-ontology"
    write_json(valid_package / "labels.json", labels)
    rehash_package(valid_package)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "labels_ontology_mismatch" in rules_of(result)


def test_duplicate_output_index_in_labels(store: SchemaStore, valid_package: Path):
    labels = read_json(valid_package / "labels.json")
    labels["classes"].append(labels["classes"][0])
    write_json(valid_package / "labels.json", labels)
    rehash_package(valid_package)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "duplicate_id" in rules_of(result)


def test_record_identity_mismatch(store: SchemaStore, valid_package: Path):
    training = read_json(valid_package / "records" / "training.json")
    training["id"] = "wrong-training-id"
    write_json(valid_package / "records" / "training.json", training)
    rehash_package(valid_package)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "record_identity" in rules_of(result)


def test_record_stage_mismatch(store: SchemaStore, valid_package: Path):
    training = read_json(valid_package / "records" / "training.json")
    training["stage"] = "export"
    write_json(valid_package / "records" / "training.json", training)
    rehash_package(valid_package)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "record_identity" in rules_of(result)


def test_expected_manifest_hash_mismatch(store: SchemaStore, valid_package: Path):
    result = PackageValidator(store, adapters()).validate(
        valid_package, mode="preflight", expected_manifest_hash="f" * 64)
    assert "manifest_hash_expected" in rules_of(result)


def test_path_symlink_escape_in_labels(store: SchemaStore, valid_package: Path, tmp_path: Path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"escape": true}', encoding="utf-8")
    manifest = read_json(valid_package / "manifest.json")
    manifest["labels"]["path"] = "esc.json"
    write_json(valid_package / "manifest.json", manifest)
    (valid_package / "esc.json").symlink_to(outside)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "path_escape" in rules_of(result)


def test_unsupported_schema_version_in_manifest(store: SchemaStore, valid_package: Path):
    manifest = read_json(valid_package / "manifest.json")
    manifest["schema_version"] = "2.0.0"
    write_json(valid_package / "manifest.json", manifest)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert not result.ok
    assert any("schema_version" in e.message for e in result.errors)


def test_manifest_self_reference_is_a_hash_cycle(store: SchemaStore, valid_package: Path):
    # A record included by the manifest must not reference the manifest itself.
    training = read_json(valid_package / "records" / "training.json")
    training["inputs"].append({
        "role": "manifest",
        "artifact": {"path": "manifest.json", "sha256": "e" * 64},
    })
    write_json(valid_package / "records" / "training.json", training)
    rehash_package(valid_package)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert "hash_cycle" in rules_of(result)


def test_audit_detects_unresolved_run_record_artifact(store: SchemaStore, valid_package: Path):
    # A run-record-only input that exists nowhere; preflight does not look there.
    training = read_json(valid_package / "records" / "training.json")
    training["inputs"].append({
        "role": "intermediate",
        "artifact": {"path": "intermediate.bin", "sha256": "e" * 64},
    })
    write_json(valid_package / "records" / "training.json", training)
    rehash_package(valid_package)
    preflight = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert preflight.ok, preflight.errors
    audit = PackageValidator(store, adapters()).validate(
        valid_package, mode="audit", store_roots=())
    assert "artifact_unresolved" in rules_of(audit)


def test_audit_resolves_artifact_from_store(store: SchemaStore, valid_package: Path,
                                            tmp_path: Path):
    intermediate = valid_package.parent / "store" / "intermediate.bin"
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    intermediate.write_bytes(b"intermediate-data")
    training = read_json(valid_package / "records" / "training.json")
    training["inputs"].append({
        "role": "intermediate",
        "artifact": {"path": "intermediate.bin", "sha256": sha256(intermediate)},
    })
    write_json(valid_package / "records" / "training.json", training)
    rehash_package(valid_package)

    without_store = PackageValidator(store, adapters()).validate(
        valid_package, mode="audit", store_roots=())
    assert "artifact_unresolved" in rules_of(without_store)

    with_store = PackageValidator(store, adapters()).validate(
        valid_package, mode="audit",
        store_roots=(valid_package.parent / "store",))
    assert with_store.ok, with_store.errors


def test_audit_export_output_must_match_model_hash(store: SchemaStore, valid_package: Path):
    export = read_json(valid_package / "records" / "export.json")
    export["outputs"][0]["artifact"]["sha256"] = "d" * 64
    write_json(valid_package / "records" / "export.json", export)
    rehash_package(valid_package)
    audit = PackageValidator(store, adapters()).validate(valid_package, mode="audit")
    assert "export_mismatch" in rules_of(audit)


def test_preflight_does_not_execute_model_code(store: SchemaStore, valid_package: Path):
    # The model file is a text placeholder; preflight only hashes it.
    (valid_package / "model.fixture").write_text("not a real model", encoding="utf-8")
    rehash_package(valid_package)
    result = PackageValidator(store, adapters()).validate(valid_package, mode="preflight")
    assert result.ok, result.errors
