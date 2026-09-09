"""Layer 1: strict JSON parsing and schema validation."""

from __future__ import annotations

import pytest

from contractcheck import common
from contractcheck.errors import ValidationResult
from contractcheck.loader import StrictJSONError, SchemaStore, loads_strict


def test_duplicate_keys_rejected():
    with pytest.raises(StrictJSONError, match="duplicate key"):
        loads_strict('{"a": 1, "a": 2}')


def test_non_finite_numbers_rejected():
    for blob in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(StrictJSONError, match="non-finite"):
            loads_strict(f"[{blob}]")


def test_valid_json_parses():
    assert loads_strict('{"a": [1, 2.5]}') == {"a": [1, 2.5]}


def test_schema_store_loads_all_contract_schemas(store: SchemaStore):
    expected = {
        "annotation.schema.json", "dataset.schema.json",
        "detection-event.schema.json", "labels.schema.json",
        "model-package.schema.json", "ontology.schema.json",
        "preprocessing.schema.json", "run-record.schema.json",
    }
    assert expected <= set(store.schemas)


def test_unsupported_schema_version_rejected(store: SchemaStore, valid_dataset):
    import json as _json
    doc = _json.loads(valid_dataset.read_text(encoding="utf-8"))
    doc["schema_version"] = "0.9.0"
    _json.dump(doc, valid_dataset.open("w"))
    from contractcheck.dataset import DatasetValidator
    result = DatasetValidator(store).validate(valid_dataset)
    assert not result.ok
    assert any("schema_version" in e.message and "1.0.0" in e.message for e in result.errors)


def test_relative_posix_path_rules():
    assert common.validate_relative_posix_path("a/b/c.json")
    assert not common.validate_relative_posix_path("/abs/path.json")
    assert not common.validate_relative_posix_path("../up.json")
    assert not common.validate_relative_posix_path("a/./b.json")
    assert not common.validate_relative_posix_path("a//b.json")
    assert not common.validate_relative_posix_path("a\\\\b.json")
    assert not common.validate_relative_posix_path("C:/x.json")


def test_utc_timestamp_rules():
    result = ValidationResult()
    common.check_utc_timestamp(result, "2026-01-01T00:00:00Z", "d", "t")
    assert result.ok
    common.check_utc_timestamp(result, "2026-01-01T08:00:00+08:00", "d", "t")
    common.check_utc_timestamp(result, "2026-01-01T00:00:00+00:00", "d", "t")
    common.check_utc_timestamp(result, "01/01/2026", "d", "t")
    assert {e.rule for e in result.errors} == {"utc_timestamp"}
    assert len(result.errors) == 3
