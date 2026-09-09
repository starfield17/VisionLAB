"""Detection Event contract: semantic negative cases."""

from __future__ import annotations

from pathlib import Path

from contractcheck.detect_event import DetectionEventValidator
from contractcheck.loader import SchemaStore

from conftest import EXAMPLE_EVENTS, read_json, rules_of, write_json


def load_event(name: str, tmp_path: Path) -> Path:
    src = EXAMPLE_EVENTS / f"{name}.json"
    dst = tmp_path / f"{name}.json"
    write_json(dst, read_json(src))
    return dst


def test_valid_events_pass(store: SchemaStore, tmp_path: Path):
    for name in ("detection-event", "empty-event"):
        result = DetectionEventValidator(store).validate(load_event(name, tmp_path))
        assert result.ok, result.errors


def test_emitted_before_captured(store: SchemaStore, tmp_path: Path):
    path = load_event("detection-event", tmp_path)
    event = read_json(path)
    event["captured_at"] = "2026-01-01T00:00:05Z"
    write_json(path, event)
    result = DetectionEventValidator(store).validate(path)
    assert "time_order" in rules_of(result)


def test_out_of_bounds_box(store: SchemaStore, tmp_path: Path):
    path = load_event("detection-event", tmp_path)
    event = read_json(path)
    event["detections"][0]["bbox"] = [10, 10, 500, 30]
    write_json(path, event)
    result = DetectionEventValidator(store).validate(path)
    assert "box_out_of_bounds" in rules_of(result)


def test_inverted_box(store: SchemaStore, tmp_path: Path):
    path = load_event("detection-event", tmp_path)
    event = read_json(path)
    event["detections"][0]["bbox"] = [30, 10, 10, 30]
    write_json(path, event)
    result = DetectionEventValidator(store).validate(path)
    assert "box_inverted" in rules_of(result)


def test_non_utc_timestamp_rejected(store: SchemaStore, tmp_path: Path):
    path = load_event("detection-event", tmp_path)
    event = read_json(path)
    event["captured_at"] = "2026-01-01T08:00:00+08:00"
    write_json(path, event)
    result = DetectionEventValidator(store).validate(path)
    assert "utc_timestamp" in rules_of(result)


def test_missing_event_id_fails_schema(store: SchemaStore, tmp_path: Path):
    path = load_event("detection-event", tmp_path)
    event = read_json(path)
    del event["event_id"]
    write_json(path, event)
    result = DetectionEventValidator(store).validate(path)
    assert "schema" in rules_of(result)
