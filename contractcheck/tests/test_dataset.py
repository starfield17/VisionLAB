"""Dataset and Annotation contract: semantic negative cases from the handoff."""

from __future__ import annotations

from pathlib import Path

from contractcheck.dataset import DatasetValidator
from contractcheck.loader import SchemaStore

from conftest import read_json, rehash_annotation, rules_of, write_json


def dataset_validator(store: SchemaStore, path: Path):
    return DatasetValidator(store).validate(path)


def test_valid_standalone_dataset_passes(store: SchemaStore, valid_dataset: Path):
    result = DatasetValidator(store).validate(valid_dataset)
    assert result.ok, result.errors


def test_candidate_in_released_dataset(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["status"] = "candidate"
    ann.pop("review", None)
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "annotation_not_approved" in rules_of(result)


def test_missing_review_on_approved(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann.pop("review", None)
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "missing_review" in rules_of(result)
    assert "schema" in rules_of(result)  # the if/then in annotation schema also fires


def test_unknown_class(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["objects"][0]["class_id"] = "no-such-shape"
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "unknown_class" in rules_of(result)


def test_duplicate_ontology_class_id(store: SchemaStore, valid_dataset: Path):
    ds = read_json(valid_dataset)
    ds["ontology"]["classes"].append({"id": "circle", "label": "Circle clone"})
    write_json(valid_dataset, ds)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "duplicate_id" in rules_of(result)
    assert any("class id" in e.message for e in result.errors)


def test_duplicate_item_id(store: SchemaStore, valid_dataset: Path):
    ds = read_json(valid_dataset)
    ds["items"].append(json_copy(ds["items"][0]))
    write_json(valid_dataset, ds)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "duplicate_id" in rules_of(result)
    assert any("item id" in e.message for e in result.errors)


def json_copy(obj):
    import json as _json
    return _json.loads(_json.dumps(obj))


def test_inverted_box(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["objects"][0]["bbox"] = [40, 10, 10, 30]  # x_min > x_max
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "box_inverted" in rules_of(result)


def test_out_of_bounds_box(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["objects"][0]["bbox"] = [10, 10, 300, 30]  # wider than the 100px image
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "box_out_of_bounds" in rules_of(result)


def test_annotation_item_mismatch(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["item_id"] = "some-other-item"
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "annotation_mismatch" in rules_of(result)


def test_annotation_image_hash_mismatch(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["image_sha256"] = "f" * 64
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "annotation_mismatch" in rules_of(result)


def test_mixed_split_group(store: SchemaStore, valid_dataset: Path):
    ds = read_json(valid_dataset)
    second = json_copy(ds["items"][0])
    second["id"] = "item-002"
    second["split"] = "train"
    ds["items"].append(second)
    write_json(valid_dataset, ds)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "split_leakage" in rules_of(result)


def test_altered_annotation_hash(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["objects"][0]["bbox"] = [5, 5, 15, 15]
    write_json(annotation_path, ann)
    # deliberately NOT rehashing -> stale hash in the manifest
    result = DatasetValidator(store).validate(valid_dataset)
    assert "hash_mismatch" in rules_of(result)


def test_path_symlink_escape(store: SchemaStore, valid_dataset: Path, tmp_path: Path):
    # An image path that is a symlink pointing outside the dataset dir.
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-an-image")
    ds = read_json(valid_dataset)
    ds["items"][0]["image"]["path"] = "esc.png"
    write_json(valid_dataset, ds)
    (valid_dataset.parent / "esc.png").symlink_to(outside)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "path_escape" in rules_of(result)


def test_review_before_creation(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["created_at"] = "2026-01-02T00:00:00Z"
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert "time_order" in rules_of(result)


def test_negative_image_passes(store: SchemaStore, valid_dataset: Path):
    annotation_path = valid_dataset.parent / "annotation.json"
    ann = read_json(annotation_path)
    ann["objects"] = []  # reviewer confirmed a negative image
    write_json(annotation_path, ann)
    rehash_annotation(valid_dataset, annotation_path)
    result = DatasetValidator(store).validate(valid_dataset)
    assert result.ok, result.errors


def test_test_only_manifest_passes(store: SchemaStore, valid_dataset: Path):
    # A manifest containing only a test split is allowed.
    result = DatasetValidator(store).validate(valid_dataset)
    assert result.ok


def test_missing_image_file(store: SchemaStore, valid_dataset: Path):
    (valid_dataset.parent / "image.fixture").unlink()
    result = DatasetValidator(store).validate(valid_dataset)
    assert "artifact_missing" in rules_of(result)
