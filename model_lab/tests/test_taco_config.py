"""The TACO ontology config must classify every category the dataset publishes."""
from __future__ import annotations

from pathlib import Path

from model_lab.ingest import load_ontology_config

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'taco_ontology.json'

# Category names published by the TACO dataset (COCO "categories"), pinned here so
# a typo in the mapping config cannot silently drop a class at ingest time.
TACO_CATEGORIES = [
    'Cigarette', 'Unlabeled litter', 'Plastic film', 'Clear plastic bottle', 'Other plastic',
    'Other plastic wrapper', 'Drink can', 'Plastic bottle cap', 'Plastic straw', 'Broken glass',
    'Styrofoam piece', 'Glass bottle', 'Disposable plastic cup', 'Pop tab', 'Other carton',
    'Normal paper', 'Metal bottle cap', 'Plastic lid', 'Paper cup', 'Corrugated carton',
    'Aluminium foil', 'Single-use carrier bag', 'Other plastic bottle', 'Drink carton', 'Tissues',
    'Crisp packet', 'Disposable food container', 'Plastic utensils', 'Food Can', 'Garbage bag',
    'Meal carton', 'Rope & strings', 'Paper bag', 'Scrap metal', 'Foam food container', 'Foam cup',
    'Magazine paper', 'Wrapping paper', 'Egg carton', 'Aerosol', 'Metal lid', 'Spread tub',
    'Food waste', 'Squeezable tube', 'Shoe', 'Glass cup', 'Glass jar', 'Aluminium blister pack',
    'Other plastic container', 'Toilet tube', 'Six pack rings', 'Paper straw', 'Plastic glooves',
    'Plastified paper bag', 'Tupperware', 'Polypropylene bag', 'Pizza box', 'Battery', 'Other plastic cup',
    'Carded blister pack',
]


def test_every_taco_category_is_mapped_to_the_declared_ontology():
    ontology, mapping, ignore = load_ontology_config(CONFIG)
    class_ids = [entry['id'] for entry in ontology['classes']]
    assert class_ids == ['plastic', 'metal', 'glass', 'paper', 'other']
    assert len(TACO_CATEGORIES) == 60
    assert set(mapping) == set(TACO_CATEGORIES), sorted(set(TACO_CATEGORIES) ^ set(mapping))
    assert not ignore
    assert {value for value in mapping.values()} <= set(class_ids)


def test_every_ontology_class_receives_at_least_one_category():
    ontology, mapping, _ = load_ontology_config(CONFIG)
    used = set(mapping.values())
    assert used == {entry['id'] for entry in ontology['classes']}
