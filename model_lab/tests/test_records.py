import pytest

from model_lab.records import labeling_record

PROMPT = 'Locate all the instances that matches the following description: person.'


def make_record(**overrides):
    record = {
        'schema_version': '1.0.0',
        'kind': 'labeling_run',
        'id': 'run-1',
        'created_at': '2026-01-01T00:00:00Z',
        'tool': {'name': 'locate-anything.cpp', 'revision': '77376ab', 'version': 'pinned-cpp'},
        'checkpoint': {'revision': 'locate-anything-q4-k', 'sha256': 'a' * 64, 'size_bytes': 4716320320,
                       'reference': 'huggingface.co/mudler/locate-anything.cpp-gguf',
                       'license': 'nvidia-academic-non-commercial'},
        'prompt': PROMPT,
        'backend': 'cpu',
        'decode_mode': 'hybrid',
        'command': {'args': ['--mode', 'hybrid', '--threads', '2'],
                    'threads': 2, 'generation_token_limit': 256},
        'inputs': [{'role': 'canonical_image',
                    'artifact': {'path': 'images/item-000000.png', 'sha256': 'b' * 64}}],
        'outputs': [{'role': 'raw_result', 'artifact': {'path': 'results/item-000000.json', 'sha256': 'c' * 64}},
                    {'role': 'candidate', 'artifact': {'path': 'candidates/item-000000.json', 'sha256': 'd' * 64}}],
    }
    record.update(overrides)
    return record


def test_labeling_record_validates_portable_fields():
    doc = labeling_record(make_record())
    assert doc['kind'] == 'labeling_run'
    assert doc['checkpoint']['sha256'] == 'a' * 64
    assert doc['command']['generation_token_limit'] == 256
    assert doc['outputs'][1]['role'] == 'candidate'


def test_labeling_record_is_an_independent_snapshot():
    record = make_record()
    doc = labeling_record(record)
    record['prompt'] = 'tampered'
    record['outputs'].append({'role': 'x', 'artifact': {'path': 'x.json', 'sha256': 'e' * 64}})
    assert doc['prompt'] == PROMPT
    assert len(doc['outputs']) == 2


def test_labeling_record_rejects_machine_paths():
    record = make_record()
    record['command']['args'] = ['--input', '/absolute/host/secret/canonical.png']
    with pytest.raises(ValueError):
        labeling_record(record)


def test_labeling_record_rejects_bad_hash_and_timestamp():
    with pytest.raises(ValueError):
        labeling_record(make_record(checkpoint={'revision': 'q', 'sha256': 'zz', 'size_bytes': 1,
                                                'reference': 'r', 'license': 'l'}))
    with pytest.raises(ValueError):
        labeling_record(make_record(created_at='2026-01-01 00:00:00'))


def test_labeling_record_requires_portable_relative_paths():
    record = make_record()
    record['outputs'][0]['artifact']['path'] = 'C:\\results\\item.json'
    with pytest.raises(ValueError):
        labeling_record(record)


def test_labeling_record_requires_inputs_and_outputs():
    with pytest.raises(ValueError):
        labeling_record(make_record(inputs=[]))
