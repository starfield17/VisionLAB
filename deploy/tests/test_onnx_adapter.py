"""ONNX adapter: declared layout, strict validation and preprocessing inversion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from contractcheck.package import ValidatedPackage

from deploy.adapters.onnx_detector import ADAPTER_SPEC, OnnxDetectorAdapter
from deploy.types import Transform

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_FILE = REPO_ROOT / 'deploy' / 'adapters' / 'specs' / 'yolo26_onnx.adapter.json'
MAX_DETECTIONS = 4


class _Meta:
    def __init__(self, name, shape, type_='tensor(float)'):
        self.name = name
        self.shape = list(shape)
        self.type = type_


class FakeSession:
    def __init__(self, output, *, input_name='images', input_shape=(1, 3, 64, 64),
                 output_shape=(1, MAX_DETECTIONS, 6), input_type='tensor(float)'):
        values = np.asarray(output, dtype=np.float32).ravel()
        self._output = np.resize(values, int(np.prod(output_shape))).reshape(output_shape)
        self._input = _Meta(input_name, input_shape, input_type)
        self._output_meta = _Meta('output0', output_shape)
        self.calls = 0

    def get_inputs(self):
        return [self._input]

    def get_outputs(self):
        return [self._output_meta]

    def run(self, _names, _feed):
        self.calls += 1
        return [self._output]


def rows(*entries):
    values = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(MAX_DETECTIONS)]
    for index, entry in enumerate(entries):
        values[index] = list(entry)
    return values


def package(tmp_path, **overrides) -> ValidatedPackage:
    manifest = {
        'model': {'artifact': {'path': 'model.onnx', 'sha256': 'a' * 64}, 'format': 'onnx',
                  'adapter': {'id': 'yolo26-onnx-detections', 'version': '1',
                              'config': overrides.pop('config', {'decoder': 'end2end_detections_v1',
                                                                 'max_detections': MAX_DETECTIONS})}},
    }
    labels = {'schema_version': '1.0.0', 'ontology_id': 'fixture', 'ontology_version': '1',
              'classes': [{'output_index': 0, 'id': 'plastic', 'label': 'plastic'},
                          {'output_index': 1, 'id': 'metal', 'label': 'metal'}]}
    preprocessing = {'schema_version': '1.0.0', 'input_name': 'images', 'dtype': 'float32',
                     'layout': 'NCHW', 'color_space': 'RGB', 'width': 64, 'height': 64,
                     'resize': {'mode': 'stretch', 'interpolation': 'bilinear'},
                     'normalization': {'scale': 1 / 255, 'mean': [0, 0, 0], 'std': [1, 1, 1]}}
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'graph')
    return ValidatedPackage(root=tmp_path, manifest_sha256='b' * 64, manifest=manifest,
                            labels=labels, preprocessing=preprocessing, model_path=model_path)


def adapter_for(session) -> OnnxDetectorAdapter:
    return OnnxDetectorAdapter(session_factory=lambda _path: session)


def test_committed_adapter_spec_matches_the_module():
    document = json.loads(SPEC_FILE.read_text())
    assert document['id'] == ADAPTER_SPEC.id
    assert document['version'] == ADAPTER_SPEC.version
    assert document['formats'] == list(ADAPTER_SPEC.formats)
    assert document['config_schema'] == ADAPTER_SPEC.config_schema


def test_decodes_detections_and_inverts_the_stretch_transform(tmp_path):
    session = FakeSession(rows([10, 10, 20, 20, 0.9, 1], [0, 0, 8, 8, 0.4, 0]))
    adapter = adapter_for(session)
    adapter.open(package(tmp_path))
    detections = adapter.infer(np.zeros((1, 3, 64, 64), dtype=np.float32), Transform(128, 96, 64, 64))
    assert session.calls == 1
    assert [detection.class_id for detection in detections] == ['metal', 'plastic']
    assert detections[0].bbox == (20.0, 15.0, 40.0, 30.0)
    assert detections[0].confidence == pytest.approx(0.9)
    adapter.close()
    with pytest.raises(RuntimeError):
        adapter.infer(np.zeros((1, 3, 64, 64), dtype=np.float32), Transform(128, 96, 64, 64))


def test_zero_area_rows_are_discarded(tmp_path):
    session = FakeSession(rows([4, 4, 4, 4, 0.8, 0]))
    adapter = adapter_for(session)
    adapter.open(package(tmp_path))
    assert adapter.infer(np.zeros((1, 3, 64, 64), dtype=np.float32), Transform(64, 64, 64, 64)) == []


@pytest.mark.parametrize('entry', [
    [1, 1, 5, 5, 0.9, 7],
    [1, 1, 5, 5, 1.4, 0],
    [1, 1, 5, 5, float('nan'), 0],
    [5, 1, 1, 5, 0.9, 0],
])
def test_invalid_rows_are_rejected(tmp_path, entry):
    session = FakeSession(rows(entry))
    adapter = adapter_for(session)
    adapter.open(package(tmp_path))
    with pytest.raises(ValueError):
        adapter.infer(np.zeros((1, 3, 64, 64), dtype=np.float32), Transform(64, 64, 64, 64))


def test_nan_inside_the_model_output_is_rejected(tmp_path):
    session = FakeSession(rows([1, 1, 5, 5, 0.9, 0]))
    session._output[0, 1, 3] = np.nan
    adapter = adapter_for(session)
    adapter.open(package(tmp_path))
    with pytest.raises(ValueError, match='nonfinite'):
        adapter.infer(np.zeros((1, 3, 64, 64), dtype=np.float32), Transform(64, 64, 64, 64))


def test_preprocessed_tensor_must_match_the_model_input(tmp_path):
    adapter = adapter_for(FakeSession(rows()))
    adapter.open(package(tmp_path))
    with pytest.raises(ValueError, match='does not match the model input'):
        adapter.infer(np.zeros((1, 3, 32, 32), dtype=np.float32), Transform(64, 64, 64, 64))
    with pytest.raises(ValueError, match='nonfinite'):
        adapter.infer(np.full((1, 3, 64, 64), np.nan, dtype=np.float32), Transform(64, 64, 64, 64))


@pytest.mark.parametrize('session_kwargs', [
    {'input_name': 'input'},
    {'input_shape': (1, 3, 32, 32)},
    {'output_shape': (1, MAX_DETECTIONS, 7)},
    {'input_type': 'tensor(uint8)'},
])
def test_unknown_graph_shape_is_refused_at_open(tmp_path, session_kwargs):
    adapter = adapter_for(FakeSession(rows(), **session_kwargs))
    with pytest.raises(ValueError):
        adapter.open(package(tmp_path))


def test_unknown_decoder_config_is_refused(tmp_path):
    adapter = adapter_for(FakeSession(rows()))
    with pytest.raises(ValueError, match='unsupported decoder'):
        adapter.open(package(tmp_path, config={'decoder': 'raw_tensor', 'max_detections': MAX_DETECTIONS}))
    adapter = adapter_for(FakeSession(rows()))
    with pytest.raises(ValueError, match='exactly decoder and max_detections'):
        adapter.open(package(tmp_path, config={'decoder': 'end2end_detections_v1',
                                               'max_detections': MAX_DETECTIONS, 'extra': 1}))


def test_non_float32_preprocessing_is_refused(tmp_path):
    prepared = package(tmp_path)
    assert prepared.preprocessing is not None
    prepared_manifest = prepared.preprocessing.copy()
    prepared_manifest['layout'] = 'NHWC'
    adapter = adapter_for(FakeSession(rows()))
    with pytest.raises(ValueError, match='float32 NCHW'):
        adapter.open(ValidatedPackage(root=prepared.root, manifest_sha256=prepared.manifest_sha256,
                                      manifest=prepared.manifest, labels=prepared.labels,
                                      preprocessing=prepared_manifest, model_path=prepared.model_path))
