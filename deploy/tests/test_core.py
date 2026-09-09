"""Behavioral tests use real numerical code and port test doubles only."""
import copy
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contractcheck.package import AdapterSpec
from deploy import Detection, Frame, Pipeline, RuntimeRegistry, SourceInfo, Transform
from deploy.postprocessing import postprocess
from deploy.preprocessing import preprocess
from deploy.sinks import StdoutSink

T = '2026-01-01T00:00:00Z'
SPEC = AdapterSpec('fixture', '1', ('fixture',), {'type': 'object', 'properties': {}, 'additionalProperties': False})
PRE = json.loads(Path('contracts/examples/model-package/preprocessing.json').read_text())
POST = {'confidence_threshold': 0.25, 'nms': {'mode': 'class_aware', 'iou_threshold': 0.5}}


def test_bilinear_half_pixel_without_rounding():
    spec = copy.deepcopy(PRE)
    spec.update(width=4, height=1, layout='NHWC')
    spec['normalization'] = {'scale': 1, 'mean': [0, 0, 0], 'std': [1, 1, 1]}
    image = np.array([[[0, 4, 8], [2, 6, 10]]], dtype=np.uint8)
    tensor, transform = preprocess(image, spec)
    np.testing.assert_array_equal(tensor[0, 0, :, 0], [0, 0.5, 1.5, 2])
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 1, 4, 3)
    assert transform.original_box((0, 0, 4, 1)) == (0, 0, 2, 1)


def test_color_normalization_layout_and_single_pixel():
    spec = copy.deepcopy(PRE)
    spec.update(width=2, height=2, color_space='BGR')
    spec['normalization'] = {'scale': 0.5, 'mean': [1, 2, 3], 'std': [2, 1, 0.5]}
    tensor, _ = preprocess(np.array([[[10, 20, 30]]], dtype=np.uint8), spec)
    np.testing.assert_array_equal(tensor[0, :, 0, 0], [7, 8, 4])
    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor[0, :, 1, 1], [7, 8, 4])


@pytest.mark.parametrize('image', [np.zeros((0, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.float32)])
def test_invalid_source_pixels(image):
    with pytest.raises(ValueError):
        preprocess(image, PRE)


def test_transform_clip_and_discard():
    transform = Transform(100, 80, 64, 64)
    assert transform.original_box((-10, -10, 100, 100)) == (0, 0, 100, 80)
    assert transform.original_box((64, 0, 70, 5)) is None
    with pytest.raises(ValueError):
        transform.original_box((10, 0, 5, 1))
    with pytest.raises(ValueError):
        transform.original_box((0, 0, float('inf'), 1))


def test_nms_threshold_ties_classes_and_none_order():
    a = Detection((0, 0, 2, 2), 'a', 0.5)
    b = Detection((0, 0, 2, 1), 'a', 0.5)  # IoU exactly .5, must survive.
    c = Detection(a.bbox, 'b', 0.5)
    d = Detection(a.bbox, 'a', 0.25)
    values = [a, b, c, d, replace(a, confidence=0.249)]
    assert postprocess(values, 2, 2, {'a': 'A', 'b': 'B'}, POST) == [a, b, c]
    none = copy.deepcopy(POST)
    none['nms']['mode'] = 'none'
    assert postprocess([d, c, a], 2, 2, {'a': 'A', 'b': 'B'}, none) == [d, c, a]
    boundary = copy.deepcopy(POST)
    boundary['nms']['iou_threshold'] = 1
    assert postprocess([d, a], 2, 2, {'a': 'A'}, boundary) == [a, d]


@pytest.mark.parametrize('det', [Detection((2, 0, 1, 2), 'a', .5), Detection((0, 0, 3, 2), 'a', .5),
                                Detection((0, 0, 1, 2), 'unknown', .5), Detection((0, 0, 1, 2), 'a', float('nan')),
                                Detection((0, 0, 1, 2), 'a', 2)])
def test_invalid_adapter_output(det):
    with pytest.raises(ValueError):
        postprocess([det], 2, 2, {'a': 'A'}, POST)


class MemorySource:
    def __init__(self, log, fail=None):
        self.log, self.fail = log, fail
        self.frames = [Frame(np.zeros((80, 100, 3), dtype=np.uint8), SourceInfo('s', 'memory', 'session', 0), T)]

    def open(self):
        self.log.append('source.open')
        if self.fail == 'source.open':
            raise RuntimeError(self.fail)

    def read(self):
        if self.fail == 'source.read':
            raise RuntimeError(self.fail)
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.log.append('source.close')
        if self.fail == 'close':
            raise RuntimeError('cleanup')


class TestAdapter:
    __test__ = False

    def __init__(self, log, fail=None, empty=False):
        self.log, self.fail, self.empty = log, fail, empty

    def open(self, package):
        self.log.append('adapter.open')
        self.class_id = package.labels['classes'][0]['id']
        if self.fail == 'adapter.open':
            raise RuntimeError(self.fail)

    def infer(self, tensor, transform):
        assert tensor.shape == (1, 3, 64, 64)
        if self.fail == 'infer':
            raise RuntimeError(self.fail)
        if self.fail == 'interrupt':
            raise KeyboardInterrupt()
        if self.empty:
            return []
        return [Detection(transform.original_box((6.4, 8, 19.2, 24)), self.class_id, 0.9)]

    def close(self):
        self.log.append('adapter.close')


class SecondAdapter(TestAdapter):
    def infer(self, tensor, transform):
        return [Detection((10, 10, 30, 30), self.class_id, 0.8)]


class MemorySink:
    def __init__(self, log, fail=None):
        self.log, self.fail, self.events = log, fail, []

    def open(self):
        self.log.append('sink.open')
        if self.fail == 'sink.open':
            raise RuntimeError(self.fail)

    def write(self, event):
        if self.fail == 'sink.write':
            raise RuntimeError(self.fail)
        self.events.append(event)

    def close(self):
        self.log.append('sink.close')


@pytest.fixture
def package(tmp_path):
    root = tmp_path / 'package'
    shutil.copytree('contracts/examples/model-package', root)
    return root


def configured(package, fail=None, empty=False, adapter_type=TestAdapter):
    log = []
    source, sink = MemorySource(log, fail), MemorySink(log, fail)
    registry = RuntimeRegistry()
    registry.register(SPEC, lambda: adapter_type(log, fail, empty))
    return Pipeline(source, sink, registry, package, clock=lambda: T, event_id=lambda: 'event-1'), source, sink, log


@pytest.mark.parametrize('adapter_type', [TestAdapter, SecondAdapter])
def test_pipeline_original_coordinates(package, adapter_type):
    pipeline, source, sink, log = configured(package, adapter_type=adapter_type)
    assert pipeline.run() == 1
    event = sink.events[0]
    np.testing.assert_allclose(event['detections'][0]['bbox'], [10, 10, 30, 30], rtol=0, atol=1e-12)
    assert event['image'] == {'width': 100, 'height': 80}
    assert event['detections'][0]['label'] == 'Circle'
    assert log == ['source.open', 'adapter.open', 'sink.open', 'sink.close', 'adapter.close', 'source.close']
    with pytest.raises(RuntimeError, match='single-use'):
        pipeline.run()


def test_empty_success_and_sink_replacement(package):
    pipeline, _, sink, _ = configured(package, empty=True)
    assert pipeline.run() == 1
    assert sink.events[0]['detections'] == []
    pipeline, _, _, _ = configured(package)
    stream = io.StringIO()
    pipeline.sink = StdoutSink(stream)
    assert pipeline.run() == 1
    assert len(stream.getvalue().splitlines()) == 1
    assert json.loads(stream.getvalue())['event_id'] == 'event-1'
    assert not stream.closed


@pytest.mark.parametrize('fail,closed', [('source.open', ['source.close']),
    ('adapter.open', ['adapter.close', 'source.close']), ('sink.open', ['sink.close', 'adapter.close', 'source.close']),
    ('source.read', ['sink.close', 'adapter.close', 'source.close']), ('infer', ['sink.close', 'adapter.close', 'source.close']),
    ('sink.write', ['sink.close', 'adapter.close', 'source.close']), ('interrupt', ['sink.close', 'adapter.close', 'source.close'])])
def test_failure_never_emits_success_and_closes(package, fail, closed):
    pipeline, _, sink, log = configured(package, fail)
    with pytest.raises(KeyboardInterrupt if fail == 'interrupt' else RuntimeError):
        pipeline.run()
    assert sink.events == []
    assert [entry for entry in log if entry.endswith('.close')] == closed


def test_cleanup_preserves_primary_and_still_closes(package):
    pipeline, source, _, log = configured(package, 'infer')
    source.fail = 'close'
    with pytest.raises(RuntimeError, match='infer'):
        pipeline.run()
    assert log[-3:] == ['sink.close', 'adapter.close', 'source.close']


def test_cleanup_failure_after_success_is_failure(package):
    pipeline, source, _, _ = configured(package)
    source.fail = 'close'
    with pytest.raises(RuntimeError, match='cleanup'):
        pipeline.run()


def test_bad_package_does_not_open_ports(package):
    (package / 'model.fixture').write_text('tampered')
    pipeline, _, sink, log = configured(package)
    with pytest.raises(ValueError, match='hash_mismatch'):
        pipeline.run()
    assert log == [] and sink.events == []


def test_frame_order_and_duplicate_event_ids(package):
    pipeline, source, sink, _ = configured(package)
    source.frames.append(source.frames[0])
    with pytest.raises(ValueError, match='indices'):
        pipeline.run()
    assert len(sink.events) == 1
    pipeline, source, sink, _ = configured(package)
    source.frames.append(replace(source.frames[0], source=replace(source.frames[0].source, frame_index=1)))
    with pytest.raises(ValueError, match='IDs'):
        pipeline.run()
    assert len(sink.events) == 1


def test_changed_ontology_needs_no_core_edits(package):
    import hashlib
    for name in ('labels.json', 'data/dataset.json'):
        path = package / name
        doc = json.loads(path.read_text())
        classes = doc['classes'] if 'classes' in doc else doc['ontology']['classes']
        classes[0].update(id='square', label='Square')
        path.write_text(json.dumps(doc))
    path = package / 'manifest.json'
    manifest = json.loads(path.read_text())
    for ref in (manifest['labels'], manifest['trace']['dataset']['artifact']):
        ref['sha256'] = hashlib.sha256((package / ref['path']).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    pipeline, _, sink, _ = configured(package)
    assert pipeline.run() == 1
    assert sink.events[0]['detections'][0]['class_id'] == 'square'
    assert sink.events[0]['detections'][0]['label'] == 'Square'
