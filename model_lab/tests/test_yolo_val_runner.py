"""Metric extraction from a framework metrics object (array-valued fields included)."""
from __future__ import annotations

import numpy as np

from model_lab.runners.yolo_val import metrics_payload


class _Box:
    def __init__(self, maps):
        self.maps = maps


class FakeMetrics:
    def __init__(self, maps):
        self.results_dict = {
            'metrics/precision(B)': 0.31,
            'metrics/recall(B)': 0.29,
            'metrics/mAP50(B)': 0.24,
            'metrics/mAP50-95(B)': 0.18,
        }
        self.names = {0: 'plastic', 1: 'metal'}
        self.box = _Box(maps)
        self.speed = {'inference': 2.6}


def test_metrics_payload_handles_array_valued_class_maps():
    payload = metrics_payload(FakeMetrics(np.array([0.30, 0.33])), imgsz=640, batch=8,
                              conf=0.001, iou=0.7)
    assert payload['overall']['mAP50-95'] == 0.18
    assert payload['per_class'] == {'plastic': 0.30, 'metal': 0.33}
    assert payload['names'] == {'0': 'plastic', '1': 'metal'}
    assert payload['speed_ms'] == {'inference': 2.6}
    assert payload['imgsz'] == 640


def test_metrics_payload_tolerates_missing_class_maps():
    metrics = FakeMetrics(None)
    metrics.box = None
    payload = metrics_payload(metrics, imgsz=320, batch=1, conf=0.5, iou=0.6)
    assert payload['per_class'] == {}
    assert payload['overall']['precision'] == 0.31
