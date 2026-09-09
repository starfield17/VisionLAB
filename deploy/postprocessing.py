"""Stable filtering and class-aware NMS on original-image boxes."""
from numbers import Real
from typing import Any
import math

from .types import Box, Detection


def iou(a: Box, b: Box) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def postprocess(detections: list[Detection], width: int, height: int,
                labels: dict[str, str], spec: dict[str, Any]) -> list[Detection]:
    for det in detections:
        if not isinstance(det, Detection) or len(det.bbox) != 4:
            raise ValueError('adapter must return Detection values with four coordinates')
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (*det.bbox, det.confidence)):
            raise ValueError('adapter returned nonfinite or nonnumeric detection')
        x, y, X, Y = det.bbox
        if not (0 <= x < X <= width and 0 <= y < Y <= height):
            raise ValueError('adapter returned invalid original-image box')
        if det.class_id not in labels or not 0 <= det.confidence <= 1:
            raise ValueError('adapter returned unknown class or invalid confidence')
    candidates = [d for d in detections if d.confidence >= spec['confidence_threshold']]
    mode = spec['nms']['mode']
    if mode == 'none':
        return candidates
    if mode != 'class_aware':
        raise ValueError('unsupported NMS mode')
    candidates.sort(key=lambda d: -d.confidence)  # Python stable sort retains adapter order on ties.
    kept: list[Detection] = []
    for candidate in candidates:
        if not any(previous.class_id == candidate.class_id and iou(previous.bbox, candidate.bbox) > spec['nms']['iou_threshold']
                   for previous in kept):
            kept.append(candidate)
    return kept
