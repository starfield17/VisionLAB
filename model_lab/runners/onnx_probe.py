"""Guarded probe runner: reports what an exported ONNX graph really is.

Records input/output names, shapes and dtypes, whether the graph emits decoded
detections, how the runtime reacts to unit-range and byte-range inputs, how it
compares with the training framework on the same image, and what the contract's
stretch preprocessing produces. Nothing here is inferred from documentation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Match the training framework's letterbox: INTER_LINEAR resize, 114 padding."""
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_w, new_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114.0, dtype=np.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized.astype(np.float32)
    return canvas, ratio, left, top


def stretch(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def to_tensor(pixels: np.ndarray, scale: float, mean: list, std: list) -> np.ndarray:
    tensor = (pixels * np.float32(scale) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)


def decode_detections(raw: np.ndarray) -> list[dict] | None:
    """Return rows of xyxy + score + class for a decoded detection layout."""
    array = np.asarray(raw)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 6:
        return None
    return [{'bbox': [float(x1), float(y1), float(x2), float(y2)], 'score': float(score),
             'class_id': int(class_id)}
            for x1, y1, x2, y2, score, class_id in array.tolist()]


def _iou(a: list, b: list) -> float:
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _finite(rows: list[dict]) -> bool:
    values = [value for row in rows for value in [*row['bbox'], row['score']]]
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all()) if values else True


def _predictions(model, source: str, size: int, threshold: float, device: str) -> list[dict]:
    """Reference detections from a file path, so channel handling is the framework's own."""
    result = model.predict(source=source, imgsz=size, conf=threshold, iou=0.7, device=device,
                           verbose=False)[0]
    return [{'bbox': [float(value) for value in box], 'score': float(score), 'class_id': int(cls)}
            for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                       result.boxes.cls.tolist())]


def _compare(decoded: list[dict], reference: list[dict], *, label: str, limit: int = 20) -> dict:
    """Match the highest scoring detections against a reference set by class and IoU 0.5."""
    decoded = sorted(decoded, key=lambda row: -row['score'])[:limit]
    reference = sorted(reference, key=lambda row: -row['score'])[:limit]
    matched, box_delta, score_delta = 0, 0.0, 0.0
    used: set[int] = set()
    nearest_deltas, nearest_matches = [], 0
    for reference_row in reference:
        best, best_iou = None, 0.0
        for index, row in enumerate(decoded):
            if index in used or row['class_id'] != reference_row['class_id']:
                continue
            overlap = _iou(reference_row['bbox'], row['bbox'])
            if overlap > best_iou:
                best, best_iou = index, overlap
        if best is not None and best_iou >= 0.5:
            used.add(best)
            matched += 1
            row = decoded[best]
            box_delta = max(box_delta, max(abs(a - b) for a, b in zip(reference_row['bbox'], row['bbox'])))
            score_delta = max(score_delta, abs(reference_row['score'] - row['score']))
    for row in decoded:
        best_iou, best_delta = 0.0, None
        for reference_row in reference:
            if reference_row['class_id'] != row['class_id']:
                continue
            best_iou = max(best_iou, _iou(reference_row['bbox'], row['bbox']))
            delta = max(abs(a - b) for a, b in zip(reference_row['bbox'], row['bbox']))
            best_delta = delta if best_delta is None else min(best_delta, delta)
        if best_delta is not None:
            nearest_deltas.append(best_delta)
        if best_iou >= 0.5:
            nearest_matches += 1
    return {
        'reference': label,
        'compared': limit,
        'onnx_detections': len(decoded),
        'reference_detections': len(reference),
        'matched_at_iou_0.5': matched,
        'max_abs_box_delta_px': box_delta,
        'max_abs_score_delta': score_delta,
        'nearest_box_matches_at_iou_0.5': nearest_matches,
        'nearest_median_box_delta_px': float(np.median(nearest_deltas)) if nearest_deltas else 0.0,
        'nearest_max_box_delta_px': max(nearest_deltas, default=0.0),
        'reference_max_score': max((row['score'] for row in reference), default=None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--score-threshold', type=float, default=0.001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--json-out', type=Path, required=True)
    args = parser.parse_args(argv)

    import onnx
    import onnxruntime as ort

    facts: dict = {'checker': 'failed'}
    onnx.checker.check_model(str(args.model))
    facts['checker'] = 'passed'
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    facts['onnxruntime_version'] = ort.__version__
    facts['providers'] = list(session.get_providers())
    facts['inputs'] = [{'name': item.name, 'type': item.type, 'shape': list(item.shape)}
                       for item in session.get_inputs()]
    facts['outputs'] = [{'name': item.name, 'type': item.type, 'shape': list(item.shape)}
                        for item in session.get_outputs()]
    if len(facts['inputs']) != 1 or len(facts['outputs']) != 1:
        raise ValueError('probe requires exactly one model input and one model output')
    input_name = facts['inputs'][0]['name']
    output_name = facts['outputs'][0]['name']
    size = args.imgsz

    image = np.asarray(Image.open(args.image).convert('RGB'), dtype=np.uint8)
    height, width = image.shape[:2]
    padded, ratio, left, top = letterbox(image, size)

    def run(tensor: np.ndarray) -> list[dict] | None:
        return decode_detections(np.asarray(session.run([output_name], {input_name: tensor})[0]))

    unit_rows = run(to_tensor(padded, 1 / 255, [0, 0, 0], [1, 1, 1]))
    byte_rows = run(to_tensor(padded, 1.0, [0, 0, 0], [1, 1, 1]))
    facts['decoded'] = {
        'layout': 'end2end_detections_v1' if unit_rows is not None else 'unsupported_by_probe',
        'unit_range_rows': len(unit_rows or []),
        'byte_range_rows': len(byte_rows or []),
        'unit_range_max_score': max((row['score'] for row in unit_rows or []), default=None),
        'byte_range_max_score': max((row['score'] for row in byte_rows or []), default=None),
    }
    facts['declared'] = {
        'decoder': 'end2end_detections_v1' if unit_rows is not None else None,
        'box_format': 'xyxy',
        'box_units': 'input_pixels',
        'class_index_base': 0,
        'input_scale': 1 / 255 if unit_rows is not None else None,
    }
    if unit_rows is None:
        facts['contract_stretch'] = {'status': 'unsupported output layout'}
        _write(facts, args.json_out)
        return 0

    scores = [row['score'] for row in unit_rows]
    classes = [row['class_id'] for row in unit_rows]
    facts['observed'] = {
        'rows': len(unit_rows),
        'rows_above_threshold': sum(1 for score in scores if score >= args.score_threshold),
        'score_range': [min(scores), max(scores)],
        'class_index_range': [min(classes), max(classes)],
        'raw_box_range': _box_range(unit_rows),
        'finite': _finite(unit_rows),
    }

    mapped, mapped_all = [], []
    for row in unit_rows:
        x1 = min(max((row['bbox'][0] - left) / ratio, 0.0), float(width))
        y1 = min(max((row['bbox'][1] - top) / ratio, 0.0), float(height))
        x2 = min(max((row['bbox'][2] - left) / ratio, 0.0), float(width))
        y2 = min(max((row['bbox'][3] - top) / ratio, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        mapped_row = {'bbox': [x1, y1, x2, y2], 'score': row['score'], 'class_id': row['class_id']}
        mapped_all.append(mapped_row)
        if row['score'] >= args.score_threshold:
            mapped.append(mapped_row)
    from ultralytics import YOLO

    facts['reference'] = _compare(
        mapped_all,
        _predictions(YOLO(str(args.model)), str(args.image), size, 0.0, args.device),
        label='framework_onnx_session')
    facts['checkpoint_reference'] = _compare(
        mapped,
        _predictions(YOLO(str(args.checkpoint)), str(args.image), size, args.score_threshold, args.device),
        label='framework_pytorch_checkpoint')
    facts['score_semantics'] = {
        'unit_range_max_score': facts['decoded']['unit_range_max_score'],
        'byte_range_max_score': facts['decoded']['byte_range_max_score'],
        'framework_onnx_max_score': facts['reference']['reference_max_score'],
        'framework_pytorch_max_score': facts['checkpoint_reference']['reference_max_score'],
    }

    stretch_rows = run(to_tensor(stretch(image, size), 1 / 255, [0, 0, 0], [1, 1, 1])) or []
    contract_rows, contract_boxes = [], []
    for row in stretch_rows:
        x1, y1, x2, y2 = row['bbox']
        mapped_row = {'bbox': [min(max(x1 * width / size, 0.0), float(width)),
                               min(max(y1 * height / size, 0.0), float(height)),
                               min(max(x2 * width / size, 0.0), float(width)),
                               min(max(y2 * height / size, 0.0), float(height))],
                      'score': row['score'], 'class_id': row['class_id']}
        contract_rows.append(mapped_row)
        if row['score'] >= args.score_threshold:
            contract_boxes.append(mapped_row)
    facts['contract_stretch'] = {
        'status': 'run',
        'input_scale': 1 / 255, 'layout': 'NCHW', 'color_space': 'RGB',
        'detections': len(contract_boxes),
        'max_score': max((row['score'] for row in stretch_rows), default=None),
        'top': sorted(contract_rows, key=lambda row: -row['score'])[:5],
    }
    _write(facts, args.json_out)
    return 0


def _write(facts: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(facts, indent=2) + '\n', encoding='utf-8')
    layout = facts['declared']['decoder'] if 'declared' in facts else 'unsupported'
    print(json.dumps({'checker': facts['checker'], 'layout': layout,
                      'contract_stretch': facts['contract_stretch']['status']}, sort_keys=True))


def _box_range(rows: list[dict]) -> list[float]:
    values = [value for row in rows for value in row['bbox']]
    return [min(values, default=0.0), max(values, default=0.0)] if values else [0.0, 0.0]


if __name__ == '__main__':
    raise SystemExit(main())
