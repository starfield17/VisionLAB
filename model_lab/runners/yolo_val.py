"""Guarded validation runner: writes library metrics as portable JSON.

Executed as `python -m model_lab.runners.yolo_val` so the caller keeps the same
interpreter and environment. This is one of the only places Model Lab imports a
training framework, and it never selects or downloads weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--json-out', type=Path, required=True)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    metrics = model.val(data=str(args.data), split='val', imgsz=args.imgsz, batch=args.batch,
                        device=args.device, project=str(args.project), name=args.name, plots=False,
                        save_json=False, conf=args.conf, iou=args.iou, verbose=True)
    payload = metrics_payload(metrics, imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou)
    validated = None
    loader = getattr(getattr(model, 'validator', None), 'dataloader', None)
    files = getattr(getattr(loader, 'dataset', None), 'im_files', None)
    if files is not None:
        validated = len(files)
    payload['validated_images'] = validated
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload['overall'], sort_keys=True))
    return 0


def metrics_payload(metrics, *, imgsz: int, batch: int, conf: float, iou: float) -> dict:
    """Portable, JSON-safe metrics extracted from a framework metrics object."""
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    names = getattr(metrics, 'names', None)
    names = names if isinstance(names, dict) else {}
    box = getattr(metrics, 'box', None)
    maps = getattr(box, 'maps', None) if box is not None else None
    per_class = {}
    if maps is not None:
        for index, value in enumerate(list(maps)):
            per_class[str(names.get(index, str(index)))] = float(value)
    speed = getattr(metrics, 'speed', None)
    return {
        'imgsz': imgsz,
        'batch': batch,
        'conf': conf,
        'iou': iou,
        'names': {str(key): str(value) for key, value in names.items()},
        'overall': {
            'precision': results['metrics/precision(B)'],
            'recall': results['metrics/recall(B)'],
            'mAP50': results['metrics/mAP50(B)'],
            'mAP50-95': results['metrics/mAP50-95(B)'],
        },
        'per_class': per_class,
        'raw_results': results,
        'speed_ms': {str(key): float(value) for key, value in (speed or {}).items()},
    }


if __name__ == '__main__':
    raise SystemExit(main())
