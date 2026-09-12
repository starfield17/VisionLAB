"""Offline Model Lab commands; never downloads weights or starts training itself."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contractcheck.loader import read_and_parse

from . import preflight as preflight_module
from .conversion import export_yolo
from .evaluation import run_evaluation
from .export import run_export
from .guard import run
from .ingest import ingest_coco
from .overlay import render_overlay
from .package import assemble
from .runrecords import build_training_record
from .train import build_train_command

PINNED_ULTRALYTICS = '8.4.145'


def _cmd_train(args) -> int:
    if args.dataset:
        report = preflight_module.analyze(Path(args.dataset))
        issues = preflight_module.findings(report)
        report_path = Path(args.preflight_report or f'{args.guard_output}-preflight.json')
        preflight_module.write_report(report_path, report, issues, args.override_blocking)
        for line in preflight_module.summary_lines(report, issues):
            print(line)
        print(f'preflight report -> {report_path}')
        if preflight_module.blocking(issues) and not args.override_blocking:
            print('ERROR blocking preflight findings; rerun with --override-blocking REASON to proceed',
                  file=sys.stderr)
            return 2
    else:
        print('WARNING preflight was not run: pass --dataset for a production training run')
    command = build_train_command(root=Path.cwd(), project=args.project, name=args.name,
                                  cfg=args.cfg, model=args.model, data=args.data,
                                  device=args.device, guard_output=args.guard_output)
    print('resolved project ->', command.project)
    print('resolved guard output ->', command.guard_dir)
    return run(command.yolo_args, command.guard_dir,
               minimum_available=int(args.minimum_available_gib * 1024**3), timeout=args.timeout)


def _cmd_preflight(args) -> int:
    report = preflight_module.analyze(Path(args.dataset))
    issues = preflight_module.findings(report)
    if args.report:
        preflight_module.write_report(Path(args.report), report, issues)
    if args.json:
        print(json.dumps({'report': report, 'findings': issues}, indent=2))
    else:
        print('\n'.join(preflight_module.summary_lines(report, issues)))
    return 1 if preflight_module.blocking(issues) else 0


def _cmd_ingest(args) -> int:
    report = ingest_coco(images_dir=args.images, annotations_path=args.annotations,
                         ontology_config=args.ontology, destination=args.destination,
                         dataset_id=args.dataset_id, version=args.version, actor_id=args.actor_id,
                         run_id=args.run_id, tool_version=args.tool_version,
                         acquired_at=args.acquired_at, longest_side=args.longest_side,
                         seed=args.seed, train_fraction=args.train_fraction,
                         source_prefix=args.source_prefix)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_record_training(args) -> int:
    record = build_training_record(
        run_dir=args.run, dataset_manifest=args.dataset, record_id=args.id, profile=args.profile,
        config_file=args.config_file, pretrained_ref=args.pretrained,
        pretrained_path=Path(args.pretrained_path or args.pretrained), dataset_yolo_ref=args.dataset_yolo,
        implementation_version=args.implementation_version, output=args.out)
    print(json.dumps({'record': record['id'], 'metrics': record['metrics']}, indent=2))
    print('training record ->', args.out or Path(args.run) / 'training-record.json')
    return 0


def _cmd_evaluate(args) -> int:
    record = run_evaluation(
        checkpoint=args.checkpoint, data=args.data, dataset_manifest=args.dataset,
        project=args.project, name=args.name, guard_output=args.guard_output,
        metrics_out=args.metrics_out, output=args.record, record_id=args.id, split=args.split,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        implementation_version=args.implementation_version, confidence=args.confidence,
        iou=args.iou, timeout=args.timeout,
        minimum_available=int(args.minimum_available_gib * 1024**3))
    print(json.dumps({'record': record['id'], 'metrics': record['metrics']}, indent=2))
    return 0


def _cmd_export(args) -> int:
    record = run_export(
        checkpoint=args.checkpoint, dataset_manifest=args.dataset, probe_image=args.probe_image,
        staging=args.staging, guard_output=args.guard_output, probe_output=args.probe_out,
        output=args.record, record_id=args.id, imgsz=args.imgsz,
        training_record=args.training_record, implementation_version=args.implementation_version,
        nms=args.nms, simplify=args.simplify, export_timeout=args.timeout,
        minimum_available=int(args.minimum_available_gib * 1024**3))
    print(json.dumps({'record': record['id'], 'model': record['config']['model_output']}, indent=2))
    return 0


def _numbers(text: str) -> list[float]:
    values = [float(part) for part in text.split(',')]
    if len(values) != 3:
        raise ValueError('expected three comma separated numbers')
    return values


def _cmd_package(args) -> int:
    export_record = read_and_parse(args.export_record)
    export_config = export_record['config']
    decoder = args.decoder or export_config.get('decoder')
    if not decoder:
        raise ValueError('export record does not declare a decoder; pass --decoder')
    max_detections = args.max_detections or export_config['output']['shape'][1]
    preprocessing = {
        'schema_version': '1.0.0', 'input_name': export_config['input']['name'], 'dtype': 'float32',
        'layout': 'NCHW', 'color_space': 'RGB', 'width': args.imgsz, 'height': args.imgsz,
        'resize': {'mode': 'stretch', 'interpolation': 'bilinear'},
        'normalization': {'scale': args.scale, 'mean': _numbers(args.mean), 'std': _numbers(args.std)},
    }
    manifest = assemble(
        destination=args.out, package_id=args.package_id, dataset_manifest=args.dataset,
        training_record=args.training_record, evaluation_record=args.evaluation_record,
        export_record=args.export_record, model_artifact=args.model, checkpoint=args.checkpoint,
        metrics_artifact=args.metrics, probe_artifact=args.probe, adapter_spec=args.adapter_spec,
        adapter_id=args.adapter_id, adapter_version=args.adapter_version,
        adapter_config={'decoder': decoder, 'max_detections': int(max_detections)},
        preprocessing=preprocessing, confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold, nms_mode=args.nms_mode,
        store_roots=tuple(Path(root) for root in args.store_root))
    print(json.dumps({'package_id': manifest['package_id'], 'package_dir': str(args.out),
                      'model': manifest['model']['artifact']}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='model_lab')
    sub = parser.add_subparsers(dest='command', required=True)

    export = sub.add_parser('export-yolo', help='export a validated Dataset manifest to YOLO files')
    export.add_argument('manifest', type=Path)
    export.add_argument('destination', type=Path)

    overlay = sub.add_parser('overlay', help='render a human-review overlay for one annotation')
    overlay.add_argument('image', type=Path)
    overlay.add_argument('annotation', type=Path)
    overlay.add_argument('ontology', type=Path)
    overlay.add_argument('destination', type=Path)

    train = sub.add_parser('train', help='run a guarded YOLO training; relative paths resolve to absolute')
    train.add_argument('--project', required=True, help='relative project dir, e.g. workdir/training')
    train.add_argument('--name', required=True, help='unique run name')
    train.add_argument('--cfg', required=True, help='relative training config yaml')
    train.add_argument('--model', required=True, help='relative model weights')
    train.add_argument('--data', required=True, help='relative YOLO dataset data.yaml')
    train.add_argument('--device', default='mps')
    train.add_argument('--guard-output', required=True, help='relative guard output dir')
    train.add_argument('--timeout', type=float, default=1800)
    train.add_argument('--minimum-available-gib', type=float, default=3)
    train.add_argument('--dataset', help='dataset manifest used for the preflight gate')
    train.add_argument('--preflight-report', help='where to write the preflight report')
    train.add_argument('--override-blocking', help='reason recorded when blocking findings are overridden')

    check = sub.add_parser('preflight', help='report per-split image/object/class distribution')
    check.add_argument('--dataset', type=Path, required=True)
    check.add_argument('--json', action='store_true')
    check.add_argument('--report', type=Path)

    ingest = sub.add_parser('ingest-coco', help='ingest an annotated COCO dataset into the Dataset contract')
    ingest.add_argument('--images', type=Path, required=True)
    ingest.add_argument('--annotations', type=Path, required=True)
    ingest.add_argument('--ontology', type=Path, required=True)
    ingest.add_argument('--destination', type=Path, required=True)
    ingest.add_argument('--dataset-id', required=True)
    ingest.add_argument('--version', required=True)
    ingest.add_argument('--actor-id', required=True)
    ingest.add_argument('--run-id', required=True)
    ingest.add_argument('--tool-version', required=True)
    ingest.add_argument('--acquired-at', required=True, help='RFC 3339 UTC acquisition timestamp')
    ingest.add_argument('--longest-side', type=int, default=1280)
    ingest.add_argument('--seed', type=int, default=42)
    ingest.add_argument('--train-fraction', type=float, default=0.8)
    ingest.add_argument('--source-prefix')

    record = sub.add_parser('record-training', help='write a training run record from a finished run')
    record.add_argument('--run', type=Path, required=True)
    record.add_argument('--dataset', type=Path, required=True)
    record.add_argument('--id', required=True)
    record.add_argument('--profile', required=True)
    record.add_argument('--config-file', required=True)
    record.add_argument('--pretrained', required=True, help='portable reference to the pretrained weights')
    record.add_argument('--pretrained-path', help='locally readable path to those weights')
    record.add_argument('--dataset-yolo', required=True, help='portable reference to the YOLO export')
    record.add_argument('--implementation-version', default=PINNED_ULTRALYTICS)
    record.add_argument('--out', type=Path, help='record path; defaults to the run directory')

    evaluate = sub.add_parser('evaluate', help='guarded validation producing an evaluation record')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--data', type=Path, required=True)
    evaluate.add_argument('--dataset', type=Path, required=True)
    evaluate.add_argument('--project', type=Path, required=True)
    evaluate.add_argument('--name', required=True)
    evaluate.add_argument('--guard-output', type=Path, required=True)
    evaluate.add_argument('--metrics-out', type=Path, required=True)
    evaluate.add_argument('--record', type=Path, required=True)
    evaluate.add_argument('--id', required=True)
    evaluate.add_argument('--split', default='validation')
    evaluate.add_argument('--imgsz', type=int, default=640)
    evaluate.add_argument('--batch', type=int, default=8)
    evaluate.add_argument('--device', default='cuda:0')
    evaluate.add_argument('--confidence', type=float, default=0.001)
    evaluate.add_argument('--iou', type=float, default=0.7)
    evaluate.add_argument('--timeout', type=float, default=3600)
    evaluate.add_argument('--minimum-available-gib', type=float, default=3)
    evaluate.add_argument('--implementation-version', default=PINNED_ULTRALYTICS)

    export = sub.add_parser('export-onnx', help='guarded ONNX export plus an executable probe')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--dataset', type=Path, required=True)
    export.add_argument('--training-record', type=Path, required=True)
    export.add_argument('--probe-image', type=Path, required=True)
    export.add_argument('--staging', type=Path, required=True)
    export.add_argument('--guard-output', type=Path, required=True)
    export.add_argument('--probe-out', type=Path, required=True)
    export.add_argument('--record', type=Path, required=True)
    export.add_argument('--id', required=True)
    export.add_argument('--imgsz', type=int, default=640)
    export.add_argument('--nms', action='store_true', help='request export-time NMS')
    export.add_argument('--simplify', action='store_true')
    export.add_argument('--timeout', type=float, default=1800)
    export.add_argument('--minimum-available-gib', type=float, default=3)
    export.add_argument('--implementation-version', default=PINNED_ULTRALYTICS)

    package = sub.add_parser('package', help='assemble and audit a Model Package from run artifacts')
    package.add_argument('--dataset', type=Path, required=True)
    package.add_argument('--training-record', type=Path, required=True)
    package.add_argument('--evaluation-record', type=Path, required=True)
    package.add_argument('--export-record', type=Path, required=True)
    package.add_argument('--model', type=Path, required=True, help='exported model graph')
    package.add_argument('--checkpoint', type=Path, required=True, help='trained best checkpoint')
    package.add_argument('--metrics', type=Path, required=True, help='evaluation metrics artifact')
    package.add_argument('--probe', type=Path, required=True, help='export probe artifact')
    package.add_argument('--out', type=Path, required=True)
    package.add_argument('--package-id', required=True)
    package.add_argument('--adapter-spec', type=Path, required=True)
    package.add_argument('--adapter-id', default='yolo26-onnx-detections')
    package.add_argument('--adapter-version', default='1')
    package.add_argument('--decoder')
    package.add_argument('--max-detections', type=int)
    package.add_argument('--confidence-threshold', type=float, default=0.25)
    package.add_argument('--iou-threshold', type=float, default=0.7)
    package.add_argument('--nms-mode', default='none', choices=('none', 'class_aware'))
    package.add_argument('--imgsz', type=int, default=640)
    package.add_argument('--scale', type=float, default=1 / 255)
    package.add_argument('--mean', default='0,0,0')
    package.add_argument('--std', default='1,1,1')
    package.add_argument('--store-root', action='append', default=[],
                         help='retained store root used by the audit (repeatable)')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        'export-yolo': lambda: export_yolo(args.manifest, args.destination),
        'overlay': lambda: render_overlay(args.image, read_and_parse(args.annotation),
                                          read_and_parse(args.ontology), args.destination),
        'train': lambda: _cmd_train(args),
        'preflight': lambda: _cmd_preflight(args),
        'ingest-coco': lambda: _cmd_ingest(args),
        'record-training': lambda: _cmd_record_training(args),
        'evaluate': lambda: _cmd_evaluate(args),
        'export-onnx': lambda: _cmd_export(args),
        'package': lambda: _cmd_package(args),
    }
    try:
        return handlers[args.command]() or 0
    except (ValueError, OSError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
