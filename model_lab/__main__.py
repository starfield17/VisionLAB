"""Offline Model Lab commands; never downloads weights or starts training itself."""
import argparse
from pathlib import Path

from contractcheck.loader import read_and_parse

from .conversion import export_yolo
from .guard import run
from .overlay import render_overlay
from .train import build_train_command


def _cmd_train(args):
    command = build_train_command(root=Path.cwd(), project=args.project, name=args.name,
                                  cfg=args.cfg, model=args.model, data=args.data,
                                  device=args.device, guard_output=args.guard_output)
    print('resolved project ->', command.project)
    print('resolved guard output ->', command.guard_dir)
    return run(command.yolo_args, command.guard_dir,
               minimum_available=int(args.minimum_available_gib * 1024**3), timeout=args.timeout)


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    if args.command == 'export-yolo':
        export_yolo(args.manifest, args.destination)
    elif args.command == 'overlay':
        render_overlay(args.image, read_and_parse(args.annotation), read_and_parse(args.ontology), args.destination)
    else:
        return _cmd_train(args)


if __name__ == '__main__':
    raise SystemExit(main())
