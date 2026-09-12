"""Run one real Model Package against one image and emit detection events."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compose import build_registry
from .pipeline import Pipeline
from .sinks.stdout import StdoutSink
from .sources.image_file import ImageFileSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionlab-deploy")
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect", help="run one image through a Model Package")
    detect.add_argument("--package", type=Path, required=True, help="Model Package directory")
    detect.add_argument("--image", type=Path, required=True, help="image file to process")
    detect.add_argument("--source-id", default="image-file")
    detect.add_argument("--expected-manifest-hash", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = Pipeline(ImageFileSource(args.image, source_id=args.source_id), StdoutSink(),
                        build_registry(), args.package,
                        expected_manifest_hash=args.expected_manifest_hash)
    try:
        pipeline.run()
    except Exception as exc:  # surfaced as an operational failure, never as an empty event
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
