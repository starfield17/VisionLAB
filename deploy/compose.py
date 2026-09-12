"""Trusted static composition of the real deployment slice.

Concrete ports are bound here, never discovered from package metadata, so a
Model Package can only select an adapter that this build was shipped with.
"""
from __future__ import annotations

from .adapters.onnx_detector import ADAPTER_SPEC, OnnxDetectorAdapter
from .registry import RuntimeRegistry


def build_registry() -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.register(ADAPTER_SPEC, OnnxDetectorAdapter)
    return registry
