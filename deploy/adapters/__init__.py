"""Concrete inference adapters. Core imports none of them; callers compose them."""

from .onnx_detector import ADAPTER_SPEC, OnnxDetectorAdapter

__all__ = ["ADAPTER_SPEC", "OnnxDetectorAdapter"]
