"""ONNX Runtime adapter for an export that already emits decoded detections.

The package contract stays in charge of geometry: Core's preprocessing produces
the tensor, this adapter only feeds that tensor to the graph, validates the
declared layout, maps class indices through the package label mapping and
inverts the documented stretch transform. It never resizes, pads (letterbox) or
renormalizes input, and an unknown output shape is an error rather than a
guess. The runtime import stays inside the concrete adapter boundary.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contractcheck.package import AdapterSpec, ValidatedPackage

from ..types import Detection, Transform

ADAPTER_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decoder": {"const": "end2end_detections_v1"},
        "max_detections": {"type": "integer", "minimum": 1},
    },
    "required": ["decoder", "max_detections"],
    "additionalProperties": False,
}

ADAPTER_SPEC = AdapterSpec(id="yolo26-onnx-detections", version="1", formats=("onnx",),
                           config_schema=ADAPTER_CONFIG_SCHEMA)


def default_session_factory(model_path: Path) -> Any:
    """Create a runtime session for one packaged graph; imported lazily."""
    import onnxruntime as ort  # type: ignore[import-untyped]

    return ort.InferenceSession(str(model_path))


class OnnxDetectorAdapter:
    """Decode one end-to-end detection graph into contract detections."""

    def __init__(self, session_factory: Callable[[Path], Any] | None = None) -> None:
        self._session_factory = session_factory or default_session_factory
        self._session: Any = None
        self._input_name = ""
        self._input_shape: tuple[int, ...] = ()
        self._output_shape: tuple[int, ...] = ()
        self._labels: dict[int, str] = {}

    def open(self, package: ValidatedPackage) -> None:
        descriptor = package.manifest["model"]
        config = descriptor["adapter"]["config"]
        if set(config) != {"decoder", "max_detections"}:
            raise ValueError("adapter config must declare exactly decoder and max_detections")
        if config["decoder"] != "end2end_detections_v1":
            raise ValueError(f"unsupported decoder {config['decoder']!r}")
        max_detections = config["max_detections"]
        preprocessing = package.preprocessing
        if (preprocessing["dtype"], preprocessing["layout"]) != ("float32", "NCHW"):
            raise ValueError("adapter requires float32 NCHW preprocessing")
        height, width = preprocessing["height"], preprocessing["width"]
        labels = {}
        for entry in package.labels["classes"]:
            index = entry["output_index"]
            if index in labels:
                raise ValueError("label output indices must be unique")
            labels[index] = entry["id"]
        session = self._session_factory(package.model_path)
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("adapter requires exactly one model input and one model output")
        input_shape = tuple(inputs[0].shape)
        if inputs[0].name != preprocessing["input_name"]:
            raise ValueError("model input name differs from package preprocessing")
        if input_shape != (1, 3, height, width):
            raise ValueError(f"model input shape {input_shape} differs from package preprocessing")
        if "float" not in str(inputs[0].type):
            raise ValueError(f"model input type {inputs[0].type!r} must be float32")
        output_shape = tuple(outputs[0].shape)
        if output_shape != (1, max_detections, 6):
            raise ValueError(f"model output shape {output_shape} is not the declared "
                             f"[1, {max_detections}, 6] detection layout")
        self._session, self._input_name = session, inputs[0].name
        self._input_shape, self._output_shape = input_shape, output_shape
        self._labels = labels

    def infer(self, tensor: NDArray[np.float32], transform: Transform) -> list[Detection]:
        if self._session is None:
            raise RuntimeError("adapter is not open")
        if tensor.dtype != np.float32 or tuple(tensor.shape) != self._input_shape:
            raise ValueError(f"preprocessed tensor {tuple(tensor.shape)} does not match the model input")
        if not np.isfinite(tensor).all():
            raise ValueError("preprocessed tensor contains nonfinite values")
        raw = np.asarray(self._session.run(None, {self._input_name: tensor})[0])
        if tuple(raw.shape) != self._output_shape:
            raise ValueError(f"model returned shape {tuple(raw.shape)}, expected {self._output_shape}")
        if not np.isfinite(raw).all():
            raise ValueError("model output contains nonfinite values")
        detections: list[Detection] = []
        for x1, y1, x2, y2, score, index in raw[0].tolist():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"model returned a score outside [0, 1]: {score}")
            if float(index) != int(index) or int(index) not in self._labels:
                raise ValueError(f"model returned an unknown class index: {index}")
            if x2 < x1 or y2 < y1:
                raise ValueError("model returned an inverted box")
            original = transform.original_box((x1, y1, x2, y2))
            if original is not None:
                detections.append(Detection(bbox=original, class_id=self._labels[int(index)],
                                            confidence=float(score)))
        return detections

    def close(self) -> None:
        self._session = None
