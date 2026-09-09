"""Public values and ports for synchronous, single-frame object detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from contractcheck.package import ValidatedPackage

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceInfo:
    id: str
    kind: str
    session_id: str
    frame_index: int


@dataclass(frozen=True)
class Frame:
    image: NDArray[np.uint8]
    source: SourceInfo
    captured_at: str


@dataclass(frozen=True)
class Detection:
    bbox: Box
    class_id: str
    confidence: float


@dataclass(frozen=True)
class Transform:
    original_width: int
    original_height: int
    input_width: int
    input_height: int

    def original_box(self, bbox: Box) -> Box | None:
        """Invert stretch from tensor pixel coordinates, clip, discard zero area.

        Callers must convert normalized or center-size outputs before this step.
        """
        if not all(np.isfinite(v) for v in bbox):
            raise ValueError('decoded coordinates must be finite')
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError('decoded box is inverted')
        sx = self.original_width / self.input_width
        sy = self.original_height / self.input_height
        x = min(max(float(bbox[0]) * sx, 0), self.original_width)
        y = min(max(float(bbox[1]) * sy, 0), self.original_height)
        X = min(max(float(bbox[2]) * sx, 0), self.original_width)
        Y = min(max(float(bbox[3]) * sy, 0), self.original_height)
        return None if x >= X or y >= Y else (x, y, X, Y)


class Source(Protocol):
    def open(self) -> None: ...
    def read(self) -> Frame | None: ...
    def close(self) -> None: ...


class InferenceAdapter(Protocol):
    def open(self, package: ValidatedPackage) -> None: ...
    def infer(self, tensor: NDArray[np.float32], transform: Transform) -> list[Detection]: ...
    def close(self) -> None: ...


class Sink(Protocol):
    def open(self) -> None: ...
    def write(self, event: dict[str, Any]) -> None: ...
    def close(self) -> None: ...
