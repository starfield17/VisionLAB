"""Contract v1 RGB uint8 → float32 tensor, without intermediate quantization."""
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .types import Transform


def preprocess(image: NDArray[np.uint8], spec: dict[str, Any]) -> tuple[NDArray[np.float32], Transform]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Source must supply a nonempty oriented RGB uint8 HWC image')
    h, w, _ = image.shape
    width, height = spec['width'], spec['height']
    # Spec comes from a validated package. Explicit checks also guard direct callers.
    if (spec['dtype'], spec['resize']) != ('float32', {'mode': 'stretch', 'interpolation': 'bilinear'}):
        raise ValueError('unsupported preprocessing profile')
    if spec['color_space'] not in ('RGB', 'BGR') or spec['layout'] not in ('NCHW', 'NHWC'):
        raise ValueError('unsupported color space or tensor layout')
    if width <= 0 or height <= 0:
        raise ValueError('tensor dimensions must be positive')
    pixels = image.astype(np.float32)
    if spec['color_space'] == 'BGR':
        pixels = pixels[..., ::-1]
    xs = np.clip((np.arange(width, dtype=np.float64) + 0.5) * w / width - 0.5, 0, w - 1)
    ys = np.clip((np.arange(height, dtype=np.float64) + 0.5) * h / height - 0.5, 0, h - 1)
    x0, y0 = xs.astype(np.intp), ys.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (xs - x0).astype(np.float32)[None, :, None], (ys - y0).astype(np.float32)[:, None, None]
    top = pixels[y0[:, None], x0] * (1 - wx) + pixels[y0[:, None], x1] * wx
    bottom = pixels[y1[:, None], x0] * (1 - wx) + pixels[y1[:, None], x1] * wx
    resized = top * (1 - wy) + bottom * wy
    normalization = spec['normalization']
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        mean = np.asarray(normalization['mean'], dtype=np.float32)
        std = np.asarray(normalization['std'], dtype=np.float32)
        tensor = (resized * np.float32(normalization['scale']) - mean) / std
    if not np.isfinite(tensor).all():
        raise ValueError('normalization produced nonfinite tensor')
    if spec['layout'] == 'NCHW':
        tensor = tensor.transpose(2, 0, 1)
    return np.ascontiguousarray(tensor[None], dtype=np.float32), Transform(w, h, width, height)
