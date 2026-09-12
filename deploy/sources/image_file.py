"""One-image Source: decode a file, orient it, hand Core contract RGB pixels.

Ownership sits here, not in Core: EXIF orientation and grayscale/alpha handling
must produce three channels before inference, `captured_at` is the acquisition
time of the file, a corrupt file raises instead of pretending to be EOF, and a
successful read yields exactly one frame followed by EOF.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps

from ..types import Frame, SourceInfo


class SourceDecodeError(ValueError):
    """The source file cannot be decoded; this is never an end of stream."""


class ImageFileSource:
    """Decode exactly one image file into a single contract frame."""

    def __init__(self, path: str | Path, *, source_id: str = "image-file",
                 session_id: str | None = None) -> None:
        if not source_id:
            raise ValueError("source_id must be a nonempty string")
        self.path = Path(path)
        self.source_id = source_id
        self._explicit_session = session_id
        self._session_id = ""
        self._opened = False
        self._delivered = False

    def open(self) -> None:
        self._opened = True
        self._delivered = False
        # A restart is a new session; frame indices stay unique within one session.
        self._session_id = self._explicit_session or str(uuid4())

    def read(self) -> Frame | None:
        if not self._opened:
            raise RuntimeError("source is not open")
        if self._delivered:
            return None
        self._delivered = True
        try:
            with Image.open(self.path) as handle:
                handle.load()
                oriented = ImageOps.exif_transpose(handle)
                if "A" in oriented.getbands() or (oriented.mode == "P" and "transparency" in oriented.info):
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGB", rgba.size, (255, 255, 255))
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    converted = background
                else:
                    converted = oriented.convert("RGB")
                pixels = np.asarray(converted, dtype=np.uint8)
        except (OSError, ValueError, SyntaxError) as exc:
            raise SourceDecodeError(f"cannot decode {self.path.name}: {exc}") from exc
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
            raise SourceDecodeError(f"decoded {self.path.name} is not a nonempty three-channel image")
        return Frame(image=np.ascontiguousarray(pixels), source=self._info(), captured_at=self._captured_at())

    def close(self) -> None:
        self._opened = False

    def _info(self) -> SourceInfo:
        return SourceInfo(id=self.source_id, kind="image_file", session_id=self._session_id,
                          frame_index=0)

    def _captured_at(self) -> str:
        try:
            stamp = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            stamp = datetime.now(timezone.utc)
        return stamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
