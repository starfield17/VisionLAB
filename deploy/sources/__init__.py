"""Concrete input sources. Core imports none of them; callers compose them."""

from .image_file import ImageFileSource, SourceDecodeError

__all__ = ["ImageFileSource", "SourceDecodeError"]
