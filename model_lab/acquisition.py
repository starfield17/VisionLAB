"""Acquisition: normalize source images into canonical RGB PNGs for labeling.

This module only normalizes inputs. It never runs LocateAnything, never
infers and never trains. The canonical PNG and its recorded metadata are the
coordinate frame for the Dataset contract described in
docs/AUTOLABEL_TRAIN_INFORMATION.md step 3: decode, EXIF orientation, RGB,
proportional downsize without upscaling to longest side 640, and portable
hash/dimension/transform records.

Decode or conversion failures raise ImageDecodeError so the caller stops that
item; a failed image is never turned into a negative annotation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps


class ImageDecodeError(ValueError):
    """Source image cannot be decoded or converted; that item is stopped."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(source: Path, output: Path, *, longest_side: int = 640) -> dict:
    """Decode, orient and resize one image; save a canonical PNG and record metadata.

    `source_width`/`source_height` are the EXIF-corrected original dimensions
    (the coordinate frame before downscale). The returned record is portable:
    it contains only base names and hashes, never absolute paths.
    """
    source, output = Path(source), Path(output)
    if type(longest_side) is not int or longest_side <= 0:
        raise ValueError('longest_side must be a positive integer')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256_file(source)
    try:
        image = Image.open(source)
        image.load()
        applied = (image.getexif().get(274, 1) or 1) != 1
        image = ImageOps.exif_transpose(image)
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
            alpha_policy = 'composite_white'
        elif 'A' in image.getbands():
            image = image.convert('RGBA')
            alpha_policy = 'composite_white'
        else:
            alpha_policy = 'none'
        if alpha_policy == 'composite_white':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel('A'))
            image = background
        else:
            image = image.convert('RGB')
    except Exception as exc:
        raise ImageDecodeError(f'cannot decode {source.name}: {exc}') from exc

    oriented_w, oriented_h = image.size
    scale = longest_side / max(oriented_w, oriented_h)
    if scale >= 1:
        canonical, scale, resample = image, 1.0, 'none'
    else:
        width, height = max(1, round(oriented_w * scale)), max(1, round(oriented_h * scale))
        canonical = image.resize((width, height), Image.Resampling.LANCZOS)
        resample = 'lanczos'
    try:
        canonical.save(output, format='PNG')
    except Exception as exc:
        raise ValueError(f'cannot write canonical PNG: {exc}') from exc

    return {
        'source_name': source.name,
        'source_sha256': source_hash,
        'source_width': oriented_w,
        'source_height': oriented_h,
        'canonical_name': output.name,
        'canonical_sha256': _sha256_file(output),
        'canonical_width': canonical.width,
        'canonical_height': canonical.height,
        'transform': {
            'alpha_policy': alpha_policy,
            'exif_orientation_applied': applied,
            'longest_side': longest_side,
            'scale': scale,
            'resample': resample,
        },
    }
