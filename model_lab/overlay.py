"""Human-review overlays: draw candidate boxes and class labels on canonical images.

Overlays are a review aid, not contract data. They never modify the canonical
image; every annotation is drawn on a copy. Class colors are deterministic per
class ID so repeated reviews of the same class are visually stable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from contractcheck.common import check_box
from contractcheck.errors import ValidationResult

_PALETTE = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (250, 190, 212), (28, 28, 28),
)


def _color(class_id: str) -> tuple[int, int, int]:
    index = int(hashlib.sha256(class_id.encode('utf-8')).hexdigest(), 16) % len(_PALETTE)
    return _PALETTE[index]


def render_overlay(canonical: Path, annotation: dict, ontology: dict, output: Path, *,
                   line_width: int = 2, font_size: int = 14) -> None:
    """Draw every object's box and label on a copy of the canonical image.

    Boxes are canonical-pixel coordinates validated against the image bounds;
    unknown classes, nonfinite values, inverted or out-of-bounds boxes raise.
    An empty annotation still produces an overlay so a reviewer can confirm a
    negative image. The output is never overwritten.
    """
    canonical, output = Path(canonical), Path(output)
    if output.exists():
        raise FileExistsError(output)
    image = Image.open(canonical).convert('RGB')
    width, height = image.size
    labels = {c['id']: c['label'] for c in ontology['classes']}
    if len(labels) != len(ontology['classes']):
        raise ValueError('ontology class IDs must be unique')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    for i, obj in enumerate(annotation['objects']):
        if obj['class_id'] not in labels:
            raise ValueError(f'unknown class {obj["class_id"]!r} in overlay')
        bbox = obj['bbox']
        result = ValidationResult()
        check_box(result, bbox, width, height, '<overlay>', f'/objects/{i}/bbox')
        if not result.ok:
            raise ValueError(str(result.errors))
        x, y, X, Y = bbox
        color = _color(obj['class_id'])
        draw.rectangle([x, y, X, Y], outline=color, width=line_width)
        text = labels[obj['class_id']]
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        text_x = min(x, max(0, width - tw - 4))
        text_y = max(0, y - th - 4)
        draw.rectangle([text_x, text_y, text_x + tw + 4, text_y + th + 4], fill=color)
        draw.text((text_x + 2, text_y + 2), text, fill=(0, 0, 0), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format='PNG')
