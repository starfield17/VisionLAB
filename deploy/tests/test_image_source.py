"""ImageFileSource: decode policy, orientation, lifecycle and failure behaviour."""
from __future__ import annotations

import re

import pytest
from PIL import Image, ImageDraw

from deploy.sources.image_file import ImageFileSource, SourceDecodeError

RFC3339 = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$')


def make_image(path, mode='RGB', size=(16, 12), color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)
    return path


def test_rgb_frame_metadata_and_end_of_stream(tmp_path):
    path = make_image(tmp_path / 'photo.png')
    source = ImageFileSource(path, source_id='camera-1')
    source.open()
    frame = source.read()
    assert frame is not None
    assert frame.image.shape == (12, 16, 3)
    assert frame.image.dtype.name == 'uint8'
    assert frame.source.kind == 'image_file'
    assert frame.source.id == 'camera-1'
    assert frame.source.frame_index == 0
    assert frame.source.session_id
    assert RFC3339.match(frame.captured_at)
    assert source.read() is None
    source.close()
    source.close()
    with pytest.raises(RuntimeError):
        source.read()


def test_reopen_starts_a_new_session_and_keeps_explicit_ids(tmp_path):
    path = make_image(tmp_path / 'photo.png')
    source = ImageFileSource(path)
    source.open()
    first = source.read()
    source.close()
    source.open()
    second = source.read()
    assert first is not None and second is not None
    assert first.source.session_id != second.source.session_id
    fixed = ImageFileSource(path, session_id='session-fixed')
    fixed.open()
    frame = fixed.read()
    assert frame is not None
    assert frame.source.session_id == 'session-fixed'


def test_grayscale_becomes_three_channels(tmp_path):
    path = make_image(tmp_path / 'gray.png', mode='L', color=200)
    source = ImageFileSource(path)
    source.open()
    frame = source.read()
    assert frame is not None
    assert frame.image.shape[2] == 3
    assert tuple(frame.image[0, 0]) == (200, 200, 200)


def test_alpha_is_composited_on_white(tmp_path):
    path = tmp_path / 'alpha.png'
    image = Image.new('RGBA', (4, 2), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.save(path)
    source = ImageFileSource(path)
    source.open()
    frame = source.read()
    assert frame is not None
    assert tuple(frame.image[0, 0]) == (255, 0, 0)
    assert tuple(frame.image[1, 3]) == (255, 255, 255)


def test_palette_transparency_becomes_rgb(tmp_path):
    path = tmp_path / 'palette.png'
    image = Image.new('P', (4, 2))
    image.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
    image.save(path, transparency=0)
    source = ImageFileSource(path)
    source.open()
    frame = source.read()
    assert frame is not None
    assert frame.image.shape[2] == 3


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / 'oriented.jpg'
    image = Image.new('RGB', (10, 20), (5, 6, 7))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 9, 4], fill=(255, 255, 255))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif.tobytes())
    source = ImageFileSource(path)
    source.open()
    frame = source.read()
    assert frame is not None
    assert frame.image.shape[:2] == (10, 20)


@pytest.mark.parametrize('payload', [b'not an image at all', b''])
def test_corrupt_file_raises_and_never_ends_the_stream(tmp_path, payload):
    path = tmp_path / 'broken.png'
    path.write_bytes(payload)
    source = ImageFileSource(path)
    source.open()
    with pytest.raises(SourceDecodeError):
        source.read()
    assert issubclass(SourceDecodeError, ValueError)


def test_missing_file_raises(tmp_path):
    source = ImageFileSource(tmp_path / 'absent.png')
    source.open()
    with pytest.raises(SourceDecodeError):
        source.read()


def test_empty_source_id_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ImageFileSource(tmp_path / 'photo.png', source_id='')
