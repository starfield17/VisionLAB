import json

import pytest
from PIL import Image

from model_lab.acquisition import ImageDecodeError, canonicalize


def test_no_upscale_small_source(tmp_path):
    src = tmp_path / 'small.png'
    Image.new('RGB', (100, 80), (10, 20, 30)).save(src)
    out = tmp_path / 'canonical.png'
    record = canonicalize(src, out)
    assert record['canonical_width'] == 100
    assert record['canonical_height'] == 80
    assert record['transform']['scale'] == 1.0
    assert record['transform']['resample'] == 'none'
    assert record['transform']['alpha_policy'] == 'none'
    assert record['transform']['exif_orientation_applied'] is False
    assert Image.open(out).getpixel((1, 1)) == (10, 20, 30)
    assert Image.open(out).mode == 'RGB'


def test_downscale_longest_side_640(tmp_path):
    src = tmp_path / 'big.jpg'
    Image.new('RGB', (3200, 2400), (255, 0, 0)).save(src, format='JPEG')
    out = tmp_path / 'c640.png'
    record = canonicalize(src, out)
    img = Image.open(out)
    assert img.size == (640, 480)
    assert record['canonical_width'] == 640
    assert record['transform']['scale'] == 0.2
    assert record['transform']['resample'] == 'lanczos'
    assert 'source_width' in record and record['source_width'] == 3200
    assert record['source_sha256'] != record['canonical_sha256']


def test_exif_orientation_applied(tmp_path):
    src = tmp_path / 'rotated.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB', (100, 50), (0, 255, 0)).save(src, format='JPEG', exif=exif)
    out = tmp_path / 'oriented.png'
    record = canonicalize(src, out)
    img = Image.open(out)
    assert record['transform']['exif_orientation_applied'] is True
    assert img.size == (50, 100)
    assert record['source_width'] == 50


def test_grayscale_to_rgb(tmp_path):
    src = tmp_path / 'gray.png'
    Image.new('L', (64, 48), 128).save(src)
    out = tmp_path / 'c.png'
    record = canonicalize(src, out)
    assert Image.open(out).mode == 'RGB'
    assert record['transform']['alpha_policy'] == 'none'


def test_rgba_composite_white(tmp_path):
    src = tmp_path / 'rgba.png'
    img = Image.new('RGBA', (8, 8), (255, 0, 0, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))
    img.save(src)
    out = tmp_path / 'c.png'
    record = canonicalize(src, out)
    canonical = Image.open(out)
    assert canonical.mode == 'RGB'
    assert canonical.getpixel((0, 0)) == (255, 255, 255)
    assert canonical.getpixel((1, 1)) == (255, 0, 0)
    assert record['transform']['alpha_policy'] == 'composite_white'


def test_palette_with_transparency_composite_white(tmp_path):
    src = tmp_path / 'palette.png'
    img = Image.new('P', (8, 8))
    img.putpalette([v for i in range(256) for v in (i, i, i)])
    img.putpixel((0, 0), 0)
    img.putpixel((5, 5), 250)
    img.save(src, transparency=0)
    out = tmp_path / 'c.png'
    record = canonicalize(src, out)
    canonical = Image.open(out)
    assert canonical.getpixel((0, 0)) == (255, 255, 255)
    assert canonical.getpixel((5, 5)) == (250, 250, 250)
    assert record['transform']['alpha_policy'] == 'composite_white'


def test_corrupt_image_raises(tmp_path):
    src = tmp_path / 'broken.jpg'
    src.write_bytes(b'\x00\x01\x02 not an image')
    with pytest.raises(ImageDecodeError):
        canonicalize(src, tmp_path / 'c.png')
    assert not (tmp_path / 'c.png').exists()


def test_retry_profile_448_and_640_distinct(tmp_path):
    src = tmp_path / 'big.png'
    Image.new('RGB', (2000, 1000), (9, 9, 9)).save(src)
    r640 = canonicalize(src, tmp_path / 'c640.png')
    r448 = canonicalize(src, tmp_path / 'c448.png', longest_side=448)
    assert (r448['canonical_width'], r448['canonical_height']) == (448, 224)
    assert r448['transform']['longest_side'] == 448
    assert r448['canonical_sha256'] != r640['canonical_sha256']


def test_metadata_portable_and_immutable_output(tmp_path):
    src = tmp_path / 'small.png'
    Image.new('RGB', (30, 20), (1, 2, 3)).save(src)
    out = tmp_path / 'out.png'
    record = canonicalize(src, out)
    assert str(tmp_path) not in json.dumps(record)
    with pytest.raises(FileExistsError):
        canonicalize(src, out)
    again = tmp_path / 'out2.png'
    record2 = canonicalize(src, again)
    assert record['canonical_sha256'] == record2['canonical_sha256']


def test_invalid_longest_side(tmp_path):
    src = tmp_path / 'ok.png'
    Image.new('RGB', (10, 10)).save(src)
    with pytest.raises(ValueError):
        canonicalize(src, tmp_path / 'c.png', longest_side=0)
