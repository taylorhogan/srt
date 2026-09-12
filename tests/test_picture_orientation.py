"""Every picture product is written with sky parity: the array as stored,
row 0 at the top. Before 2026-09-12 each writer flipped rows to the FITS
convention, which on this rig is a mirror image of the sky (positive CD
determinant on a plate-solved light). One helper, stacker.sky_parity, holds
the rule; these tests pin every writer to it so a stray [::-1] cannot creep
back into one of them and leave the channel JPEGs disagreeing with the
colour render."""
import numpy as np
import pytest

from stacking import stacker, color_process


def _gradient(h=8, w=6):
    # Bright row at the TOP of the stored array (row 0), dark at the bottom.
    return np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None] * np.ones((h, w), np.float32)


def test_sky_parity_is_the_stored_orientation():
    a = _gradient()
    out = stacker.sky_parity(a)
    assert out.shape == a.shape
    assert out[0, 0] > out[-1, 0]


def test_save_plain_jpg_keeps_row0_on_top(tmp_path):
    from PIL import Image
    p = stacker.save_plain_jpg(_gradient(), tmp_path / "plain.jpg")
    px = np.asarray(Image.open(p).convert("L"), dtype=float)
    assert px[0].mean() > px[-1].mean()


def test_save_rgb_keeps_row0_on_top(tmp_path):
    from PIL import Image
    rgb = np.repeat(_gradient()[:, :, None], 3, axis=2)
    p = color_process.save_rgb(rgb, tmp_path / "rgb.jpg")
    px = np.asarray(Image.open(p).convert("L"), dtype=float)
    assert px[0].mean() > px[-1].mean()


def test_channel_jpgs_keep_row0_on_top(tmp_path):
    from PIL import Image
    chan = _gradient(64, 48) * 100.0
    written = color_process.save_channel_jpgs(
        {"R": chan}, tmp_path, "t", "T", black_pct=1.0, white_pct=99.0,
        softening=0.1, subtract_background=False, mesh=4)
    assert written
    px = np.asarray(Image.open(written[0]).convert("L"), dtype=float)
    assert px[0].mean() > px[-1].mean()


def test_matplotlib_preview_keeps_row0_on_top(tmp_path):
    from PIL import Image
    p = stacker._save_jpg(_gradient(64, 48), tmp_path / "prev.fits")
    px = np.asarray(Image.open(p).convert("L"), dtype=float)
    # Crop away the figure margins before comparing top and bottom bands.
    h = px.shape[0]
    top, bot = px[int(h * 0.1):int(h * 0.3)], px[int(h * 0.7):int(h * 0.9)]
    assert top.mean() > bot.mean()
