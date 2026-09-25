"""stars=rgb: palette brightness, RGB hue, inside a star mask only.

A narrowband palette colours every star with the palette; these tests build
a field of compact stars plus one large emission knot and pin: the mask
covers the stars and not the knot; inside the mask the luminance is the
palette's and the hue is the RGB image's; outside the mask nothing changes
at all. Skips without numpy/sep (the CI interpreter is pytest-only).
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sep")

from stacking import color_process as cp  # noqa: E402


def _field(seed=3):
    rng = np.random.default_rng(seed)
    h = w = 160
    yy, xx = np.mgrid[:h, :w]
    stars = [(30, 30, 400.0, 1.6), (110, 40, 900.0, 2.0), (60, 120, 300.0, 1.4), (130, 130, 600.0, 1.8)]
    ref = np.zeros((h, w), np.float32)
    for cy, cx, amp, sig in stars:
        ref += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
    knot = 60.0 * np.exp(-((yy - 80) ** 2 + (xx - 70) ** 2) / (2 * 14.0 ** 2))   # wide, faint
    ref += knot
    ref += rng.normal(0, 1.0, (h, w)).astype(np.float32)
    return ref, stars, knot


def test_mask_covers_stars_and_not_the_knot():
    ref, stars, knot = _field()
    mask, info = cp.star_mask(ref)
    assert info["stars"] == len(stars)
    for cy, cx, _a, _s in stars:
        assert mask[cy, cx] > 0.99
    assert mask[80, 70] < 0.05          # the knot's centre stays palette
    assert 0.0 < info["coverage"] < 0.1


def test_recolour_keeps_luminance_and_takes_rgb_hue():
    ref, stars, _ = _field()
    mask, _ = cp.star_mask(ref)
    lw = np.array([0.299, 0.587, 0.114], np.float32)
    # grey palette stars, an orange RGB star image
    pal = np.repeat((np.clip(ref / 2000.0, 0, 1))[:, :, None], 3, axis=2).astype(np.float32)   # no clipped cores
    rgbi = np.dstack([pal[:, :, 0] * 1.0, pal[:, :, 0] * 0.6, pal[:, :, 0] * 0.3]).astype(np.float32)
    out = cp.rgb_stars(pal, rgbi, mask)
    cy, cx = stars[1][0], stars[1][1]
    assert abs(float(out[cy, cx] @ lw) - float(pal[cy, cx] @ lw)) < 1e-3      # luminance kept
    r, g, b = out[cy, cx]
    assert r > g > b and abs(g / r - 0.6) < 0.02 and abs(b / r - 0.3) < 0.02   # RGB hue taken
    off = mask < 1e-6
    assert np.array_equal(out[off], pal[off])                                  # untouched elsewhere


def test_black_rgb_keeps_palette():
    ref, stars, _ = _field()
    mask, _ = cp.star_mask(ref)
    pal = np.repeat((np.clip(ref / 900.0, 0, 1))[:, :, None], 3, axis=2).astype(np.float32)
    out = cp.rgb_stars(pal, np.zeros_like(pal), mask)
    assert np.allclose(out, pal, atol=1e-6)
