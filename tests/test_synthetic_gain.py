"""The synthetic luminance boost (color_process.synthetic_gain): the ratio of
the deconvolved synthetic L to the one it was made from, which lights a
palette composite that has no luminance of its own. Since 2026-11-01 it is 1
below LEARNED_BOOST_FLOOR of the reference: near black the ratio is noise
over noise, and on IC 410 at full resolution it lifted the S-II specks the
denoiser kept by up to x3 ("why are some stars red").

Skips without numpy (the CI interpreter is pytest-only).
"""
import pytest

np = pytest.importorskip("numpy")

from stacking import color_process as cp  # noqa: E402


def test_boost_is_the_ratio_above_the_floor_and_one_below_it():
    ref = np.array([0.0, 0.005, 0.05, 0.1, 0.15, 0.2, 0.5], np.float32)
    lum = ref * 2.0
    g = cp.synthetic_gain(lum, ref)
    f = cp.LEARNED_BOOST_FLOOR
    assert np.allclose(g[ref <= f], 1.0)                       # nothing below the floor
    assert np.allclose(g[ref >= 2 * f], 2.0)                   # the ratio from twice it
    mid = (ref > f) & (ref < 2 * f)
    assert np.all((g[mid] > 1.0) & (g[mid] < 2.0))             # a ramp between
    assert np.allclose(cp.synthetic_gain(lum, ref, floor=0)[1:], 2.0)   # 0 turns it off


def test_boost_is_capped_and_identical_frames_change_nothing():
    ref = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    assert np.allclose(cp.synthetic_gain(ref, ref), 1.0)
    big = cp.synthetic_gain(ref * 10.0, ref)
    assert big.max() == pytest.approx(cp.MAX_LUM_BOOST)
    assert np.allclose(big[ref <= cp.LEARNED_BOOST_FLOOR], 1.0)


def test_a_speck_near_black_is_not_lifted_but_a_star_is():
    # a 3 px blob at 0.006 of the reference, x3 in the deconvolved frame
    # (what the specks measured), beside a star at 0.6 whose core sharpens
    yy, xx = np.mgrid[0:64, 0:64]
    speck = 0.006 * np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 4.0)
    star = 0.6 * np.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / 8.0)
    ref = (speck + star).astype(np.float32)
    lum = (3.0 * speck + 1.2 * star).astype(np.float32)
    g = cp.synthetic_gain(lum, ref)
    assert g[16, 16] == pytest.approx(1.0)
    assert g[48, 48] == pytest.approx(1.2, rel=1e-3)
