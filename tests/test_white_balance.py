"""Star white balance: the average field star comes out neutral.

Synthetic field with a known instrument colour -- R at 0.80 of G, B at 1.30 of
G -- so the factors that undo it are known exactly (1.25 and 0.769). Needs sep
and numpy, so it skips on the bare CI runner like the other stacking tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("sep")
cp = pytest.importorskip("stacking.color_process")


def _field(seed=3, n_stars=120, size=900, r_ratio=0.80, b_ratio=1.30, noise=2.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    g = np.zeros((size, size), np.float32)
    for _ in range(n_stars):
        x, y = rng.uniform(40, size - 40, 2)
        flux = rng.uniform(300, 6000)
        sigma = 2.6
        g += flux / (2 * np.pi * sigma**2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    r = g * r_ratio
    b = g * b_ratio
    ch = {}
    for k, v in (("R", r), ("G", g), ("B", b), ("L", g * 1.7)):
        ch[k] = (v + rng.normal(0.0, noise, v.shape)).astype(np.float32)
    return ch


def test_factors_undo_a_known_instrument_colour():
    ch = _field()
    factors, info = cp.star_white_balance(ch)
    assert info["stars"] >= 40, info
    assert abs(factors["R"] - 1.25) < 0.03, (factors, info)
    assert abs(factors["B"] - 1 / 1.30) < 0.03, (factors, info)
    assert factors["G"] == 1.0


def test_balanced_field_is_left_alone():
    ch = _field(r_ratio=1.0, b_ratio=1.0)
    factors, _ = cp.star_white_balance(ch)
    assert abs(factors["R"] - 1.0) < 0.03 and abs(factors["B"] - 1.0) < 0.03


def test_too_few_stars_means_no_change():
    ch = _field(n_stars=4)
    factors, info = cp.star_white_balance(ch)
    assert factors == {"R": 1.0, "G": 1.0, "B": 1.0}
    assert info.get("why")


def test_apply_scales_r_and_b_only_and_leaves_luminance():
    ch = _field()
    before = {k: v.copy() for k, v in ch.items()}
    out, info = cp.apply_white_balance(ch, "stars")
    assert np.allclose(out["G"], before["G"])
    assert np.allclose(out["L"], before["L"])
    assert not np.allclose(out["R"], before["R"])
    assert np.allclose(out["R"], before["R"] * info["factors"]["R"])
    same, _ = cp.apply_white_balance(before, "none")
    assert all(np.allclose(same[k], before[k]) for k in before)
