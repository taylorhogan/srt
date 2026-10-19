"""n2n_lrgb_render's third composite: the deblurred set is optional output.

A field the starless deconvolution refuses (too few stars) or a failure
inside it must cost the deblurred products only -- the raw and denoised
renders are the morning's deliverable and were there first.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import n2n_lrgb_render as r  # noqa: E402


def test_three_classes_in_pipeline_order():
    assert r.TAGS == ("raw", "denoised", "deblurred")


def test_a_refusal_or_a_failure_costs_only_the_deblurred_composite(monkeypatch):
    lines = []
    rng = np.random.default_rng(1)
    flat = (300 + rng.normal(0, 2, (300, 300))).astype(np.float32)     # no stars: refused
    assert r.deblurred_channels({"L": flat, "R": flat.copy()}, 8, log=lines.append) is None
    assert any("refused" in m for m in lines)

    from stacking import deblur
    def boom(*a, **k):
        raise RuntimeError("kernel exploded")
    monkeypatch.setattr(deblur, "deblur_channels", boom)
    lines.clear()
    assert r.deblurred_channels({"L": flat}, 8, log=lines.append) is None
    assert any("failed" in m and "kernel exploded" in m for m in lines)


def test_a_deblurred_set_comes_back_with_every_channel(monkeypatch):
    from stacking import deblur
    img = np.full((50, 50), 300.0, np.float32)
    def fake(channels, iters, clamp_to, method, log, stars="imaged"):
        return {k: v + 1 for k, v in channels.items()}, {"L": {"method": method, "iters": iters}}
    monkeypatch.setattr(deblur, "deblur_channels", fake)
    out = r.deblurred_channels({"L": img, "R": img}, 40, log=lambda m: None)
    assert set(out) == {"L", "R"} and float(out["L"][0, 0]) == 301.0
