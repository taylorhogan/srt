"""n2n_lrgb_render's deconvolved composites are optional output.

A field the deconvolution refuses (too few stars) or a failure inside it must
cost that composite only -- the raw and denoised renders are the morning's
deliverable and were there first. The starless "deblurred" composite is an
extra since 2026-10-21 (--deblur N); "learned" is the third.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import n2n_lrgb_render as r  # noqa: E402


def test_three_classes_in_pipeline_order():
    assert r.TAGS == ("raw", "denoised", "learned")
    assert r.EXTRA_TAGS == ("deblurred",)


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


# ------------------------------------------------------------ the morning push

def _push_setup(monkeypatch, tmp_path, names):
    """A fake Pushover and a written list of small JPEGs with these tags."""
    from PIL import Image
    from utils import pushover
    sent = []
    monkeypatch.setattr(pushover, "push_message_with_picture",
                        lambda msg, path: sent.append((msg, Path(path).name)) or True)
    written = []
    for tag in names:
        p = tmp_path / f"m1_HOO_{tag}.jpg"
        Image.new("RGB", (8, 8)).save(p)
        written.append(p)
    return sent, written


class _Args:
    dso, recipe, exptime, push_all = "m1", "HOO", 300, False


def test_push_sends_one_picture_the_learned(monkeypatch, tmp_path):
    # Owner, 2026-10-28: "you only need to send a picture of the learned, not all 3".
    sent, written = _push_setup(monkeypatch, tmp_path, ("raw", "denoised", "learned"))
    assert r.push_render(_Args(), tmp_path, written, {}, log=lambda m: None)
    assert [n for _, n in sent] == ["m1_HOO_learned.jpg"]


def test_push_falls_back_when_the_deconvolution_refused(monkeypatch, tmp_path):
    sent, written = _push_setup(monkeypatch, tmp_path, ("raw", "denoised"))
    r.push_render(_Args(), tmp_path, written, {}, log=lambda m: None)
    assert [n for _, n in sent] == ["m1_HOO_denoised.jpg"]
    sent, written = _push_setup(monkeypatch, tmp_path, ("raw",))
    r.push_render(_Args(), tmp_path, written, {}, log=lambda m: None)
    assert [n for _, n in sent] == ["m1_HOO_raw.jpg"]


def test_push_all_sends_the_triple_and_an_extra_is_always_sent(monkeypatch, tmp_path):
    sent, written = _push_setup(monkeypatch, tmp_path, ("raw", "denoised", "learned", "deblurred"))
    r.push_render(_Args(), tmp_path, written, {}, log=lambda m: None)
    assert [n for _, n in sent] == ["m1_HOO_learned.jpg", "m1_HOO_deblurred.jpg"]
    sent.clear()
    a = _Args(); a.push_all = True
    r.push_render(a, tmp_path, written, {}, log=lambda m: None)
    assert [n for _, n in sent] == [f"m1_HOO_{t}.jpg" for t in ("raw", "denoised", "learned", "deblurred")]
