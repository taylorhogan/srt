"""color_process.render_qc (D1 of docs/AUTOMATION_PLAN.md, 2026-10-30): is a composite publishable?

Synthetic composites in 0..1: a good one passes; a black-crushed sky, a lit
channel, a stretch that is not confident and an uncropped footprint each
fail with a reason that names the fault. Skips without numpy (CI).
"""
import pytest

np = pytest.importorskip("numpy")

from stacking import color_process as cp  # noqa: E402


def _field(sky=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full((200, 300, 3), sky, dtype=np.float32) + rng.normal(0, 0.005, (200, 300, 3)).astype(np.float32)
    a[60:140, 100:220] += 0.35                     # the object
    for _ in range(12):                            # stars, saturated cores (a real field saturates far less than 0.5 %)
        y, x = rng.integers(2, 198), rng.integers(2, 298)
        a[y - 1:y + 2, x - 1:x + 2] = 1.0
    return np.clip(a, 0, 1)


GOOD = {"confident": True, "why": []}


def test_a_good_composite_passes_and_reports_its_measures():
    qc = cp.render_qc(_field(), GOOD)
    assert qc["pass"] is True and qc["reasons"] == []
    assert 0.04 < qc["measures"]["sky_grey"] < 0.06
    assert qc["measures"]["white_clip"] < cp.QC_WHITE_CLIP_MAX


def test_a_black_crushed_sky_fails():
    qc = cp.render_qc(_field(sky=0.0), GOOD)
    assert not qc["pass"] and any("black-crushed" in r or "clipped to black" in r for r in qc["reasons"])


def test_a_too_bright_sky_fails():
    qc = cp.render_qc(_field(sky=0.3), GOOD)
    assert not qc["pass"] and any("too bright" in r for r in qc["reasons"])


def test_a_lit_channel_fails():
    a = _field()
    a[:, :, 0] = np.clip(a[:, :, 0] + 0.6, 0, 1)      # R blown over the sky (the brown M1 sky)
    qc = cp.render_qc(a, GOOD)
    assert not qc["pass"] and any("channel R lit" in r for r in qc["reasons"])


def test_an_unconfident_stretch_fails_with_its_reason():
    qc = cp.render_qc(_field(), {"confident": False, "why": ["no clean sky in R (9% of tiles at sky)"]})
    assert not qc["pass"] and qc["reasons"][0].startswith("stretch not confident: no clean sky")
    assert cp.render_qc(_field(), None)["pass"]       # no stretch given: that check is skipped


def test_an_uncropped_footprint_fails():
    qc = cp.render_qc(_field(), GOOD, cropped=False)
    assert not qc["pass"] and any("common footprint" in r for r in qc["reasons"])


def test_palette_stars_must_be_neutral():
    a = _field()
    mask = np.zeros(a.shape[:2], dtype=bool)
    mask[10:14, 10:14] = True
    a[10:14, 10:14] = (0.9, 0.3, 0.9)                 # magenta stars under a gain
    assert not cp.render_qc(a, GOOD, star_mask=mask, cls="narrowband")["pass"]
    assert cp.render_qc(a, GOOD, star_mask=mask, cls="broadband")["pass"]   # only palettes are judged
    a[10:14, 10:14] = (0.9, 0.9, 0.9)
    assert cp.render_qc(a, GOOD, star_mask=mask, cls="narrowband")["pass"]
