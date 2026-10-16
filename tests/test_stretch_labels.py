"""Stretch labels (docs/AUTO_STRETCH_PLAN.md, phase 0).

An automatic stretch is scored against the stretches the owner picked, so a
pick has to be recorded as one: a render made from a sweep panel names that
panel, `keep` stamps the keeper, and measure_stretch turns a stretch into
numbers that mean the same thing on any field (black in sigma below sky, what
clips) instead of percentiles that do not.

Skips without numpy (the CI interpreter is pytest-only).
"""
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sep")

from stacking import color_process as cp  # noqa: E402


def _log(tmp_path, entries):
    (tmp_path / cp.RENDER_LOG_NAME).write_text(json.dumps(entries), encoding="utf-8")


SWEEP = {"time": "2026-10-16T11:00:00-04:00", "kind": "sweep", "tag": "HALRGB",
         "recipe": "HALRGB", "file": "sheet.jpg",
         "variants": [{"black_pct": 45, "white_pct": 99.9, "softening": 0.05},
                      {"black_pct": 60, "white_pct": 99.9, "softening": 0.05}]}


def test_a_render_with_a_panels_settings_names_that_panel(tmp_path):
    _log(tmp_path, [SWEEP])
    link = cp.sweep_link(tmp_path, "HALRGB",
                         {"black_pct": 60.0, "white_pct": 99.9, "softening": 0.05,
                          "white_balance": "stars"})
    assert link == {"from_sweep": {"time": SWEEP["time"], "file": "sheet.jpg"}, "panel": 2}


def test_settings_on_no_panel_or_another_recipe_are_not_a_pick(tmp_path):
    _log(tmp_path, [SWEEP])
    assert cp.sweep_link(tmp_path, "HALRGB", {"black_pct": 55, "white_pct": 99.9,
                                              "softening": 0.05}) is None
    assert cp.sweep_link(tmp_path, "LRGB", {"black_pct": 60, "white_pct": 99.9,
                                            "softening": 0.05}) is None


def test_keep_stamps_the_latest_render_of_the_recipe_asked_for(tmp_path):
    _log(tmp_path, [SWEEP,
                    {"time": "t1", "kind": "render", "tag": "HALRGB", "recipe": "HALRGB"},
                    {"time": "t2", "kind": "render", "tag": "LRGB", "recipe": "LRGB"}])
    e = cp.mark_kept(tmp_path, "halrgb")
    assert e["time"] == "t1"
    saved = json.loads((tmp_path / cp.RENDER_LOG_NAME).read_text(encoding="utf-8"))
    assert [bool(x.get("kept")) for x in saved] == [False, True, False]
    assert cp.mark_kept(tmp_path)["time"] == "t2"           # no recipe: the latest
    assert cp.mark_kept(tmp_path / "nowhere") is None


def _galaxy(seed=3):
    """Flat sky (sigma 1) + a broad disk + stars, as background-subtracted planes."""
    rng = np.random.default_rng(seed)
    h = w = 256
    yy, xx = np.mgrid[:h, :w]
    disk = 30.0 * np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 30.0 ** 2))
    stars = np.zeros((h, w))
    for cy, cx in ((30, 40), (200, 60), (60, 210), (220, 220), (100, 30)):
        stars += 400.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
    planes = {c: (disk + stars + rng.normal(0, 1.0, (h, w))).astype(np.float32)
              for c in ("R", "G", "B")}
    return planes


def test_black_at_sky_measures_zero_margin_and_a_higher_black_measures_negative():
    sub = _galaxy()
    at_sky = cp.measure_stretch(sub, {c: 0.0 for c in sub}, white=60.0, softening=0.05)
    above = cp.measure_stretch(sub, {c: 1.0 for c in sub}, white=60.0, softening=0.05)
    for c in sub:
        assert abs(at_sky["margin_sigma"][c]) < 0.15
        assert above["margin_sigma"][c] < -0.8
    # A brighter rendering of the same object reads brighter in output units.
    soft = cp.measure_stretch(sub, {c: 0.0 for c in sub}, white=60.0, softening=0.01)
    assert soft["ext_p50_out"] > at_sky["ext_p50_out"]


def test_clipping_is_split_between_stars_and_the_object():
    sub = _galaxy()
    # White below the disk peak but far below the stars: the object clips.
    low = cp.measure_stretch(sub, {c: 0.0 for c in sub}, white=20.0, softening=0.05)
    high = cp.measure_stretch(sub, {c: 0.0 for c in sub}, white=200.0, softening=0.05)
    assert low["ext_clip"] > 0.05
    assert high["ext_clip"] == 0.0
    assert high["star_clip"] > 0.0          # stars may clip; that is not the object
