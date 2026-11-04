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


# --- auto_stretch_v2 (phase 1) ----------------------------------------------

def _nebula_everywhere(seed=5):
    """Emission filling the whole frame: no tile is sky."""
    rng = np.random.default_rng(seed)
    h = w = 256
    yy, xx = np.mgrid[:h, :w]
    neb = 8.0 + 4.0 * np.sin(xx / 23.0) * np.cos(yy / 31.0) + 4.0
    return {c: (neb + rng.normal(0, 1.0, (h, w))).astype(np.float32) for c in ("R", "G", "B")}


def test_v2_hits_its_targets_on_a_clean_galaxy_field():
    sub = _galaxy()
    t = {"margin": -0.4, "ext_clip": 0.01, "bright_out": 0.85}
    v2 = cp.auto_stretch_v2(sub, "broadband", t)
    assert v2["confident"], v2["why"]
    m = cp.measure_stretch(sub, v2["blacks"], v2["white"], v2["softening"], fm=v2["fm"])
    for c in ("R", "G", "B"):
        assert abs(m["margin_sigma"][c] - t["margin"]) < 0.15
    assert m["ext_clip"] <= 0.03
    assert abs(m["bright_out"] - t["bright_out"]) < 0.05


def test_v2_says_so_when_the_field_has_no_clean_sky():
    v2 = cp.auto_stretch_v2(_nebula_everywhere(), "narrowband")
    assert not v2["confident"]
    assert any("no clean sky" in w for w in v2["why"])


def test_clean_sky_finds_the_dark_corner_not_the_field_median():
    rng = np.random.default_rng(9)
    a = rng.normal(0, 1.0, (256, 256)).astype(np.float32)
    a[:, 64:] += 6.0                       # three quarters of the frame is emission
    cs = cp.clean_sky(a)
    assert abs(cs["sky"]) < 0.2 and cs["confident"]
    assert cs["clean_fraction"] < 0.4


def test_v2_white_ignores_the_stars():
    sub = _galaxy()
    v2 = cp.auto_stretch_v2(sub, "broadband", {"ext_clip": 0.0})
    # The disk peaks near 30 ADU; the stars near 400. White follows the disk.
    assert 20.0 < v2["white"] < 60.0


def test_recipes_map_to_their_class():
    assert cp.stretch_class("HOO") == "narrowband" and cp.stretch_class("hso_noflat") == "narrowband"
    assert cp.stretch_class("HALRGB") == "broadband" and cp.stretch_class("LRGB") == "broadband"


# --- the white floor and the sky ceiling (2026-11-04, the first finals) -------

def test_v2_white_never_sits_below_the_floor_and_says_which_rule_won():
    sub = _galaxy()
    v2 = cp.auto_stretch_v2(sub, "broadband", {"ext_clip": 0.0})
    assert v2["white_rule"] == "ext_clip" and 20.0 < v2["white"] < 60.0
    floored = cp.auto_stretch_v2(sub, "broadband", {"ext_clip": 0.0}, fm=v2["fm"], white_floor=400.0)
    assert floored["white"] == 400.0 and floored["white_rule"] == "floor"
    # a floor below the rule's own white changes nothing
    low = cp.auto_stretch_v2(sub, "broadband", {"ext_clip": 0.0}, fm=v2["fm"], white_floor=5.0)
    assert low["white"] == v2["white"] and low["white_rule"] == "ext_clip"


def test_v2_raises_the_softening_rather_than_draw_the_sky_above_sky_max():
    # the Melotte 15 final: bright p90 is most of the field, the curve solved
    # for it put the sky at 39 % grey
    sub = _nebula_everywhere()
    free = cp.auto_stretch_v2(sub, "narrowband", {"sky_max": None})
    held = cp.auto_stretch_v2(sub, "narrowband", fm=free["fm"])
    sky_max = cp.STRETCH_TARGETS["narrowband"]["sky_max"]
    assert free["sky_out"] > sky_max
    assert held["softening"] > free["softening"]
    # emission fills this frame: the median pixel IS the object, and even the
    # softest curve cannot hold it down; the stretch says so and is not confident
    assert held["sky_out"] > sky_max and any("no sky to hold down" in w for w in held["why"])
    assert not held["confident"]
    # a field with sky: the ceiling is met by softening alone, a note that keeps confidence
    g = _galaxy()
    g = {c: a * 0.05 for c, a in g.items()}          # faint: the solved curve would be hard
    lifted = cp.auto_stretch_v2(g, "broadband", {"bright_out": 0.95})
    assert lifted["sky_out"] <= sky_max + 1e-6
    assert all(w.startswith("softening") for w in lifted["why"]) and lifted["confident"], lifted["why"]
    # the ceiling is a softening note: it does not by itself make the stretch unconfident
    galaxy = cp.auto_stretch_v2(_galaxy(), "broadband")
    assert galaxy["confident"], galaxy["why"]
    assert galaxy["sky_out"] <= cp.STRETCH_TARGETS["broadband"]["sky_max"] + 1e-6


def test_a_doubtful_sky_is_a_note_not_a_fault_when_the_blacks_are_bounded():
    sub = _nebula_everywhere()
    v2 = cp.auto_stretch_v2(sub, "narrowband")
    assert not v2["confident"] and any(w.startswith("no clean sky") for w in v2["why"])
    routine = {c: float(np.percentile(a, 40)) - 0.5 for c, a in sub.items()}
    held = cp.auto_stretch_v2(sub, "narrowband", fm=v2["fm"], white_floor=400.0, black_floor=routine)
    assert held["white"] == 400.0
    assert all(held["blacks"][c] <= routine[c] + 1e-6 for c in "RGB")
    assert not any(w.startswith("no clean sky") for w in held["why"])
    assert any("no clean sky" in w for w in held["why"])
    # with the white at the stars the median pixel is held down and the stretch is confident
    assert held["sky_out"] <= cp.STRETCH_TARGETS["narrowband"]["sky_max"] + 1e-6
    assert held["confident"], held["why"]


def test_black_never_sits_above_the_floor_but_may_sit_below_it():
    sub = _galaxy()
    v2 = cp.auto_stretch_v2(sub, "broadband")
    above = {c: v2["blacks"][c] - 1.0 for c in v2["blacks"]}     # a floor below v2's blacks: they move down to it
    held = cp.auto_stretch_v2(sub, "broadband", fm=v2["fm"], black_floor=above)
    assert held["black_rule"] == "floor" and all(held["blacks"][c] == above[c] for c in above)
    below = {c: v2["blacks"][c] + 1.0 for c in v2["blacks"]}     # a floor above them: nothing moves
    kept = cp.auto_stretch_v2(sub, "broadband", fm=v2["fm"], black_floor=below)
    assert kept["black_rule"] == "margin" and all(kept["blacks"][c] == v2["blacks"][c] for c in below)
