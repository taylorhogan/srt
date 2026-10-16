"""spark_morning_render.pick_recipes -- which palettes a night's frame counts earn.

Pure; the script's module-level imports are stdlib, and main() is never
called here. The m33 case is the 2026-09-15 morning: L 18 / R 4 / G 4 / B 3
after a two-hour second slot, skipped by the old 12-frames-in-every-filter
rule while ngc7380 rendered three palettes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import spark_morning_render as r  # noqa: E402


def test_narrowband_needs_min_frames_in_every_filter():
    assert r.pick_recipes({"Ha": 44, "S-II": 39, "O-III": 41}) == ["HSO", "SHO", "HOO"]
    assert r.pick_recipes({"Ha": 44, "O-III": 41, "S-II": 5}) == ["HOO"]
    assert r.pick_recipes({"Ha": 44, "O-III": 3}) == []


def test_lrgb_needs_luminance_depth_but_only_a_few_colour_frames():
    assert r.pick_recipes({"L": 18, "R": 4, "G": 4, "B": 3}) == ["LRGB"]   # m33, 2026-09-15
    assert r.pick_recipes({"L": 27, "R": 4, "G": 4, "B": 2}) == ["LRGB"]   # one colour short: tolerated
    assert r.pick_recipes({"L": 27, "R": 4, "G": 2, "B": 2}) == []         # two colours short
    assert r.pick_recipes({"L": 9, "R": 4, "G": 4, "B": 4}) == []          # luminance too shallow
    assert r.pick_recipes({"R": 20, "G": 20, "B": 20}) == []               # no luminance


def test_thresholds_are_what_the_docstring_says():
    assert r.MIN_FRAMES == 12 and r.MIN_COLOUR_FRAMES == 3


def test_spark_picks_halrgb_only_with_enough_ha():
    base = {"L": 20, "R": 4, "G": 4, "B": 4}
    assert r.pick_recipes(base) == ["LRGB"]
    assert r.pick_recipes({**base, "Ha": r.MIN_FRAMES}) == ["HALRGB", "LRGB"]
    # Ha is a detail channel: a colour-sized handful does not earn the recipe.
    assert r.pick_recipes({**base, "Ha": r.MIN_COLOUR_FRAMES}) == ["LRGB"]
    # ...and without L there is no luminance to add the excess to.
    assert "HALRGB" not in r.pick_recipes({"R": 20, "G": 20, "B": 20, "Ha": 20})


def test_a_galaxys_broadband_recipes_are_white_balanced_on_stars():
    lrgb = r.render_command("ngc6946", "LRGB", first=False, exptime=300, galaxy=True)
    halrgb = r.render_command("ngc6946", "HALRGB", first=True, exptime=300, galaxy=True)
    assert lrgb[lrgb.index("--wb") + 1] == "stars"
    assert halrgb[halrgb.index("--wb") + 1] == "stars" and "--force" in halrgb
    assert "--force" not in lrgb


def test_no_white_balance_off_a_galaxy_or_on_a_palette():
    assert "--wb" not in r.render_command("ngc7789", "LRGB", first=True, exptime=300, galaxy=False)
    # A galaxy's narrowband palette is still a palette.
    assert "--wb" not in r.render_command("m33", "HOO", first=True, exptime=300, galaxy=True)
    assert set(r.WB_STARS_RECIPES).isdisjoint(r.NARROWBAND)


def test_a_failed_type_lookup_is_not_a_galaxy(monkeypatch):
    import nina_gen.nina_sequence_gen as g

    def boom(_):
        raise RuntimeError("SIMBAD down")
    monkeypatch.setattr(g, "_classify_object_type", boom)
    assert r.is_galaxy("ngc6946") is False
    monkeypatch.setattr(g, "_classify_object_type", lambda _: "galaxy")
    assert r.is_galaxy("ngc6946") is True
