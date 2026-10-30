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
    # SHO (Hubble gain) and HOO both when S-II has the frames (owner,
    # 2026-10-29), HOO alone otherwise; HSO is gone since 2026-10-28.
    assert r.pick_recipes({"Ha": 44, "S-II": 39, "O-III": 41}) == ["SHO", "HOO"]
    assert r.pick_recipes({"Ha": 44, "O-III": 41, "S-II": 5}) == ["HOO"]
    assert r.pick_recipes({"Ha": 44, "O-III": 3}) == []
    assert "HSO" not in [n for n, _ in r.RECIPE_RULES]


def test_sho_is_rendered_with_the_hubble_gain_and_hoo_is_not():
    sho = r.render_command("ic410", "SHO", first=True, exptime=300, galaxy=False)
    hoo = r.render_command("ic410", "HOO", first=True, exptime=300, galaxy=False)
    assert sho[sho.index("--gain") + 1] == "hubble"
    assert "--gain" not in hoo
    assert "--gain" not in r.render_command("m33", "LRGB", first=True, exptime=300, galaxy=True)


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


def test_a_measured_weak_s_ii_drops_sho(tmp_path):
    """C3: the PC's record says S-II is 0.06 of Ha; SHO is a channel of noise."""
    full = {"Ha": 44, "S-II": 39, "O-III": 41}
    assert r.pick_recipes(full, {"S-II": 0.06}) == ["HOO"]
    assert r.pick_recipes(full, {"S-II": 0.25}) == ["SHO", "HOO"]
    assert r.pick_recipes(full, {}) == ["SHO", "HOO"] and r.pick_recipes(full, None) == ["SHO", "HOO"]
    # the mirror reader: the entry's levels per second against Ha, {} without the file
    import json
    p = tmp_path / "convergence.json"
    p.write_text(json.dumps({"ic410": {
        "Ha": {"emission": {"level_adu_s": 0.02, "gain": 26}},
        "S-II": {"emission": {"level_adu_s": 0.0012, "gain": 26}}}}))
    assert r.emission_for("IC 410", str(p)) == {"S-II": 0.06}
    assert r.emission_for("nope", str(p)) == {} and r.emission_for("ic410", str(tmp_path / "missing.json")) == {}
