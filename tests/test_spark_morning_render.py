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
