"""The filter advisor (docs/FILTER_ADVISOR_PLAN.md): one test per rule.

Pure: advise() takes frame records and a few dicts, so these run on the
pytest-only CI runner. The fixtures are the three hand edits of 2026-10-18
that the convergence split could not make.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fits_processing import filter_advisor as fa  # noqa: E402


def frames(spec):
    """spec: {filter: [fwhm, ...]} at 300 s."""
    return [{"filter": f, "fwhm_arcsec": w, "exptime": 300.0} for f, ws in spec.items() for w in ws]


def test_a_soft_luminance_still_needs_time():
    # ngc1961: 51 L at a median ~2.55", only 7 sharp; colours 18 each.
    soft_l = [2.3] * 7 + [2.6] * 44
    adv = fa.advise(frames({"L": soft_l, "R": [2.4] * 18, "G": [2.6] * 18, "B": [2.4] * 18}),
                    "galaxy", tonight={"seeing": "good"})
    assert adv["action"] == "plan"
    assert adv["plan"]["L"] == fa.RATIO_SCALE and adv["plan"]["L"] > adv["plan"]["R"]
    assert any("only 7 sharp" in r for r in adv["reasons"])


def test_ha_joins_a_galaxy_that_has_ha_and_the_fifth_filter_sits_out():
    adv = fa.advise(frames({"L": [2.1] * 20, "R": [2.3] * 20, "G": [2.3] * 20,
                            "B": [2.3] * 22, "Ha": [2.7] * 18}), "galaxy")
    assert "Ha" in adv["plan"]
    assert len(adv["plan"]) <= fa.MAX_FILTERS
    assert "B" not in adv["plan"]                     # the smallest need waits a night
    assert any("sits out" in r for r in adv["reasons"])


def test_a_monitor_target_keeps_its_own_plan():
    adv = fa.advise(frames({"L": [2.2] * 31}), "galaxy",
                    entry={"dso": "hubblev1", "lightcurve": True, "filter_plan": {"L": 1}})
    assert adv["action"] == "monitor" and adv["plan"] == {"L": 1}


def test_seeing_tilts_the_detail_channel():
    base = frames({"L": [2.2] * 30, "R": [2.3] * 12, "G": [2.3] * 12, "B": [2.3] * 12})
    good = fa.advise(base, "galaxy", tonight={"seeing": "good"})["plan"]
    poor = fa.advise(base, "galaxy", tonight={"seeing": "poor"})["plan"]
    assert good["L"] / good["R"] > poor["L"] / poor["R"]


def test_a_bright_moon_moves_time_from_l_and_b_to_ha():
    base = frames({"L": [2.2] * 30, "R": [2.3] * 12, "G": [2.3] * 12, "B": [2.3] * 12,
                   "Ha": [2.6] * 10})
    dark = fa.advise(base, "galaxy", tonight={"moon_lit": 0.1, "moon_up": True})
    bright = fa.advise(base, "galaxy", tonight={"moon_lit": 0.9, "moon_up": True})
    assert bright["plan"].get("Ha", 0) >= dark["plan"].get("Ha", 0)
    assert any("moon" in r for r in bright["reasons"])
    set_moon = fa.advise(base, "galaxy", tonight={"moon_lit": 0.9, "moon_up": False})
    assert not any("moon" in r for r in set_moon["reasons"])   # below the horizon: no effect


def test_soft_frames_that_the_stacker_would_reject_do_not_count():
    t = fa.tally(frames({"R": [2.0] * 10 + [4.0] * 5}))
    assert t["R"]["frames"] == 15 and t["R"]["usable"] == 10


def test_short_frames_do_not_pass_for_depth():
    fr = [{"filter": "L", "fwhm_arcsec": 2.0, "exptime": 10.0}] * 100
    assert "L" not in fa.tally(fr)                          # 10 s frames are not 300 s depth
    assert fa.tally(fr, {"L": 10})["L"]["frames"] == 100    # unless that is the entry's exposure


def test_every_goal_met_says_so_and_deepens():
    adv = fa.advise(frames({"L": [2.0] * 80, "R": [2.2] * 30, "G": [2.2] * 30, "B": [2.2] * 30}),
                    "galaxy")
    assert adv["action"] == "done"
    assert adv["plan"] == {"L": 2, "R": 1, "G": 1, "B": 1}


def test_a_nebula_uses_narrowband_with_ha_as_the_detail_channel():
    adv = fa.advise(frames({"Ha": [2.8] * 40, "O-III": [2.6] * 10}), "nebula")
    assert set(adv["plan"]) <= {"Ha", "O-III", "S-II"}
    assert adv["plan"]["S-II"] == fa.RATIO_SCALE         # never shot: the largest need


def test_filter_names_from_headers_are_canonical():
    assert fa.canonical("OIII") == "O-III" and fa.canonical("H-alpha") == "Ha"
    assert fa.canonical("Lum") == "L" and fa.canonical("weird") is None
