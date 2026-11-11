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
                            "B": [2.3] * 22, "Ha": [2.7] * 18}), "galaxy", ha_gain_set=True)
    assert "Ha" in adv["plan"]
    assert len(adv["plan"]) <= fa.MAX_FILTERS
    assert "B" not in adv["plan"]                     # the smallest need waits a night
    assert any("sits out" in r for r in adv["reasons"])


def test_ha_sits_out_of_a_galaxy_until_the_narrowband_gain_is_on():
    # NGC 2276, 2026-11-11: 21 Hα frames at gain 0 recorded 0.45 ADU a frame,
    # nothing, and the advisor had given Hα half the slot. Without an Hα entry
    # in nina.gain the galaxy's set is LRGB however the entry or the frames
    # on disk ask for Hα; the nebula rule is untouched.
    have = frames({"L": [2.1] * 20, "R": [2.3] * 20, "G": [2.3] * 20, "B": [2.3] * 10, "Ha": [2.7] * 21})
    adv = fa.advise(have, "galaxy", {"recipe": "HALRGB"})
    assert "Ha" not in adv["plan"] and "Ha" not in adv["need"]
    assert any("sits out" in r and "gain" in r for r in adv["reasons"])
    assert "Ha" in fa.advise(have, "galaxy", {"recipe": "HALRGB"}, ha_gain_set=True)["need"]
    assert "Ha" in fa.advise(have, "galaxy", {"recipe": "HALRGB"}, galaxy_ha_needs_gain=False)["need"]
    neb = fa.advise(frames({"Ha": [2.4] * 10, "O-III": [2.6] * 10, "S-II": [2.6] * 10}), "nebula")
    assert "Ha" in neb["need"]


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


def test_a_soft_night_gives_the_detail_channel_a_token_share():
    """2026-10-19: ngc1961 needed 22 sharp L, colour was at goal, and the 850 hPa
    wind was 42 km/h. Halving L still gave it most of five hours. A soft night
    makes almost no sharp frames, so L gets ratio 1 and colour deepens."""
    fr = frames({"L": [2.3] * 38 + [2.7] * 46, "R": [2.4] * 25, "G": [2.5] * 25, "B": [2.3] * 25})
    adv = fa.advise(fr, "galaxy", tonight={"seeing": "poor"})
    assert adv["plan"]["L"] == 1
    assert all(adv["plan"][f] > 1 for f in ("R", "G", "B"))     # at goal, still deepening
    assert any("token share" in r for r in adv["reasons"])
    assert fa.advise(fr, "galaxy", tonight={"seeing": "bad"})["plan"]["L"] == 1


def test_a_soft_night_still_fits_four_blocks_with_the_token_share():
    fr = frames({"L": [2.2] * 30, "R": [2.3] * 20, "G": [2.3] * 15, "B": [2.4] * 12, "Ha": [2.7] * 20})
    adv = fa.advise(fr, "galaxy", tonight={"seeing": "poor", "moon_lit": 0.6, "moon_up": True})
    assert adv["plan"]["L"] == 1 and len(adv["plan"]) <= fa.MAX_FILTERS
    assert not any("L x" in r for r in adv["reasons"] if r.startswith("moon"))


def test_a_measured_weak_line_needs_no_more_frames():
    """C3: S-II measured 0.06 of Ha over enough frames -> its need is zero and the
    reason says so; O-III, measured strong, keeps its need."""
    adv = fa.advise(frames({"Ha": [2.8] * 40, "O-III": [2.6] * 30, "S-II": [2.7] * 24}), "nebula",
                    emission={"S-II": 0.06, "O-III": 0.45})
    assert adv["need"]["S-II"] == 0 and adv["need"]["O-III"] > 0
    assert "S-II" not in adv["plan"] and "O-III" in adv["plan"]
    assert any("S-II measured 0.06 of Ha over 24 usable frames" in r and "HOO target" in r
               for r in adv["reasons"])
    assert adv["emission"] == {"S-II": 0.06, "O-III": 0.45}


def test_a_weak_line_measured_on_too_few_frames_is_still_shot():
    adv = fa.advise(frames({"Ha": [2.8] * 40, "O-III": [2.6] * 30, "S-II": [2.7] * 8}), "nebula",
                    emission={"S-II": 0.06})
    assert adv["need"]["S-II"] > 0 and "S-II" in adv["plan"]
    # ...and the thresholds are the config's
    adv = fa.advise(frames({"Ha": [2.8] * 40, "O-III": [2.6] * 30, "S-II": [2.7] * 8}), "nebula",
                    emission={"S-II": 0.06}, weak_line_min_frames=5)
    assert adv["need"]["S-II"] == 0


def test_a_weak_line_can_finish_a_target():
    """Ha and O-III at goal, S-II weak: done, not 'S-II needs 16 more'."""
    adv = fa.advise(frames({"Ha": [2.0] * 40, "O-III": [2.6] * 40, "S-II": [2.7] * 24}), "nebula",
                    emission={"S-II": 0.05})
    assert adv["action"] == "done"


def test_advisor_settings_read_the_config():
    base = {"weak_line_ratio": fa.WEAK_LINE_RATIO, "weak_line_min_frames": fa.WEAK_LINE_MIN_FRAMES,
            "galaxy_ha_needs_gain": True, "ha_gain_set": False}
    assert fa.advisor_settings({}) == base
    assert fa.advisor_settings({"nina": {"advisor": {"weak_line_ratio": 0.2, "weak_line_min_frames": 9}}}) \
        == dict(base, weak_line_ratio=0.2, weak_line_min_frames=9)
    # The narrowband gain switch is read from the gain table, the rule's own flag from advisor
    assert fa.advisor_settings({"nina": {"gain": {"Ha": [26, 10]}}})["ha_gain_set"] is True
    assert fa.advisor_settings({"nina": {"advisor": {"galaxy_ha_needs_gain": False}}})["galaxy_ha_needs_gain"] is False


# The curve (2026-11-01): stack SNR weights a shortfall; with every goal met
# the night goes to the filters the curve still waits on.
IC410_CURVE = {"Ha": {"slope_pct": -0.2, "flat": True, "snr": 1.96},
               "O-III": {"slope_pct": -0.3, "flat": True, "snr": 1.39},
               "S-II": {"slope_pct": -0.3, "flat": True, "snr": 0.73}}


def test_a_shortfall_leans_to_the_noisiest_stack():
    # IC 410 on 11-01: 38 Ha, 47 O-III, 37 S-II sharp/usable of 40 each.
    fr = frames({"Ha": [2.0] * 38, "O-III": [2.0] * 47, "S-II": [2.0] * 37})
    plain = fa.advise(fr, "nebula")
    curved = fa.advise(fr, "nebula", curve=IC410_CURVE)
    assert plain["plan"] == {"Ha": 3, "S-II": 4}
    assert curved["plan"] == {"Ha": 1, "S-II": 4}
    assert curved["need"] == plain["need"] and curved["action"] == "plan"
    assert any("stack SNR" in r for r in curved["reasons"])


def test_snr_weight_is_capped():
    # Ha needs 30, S-II 3; S-II's stack is 20x noisier. Capped at x4 the
    # weights are Ha 30 / S-II 12; uncapped S-II's 60 would take the night.
    fr = frames({"Ha": [2.0] * 10, "S-II": [2.0] * 37, "O-III": [2.0] * 40})
    adv = fa.advise(fr, "nebula", curve={"Ha": {"snr": 10.0}, "S-II": {"snr": 0.5}})
    assert adv["plan"] == {"Ha": 4, "S-II": 2}


def test_goals_met_curve_not_flat_sends_the_night_to_the_unflat_filters():
    # Melotte 15 on 11-01: every goal met; O-III -2.56, S-II -0.69, Ha flat.
    fr = frames({"Ha": [2.0] * 76, "O-III": [2.0] * 51, "S-II": [2.0] * 54})
    curve = {"Ha": {"slope_pct": -0.26, "flat": True}, "O-III": {"slope_pct": -2.56, "flat": False},
             "S-II": {"slope_pct": -0.69, "flat": False}}
    adv = fa.advise(fr, "nebula", curve=curve)
    assert adv["action"] == "done"                 # auto-stop reads this as goals met
    assert adv["plan"] == {"O-III": 4, "S-II": 1}
    assert "Ha" not in adv["plan"]
    assert fa.advise(fr, "nebula")["plan"] == {"Ha": 1, "O-III": 1, "S-II": 1}   # no curve: as before


def test_goals_met_and_curve_flat_keeps_the_deepening_split():
    fr = frames({"Ha": [2.0] * 50, "O-III": [2.0] * 50, "S-II": [2.0] * 50})
    flat = {f: {"slope_pct": -0.1, "flat": True} for f in ("Ha", "O-III", "S-II")}
    assert fa.advise(fr, "nebula", curve=flat)["plan"] == {"Ha": 1, "O-III": 1, "S-II": 1}


def test_curve_status_reads_the_convergence_entries():
    import pytest
    conv = pytest.importorskip("fits_processing.convergence")
    entries = {"O2": {"tail_slope_pct": -2.56, "final_rmse_pct": 4.1, "frame_count": 56, "calibrated": True,
                      "emission": {"level_adu_s": 0.0027, "noise_adu_s": 0.0025}},
               "Ha": {"tail_slope_pct": -0.26, "final_rmse_pct": 5.3, "frame_count": 85, "calibrated": True},
               "S-II": {"tail_slope_pct": -0.1, "final_rmse_pct": 9.3, "frame_count": 55}}
    st = conv.curve_status(entries, 0.40, 25.0, 16)
    assert st["O-III"] == {"slope_pct": -2.56, "flat": False, "snr": 1.08}
    assert st["Ha"]["flat"] is True and st["Ha"]["snr"] is None
    assert st["S-II"]["flat"] is False             # uncalibrated: not known, so not flat
