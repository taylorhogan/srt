"""control/auto_stop -- which queued targets are finished (docs/AUTOMATION_PLAN.md A).

Pure verdicts over fake convergence and advisor answers; the live wiring
(night_plan.read_auto_stop, the scheduler's chat line) is exercised with
the planner's record shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control import auto_stop as A  # noqa: E402
from control import night_plan as NP  # noqa: E402

DONE_ADVICE = {"action": "done", "have": {"Ha": {"usable": 44}, "O-III": {"usable": 41}}}
PLAN_ADVICE = {"action": "plan", "have": {"Ha": {"usable": 10}}}
MONITOR = {"action": "monitor"}


def test_converged_alone_is_not_done_but_says_what_was_met():
    v = A.verdict("m1", True, PLAN_ADVICE)
    assert not v["done"] and v["why"] == ["convergence: every filter at its flat tail (is_dso_done)"]


def test_frame_goals_alone_are_not_done_either():
    v = A.verdict("ic410", False, DONE_ADVICE)
    assert not v["done"] and v["why"] == ["frame goals met (filter advisor): Ha 44, O-III 41"]


def test_both_verdicts_together_are_done_with_both_reasons():
    v = A.verdict("m1", True, DONE_ADVICE)
    assert v["done"] and len(v["why"]) == 2


def test_a_thin_channel_that_passes_the_curve_is_kept(): 
    # 2026-10-31: IC 410 passed the curve on 19 O-III frames; the advisor wanted 21 more.
    assert not A.verdict("ic410", True, {"action": "plan", "need": {"O-III": 21}})["done"]


def test_neither_is_not_done():
    assert A.verdict("m1", False, PLAN_ADVICE) == {"dso": "m1", "done": False, "why": []}
    assert A.verdict("m1", False, None)["done"] is False


def test_a_light_curve_monitor_is_never_done_by_snr():
    v = A.verdict("hubblev1", True, MONITOR)
    assert v["done"] is False and "monitor" in v["why"][0]


def _cov(hit, bins=10, nights=None, complete=None):
    return {"bins": bins, "hit": hit, "nights": nights if nights is not None else hit,
            "complete": (hit >= 8) if complete is None else complete}


def test_a_monitor_is_done_when_every_periodic_star_has_its_curve():
    # Owner 2026-10-31: "a light curve object can only be converged if it can
    # produce a curve, nothing to do with snr".
    stars = [{"name": "CF Cas", "period_days": 4.875, "coverage": _cov(9, nights=14)},
             {"name": "CG Cas", "period_days": 4.366, "coverage": _cov(8, nights=14)},
             {"name": "CE Cas A+B", "period_days": None, "coverage": _cov(0)}]
    v = A.monitor_verdict("ngc7790", stars)
    assert v["done"] and v["why"][0].startswith("light curve complete: CF Cas: 9 of 10 phase bins over 14 nights")
    assert "CE Cas A+B: no period, not judged" in v["why"][0]
    stars[1]["coverage"] = _cov(6, nights=9)
    v = A.monitor_verdict("ngc7790", stars)
    assert not v["done"] and any("CG Cas: 6 of 10 phase bins over 9 nights (not yet a curve)" == w for w in v["why"])
    v = A.monitor_verdict("blend", [stars[2]])
    assert not v["done"] and "no period" in v["why"][0]


def test_coverage_folds_the_nightly_points_on_the_period():
    from photometry import lightcurve as lc
    recs = [{"jd": 2461000.0 + 0.5 * i, "filter": "L", "night": "n%d" % i} for i in range(20)]   # 10 days, P 5 d: every bin
    c = lc.coverage(recs, 5.0, 2461000.0)
    assert c["complete"] and c["hit"] == 10 and c["nights"] == 20 and c["missing"] == []
    few = lc.coverage(recs[:4], 5.0, 2461000.0)                       # 4 points: 4 bins
    assert not few["complete"] and few["hit"] == 4 and len(few["missing"]) == 6
    assert lc.coverage(recs, None, None)["complete"] is False and "no period" in lc.coverage(recs, None, None)["why"]
    g_only = [dict(r, filter="G") for r in recs]                      # no L series: every filter counts
    assert lc.coverage(g_only, 5.0, 2461000.0)["hit"] == 10


def test_candidates_judge_a_monitor_only_by_its_curve():
    rows = [{"dso": "hubblev1", "status": "waiting", "lightcurve": True},
            {"dso": "ngc7790", "status": "waiting", "lightcurve": True}]
    curves = {"hubblev1": [{"name": "hubblev1", "period_days": 31.4, "coverage": _cov(9, nights=28)}],
              "ngc7790": [{"name": "CF Cas", "period_days": 4.9, "coverage": _cov(5)}]}
    found = A.candidates(rows, lambda d: True, lambda d, r: DONE_ADVICE, curve_fn=lambda d, r: curves[d])
    assert [v["dso"] for v in found] == ["hubblev1"]                   # SNR and frame goals said done for both; only the curve counts
    assert A.candidates(rows, lambda d: True, lambda d, r: DONE_ADVICE) == []   # no curve_fn: a monitor is left alone


ROWS = [{"dso": "m1", "status": "waiting"},
        {"dso": "ngc7790", "status": "held"},
        {"dso": "m33", "status": "completed"},
        {"dso": "hubblev1", "status": "waiting", "lightcurve": True},
        {"dso": "ic410", "status": "in process"},
        {"dso": "trunk", "status": "waiting"}]


def test_candidates_judge_only_the_rows_the_planner_can_pick():
    seen = []

    def advise(dso, row):
        seen.append(dso)
        return DONE_ADVICE if dso == "ic410" else PLAN_ADVICE

    found = A.candidates(ROWS, lambda d: d in ("m1", "ic410"), advise)
    assert [v["dso"] for v in found] == ["ic410"]          # m1: curve flat, goals not met
    # held, completed and the monitor were never evaluated
    assert seen == ["m1", "ic410", "trunk"]


def test_a_crashing_verdict_errs_toward_imaging_once_more():
    def boom(_):
        raise RuntimeError("convergence.json unreadable")
    found = A.candidates(ROWS, boom, lambda d, r: DONE_ADVICE if d == "trunk" else None)
    assert found == []                                      # a crashed curve verdict reads not done
    found = A.candidates(ROWS, lambda d: True, lambda d, r: 1 / 0)
    assert found == []                                      # a crashed advisor too


def test_the_object_type_is_taken_offline_and_unknown_reads_as_broadband(tmp_path):
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "dso_types.json").write_text('{"ic410": "N", "ngc6946": "G"}')
    assert A._type_offline({"dso": "ic410"}, str(tmp_path)) == "nebula"
    assert A._type_offline({"dso": "ngc6946"}, str(tmp_path)) == "galaxy"
    assert A._type_offline({"dso": "m1", "obj_type": "nebula"}, str(tmp_path)) == "nebula"
    assert A._type_offline({"dso": "sh2-92"}, str(tmp_path)) == "unknown"
    assert A._type_offline({"dso": "x"}, str(tmp_path / "nowhere")) == "unknown"


def test_mode_defaults_to_shadow_and_refuses_nonsense():
    assert A.mode({}) == "shadow"
    assert A.mode({"planner": {"auto_stop": "LIVE"}}) == "live"
    assert A.mode({"planner": {"auto_stop": "yes please"}}) == "shadow"
    assert A.mode({"planner": {"auto_stop": "off"}}) == "off"


def test_the_chat_line_names_the_mode_and_every_target():
    found = [{"dso": "m1", "done": True, "why": ["convergence: flat tail"]}]
    shadow = A.describe(found, live=False)
    assert "shadow, nothing changed" in shadow and "m1 -- convergence: flat tail" in shadow
    assert "reopen" in shadow
    live = A.describe(found, live=True)
    assert "marked converged" in live and "shadow" not in live
    assert A.describe([], live=False) == ""


def test_read_auto_stop_is_quiet_when_off_or_unreadable(tmp_path, monkeypatch):
    assert NP.read_auto_stop(tmp_path / "nope.json", {"planner": {"auto_stop": "off"}}) == []
    assert NP.read_auto_stop(tmp_path / "nope.json", {"planner": {"auto_stop": "shadow"}}) == []
    q = tmp_path / "q.json"
    q.write_text('[{"dso": "m1", "status": "waiting"}]')
    monkeypatch.setattr(A, "evaluate", lambda rows: [{"dso": rows[0]["dso"], "done": True, "why": ["x"]}])
    assert NP.read_auto_stop(q, {"planner": {"auto_stop": "shadow"}})[0]["dso"] == "m1"


# ------------------------------------------------------------ reopen (A4)

def test_a_reopened_target_is_left_alone_until_its_hold_passes():
    from datetime import date
    rows = [{"dso": "m1", "status": "waiting", A.HOLD_KEY: "2026-11-05"},
            {"dso": "ic410", "status": "waiting"}]
    found = A.candidates(rows, lambda d: True, lambda d, r: DONE_ADVICE, night=date(2026, 11, 5))
    assert [v["dso"] for v in found] == ["ic410"]                 # the hold's last night still holds
    found = A.candidates(rows, lambda d: True, lambda d, r: DONE_ADVICE, night=date(2026, 11, 6))
    assert [v["dso"] for v in found] == ["m1", "ic410"]           # the morning after, judged again
    # no night given: a hold stands; a malformed hold is no hold
    assert [v["dso"] for v in A.candidates(rows, lambda d: True, lambda d, r: DONE_ADVICE)] == ["ic410"]
    assert A.reopened({A.HOLD_KEY: "soon"}, date(2026, 11, 1)) is False
    assert A.reopened({}, date(2026, 11, 1)) is False


# ------------------------------------------------------------ the queue's type word (C4)

def test_type_word_maps_the_grids_letter():
    assert A.type_word("G") == "galaxy" and A.type_word("N") == "nebula"
    assert A.type_word("C") is None and A.type_word("?") is None
    assert A.type_word("") is None and A.type_word(None) is None
    assert A.type_word("g") == "galaxy"                       # a cached letter in either case


NEBULA_GOALS = {"Ha": 40, "O-III": 40, "S-II": 40}


def _advice(frames, need=None):
    return {"action": "plan", "kind": "nebula", "need": need or {},
            "have": {f: {"frames": n, "usable": n, "sharp": n} for f, n in frames.items()}}


def test_cap_retires_on_frames_shot_and_names_what_is_short():
    adv = _advice({"Ha": 150, "O-III": 60, "S-II": 30}, need={"S-II": 10})
    v = A.cap_verdict("faint", adv, 2.0, goals=NEBULA_GOALS)
    assert v["done"] and v["why"] == ["cap: 240 frames shot, 2.0x the nebula goals of 120; still short: S-II 10"]
    assert A.capped(v)


def test_cap_is_not_reached_below_the_multiple_and_never_for_a_monitor_or_when_off():
    adv = _advice({"Ha": 100, "O-III": 60, "S-II": 30})
    assert A.cap_verdict("faint", adv, 2.0, goals=NEBULA_GOALS) is None       # 190 < 240
    assert A.cap_verdict("faint", adv, 0.0, goals=NEBULA_GOALS) is None
    assert A.cap_verdict("v1", {"action": "monitor", "kind": "galaxy", "have": {}}, 2.0) is None
    assert A.cap_verdict("x", None, 2.0) is None


def test_candidates_apply_the_cap_only_where_the_verdicts_fall_short():
    rows = [{"dso": "faint", "status": "waiting"}, {"dso": "fresh", "status": "waiting"}]
    big = _advice({"Ha": 150, "O-III": 60, "S-II": 30}, need={"S-II": 10})
    small = _advice({"Ha": 10})
    found = A.candidates(rows, lambda d: d == "faint", lambda d, r: big if d == "faint" else small, cap=2.0)
    assert [v["dso"] for v in found] == ["faint"]
    assert found[0]["why"][0].startswith("cap:") and "convergence" in found[0]["why"][1]
    assert A.candidates(rows, lambda d: False, lambda d, r: big, cap=0.0) == []


def test_cap_factor_reads_the_config_and_tolerates_nonsense():
    assert A.cap_factor({"planner": {"auto_stop_cap": 3}}) == 3.0
    assert A.cap_factor({"planner": {"auto_stop_cap": "x"}}) == A.DEFAULT_CAP
    assert A.cap_factor({}) == A.DEFAULT_CAP
    assert A.cap_factor({"planner": {"auto_stop_cap": -1}}) == 0.0


def test_is_finished_needs_both_verdicts_or_the_cap(monkeypatch):
    import fits_processing.convergence as C
    monkeypatch.setattr(C, "is_dso_done", lambda d: d in ("both", "curve", "capped"))
    advice = {"both": DONE_ADVICE, "curve": PLAN_ADVICE, "goals": DONE_ADVICE,
              "capped": _advice({"Ha": 150, "O-III": 60, "S-II": 30}, need={"S-II": 10})}
    monkeypatch.setattr(A, "advise_offline", lambda d, r, root=None: advice.get(d))
    cfg = {"planner": {"auto_stop_cap": 2.0}}
    assert A.is_finished("both", {}, cfg) is True
    assert A.is_finished("curve", {}, cfg) is False       # IC 410 on 2026-10-31: curve flat, goals short
    assert A.is_finished("goals", {}, cfg) is False
    assert A.is_finished("capped", {}, cfg) is True
    assert A.is_finished("both", {"lightcurve": True}, cfg) is False
    assert A.is_finished("nothing", {}, cfg) is False
    look = A.finished_lookup([{"dso": "Both"}, {"dso": "curve"}], cfg)
    assert look("both") is True and look("curve") is False
