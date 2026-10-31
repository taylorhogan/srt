"""control/targets_board (ARCHITECTURE_PLAN phase 4, 2026-10-30): the board the Live tab shows. Pure."""
from datetime import datetime, timedelta, timezone

from control import targets_board as B

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 11, 7, 14, 0, tzinfo=TZ)
REG = {"ic410": {"state": "ACQUIRING", "priority": 90, "legacy_status": "waiting"},
       "m1": {"state": "CONVERGED", "priority": 5, "legacy_status": "converged"},
       "ngc2403": {"state": "QUEUED", "priority": 5, "legacy_status": "waiting"},
       "m33": {"state": "RETIRED", "priority": 5, "legacy_status": "completed"}}
ROWS = [{"dso": "ic410", "status": "waiting", "obj_type": "nebula", "filter_plan": {"Ha": 1, "O-III": 2},
         "filter_plan_until": "2026-11-08"},
        {"dso": "m1", "status": "converged"}, {"dso": "ngc2403", "status": "waiting", "moon": "dark"},
        {"dso": "m33", "status": "completed"}]
CONV = {"ic410": {"Ha": {"total_frames": 35, "frame_count": 32, "tail_slope_pct": -0.3, "updated": "2026-11-07",
                         "emission": {"level_adu_s": 0.02}},
                  "S-II": {"total_frames": 25, "frame_count": 25, "tail_slope_pct": -0.5}},
        "m1": {"Ha": {"total_frames": 60, "frame_count": 58, "tail_slope_pct": -0.1}}}
PLAN = {"noon": {"check": "noon", "ts": "2026-11-07T12:00:30-04:00",
                 "plan": {"best": "ic410", "will_image": True,
                          "slots": [{"dso": "ic410", "start": "2026-11-07T21:00-04:00", "end": "2026-11-08T01:00-04:00",
                                     "hours": 4, "filters": {"Ha": 1, "O-III": 2}}],
                          "auto_stop": [{"dso": "m1", "done": True, "why": ["convergence: every filter at its flat tail"]}]}}}
DRAFTS = {"m1": {"draft_hash": "a" * 64, "recipe": "HOO", "article_source": "fallback", "auto_publish_on": "2026-11-10"}}
FLAGS = {"auto_stop": "shadow", "filter_advisor": "shadow", "final_mode": "shadow", "gate": "shadow"}


def test_board_joins_every_source_per_target():
    b = B.build(REG, ROWS, CONV, PLAN, DRAFTS, {}, {"m33"}, FLAGS, NOW)
    assert b["generated"].startswith("2026-11-07T14:00") and b["flags"] == FLAGS
    assert b["tonight"] == {"best": "ic410", "will_image": True, "check": "noon"}
    assert b["counts"] == {"ACQUIRING": 1, "CONVERGED": 1, "QUEUED": 1, "RETIRED": 1}
    assert [t["dso"] for t in b["targets"]] == ["ic410", "m1", "ngc2403", "m33"]      # state order, then priority
    ic = b["targets"][0]
    assert ic["tonight"] == "21:00-01:00" and ic["filters_tonight"] == {"Ha": 1, "O-III": 2}
    assert ic["filters"]["Ha"] == {"frames": 35, "used": 32, "slope": -0.3, "rmse": None, "updated": "2026-11-07", "emission": 0.02}
    assert ic["frames"] == 60 and ic["obj_type"] == "nebula" and ic["filter_plan_until"] == "2026-11-08"
    assert ic["would_stop"] is None and ic["draft"] is None and ic["on_site"] is False
    m1 = b["targets"][1]
    assert m1["would_stop"] == ["convergence: every filter at its flat tail"]
    assert m1["draft"] == {"hash": "aaaaaaaaaaaa", "recipe": "HOO", "written_by": "facts only",
                           "auto_publish_on": "2026-11-10", "decision": "pending"}
    assert b["targets"][2]["moon"] == "dark" and b["targets"][3]["on_site"] is True


def test_draft_decision_states():
    ok = {"m1": {"decision": "approve", "draft_hash": "a" * 64}}
    assert B.build(REG, ROWS, CONV, PLAN, DRAFTS, ok, set(), FLAGS, NOW)["targets"][1]["draft"]["decision"] == "approve"
    old = {"m1": {"decision": "approve", "draft_hash": "b" * 64}}
    assert B.build(REG, ROWS, CONV, PLAN, DRAFTS, old, set(), FLAGS, NOW)["targets"][1]["draft"]["decision"] == "stale"


def test_a_stale_plan_shows_no_tonight():
    b = B.build(REG, ROWS, CONV, PLAN, {}, {}, set(), FLAGS, NOW + timedelta(days=2))
    assert b["tonight"] == {"best": None, "will_image": None, "check": None}
    assert b["targets"][0]["tonight"] is None and b["targets"][1]["would_stop"] is None
    # the sunset record, when there is one, wins over noon
    plan = dict(PLAN, sunset={"check": "sunset", "ts": "2026-11-07T17:30-04:00",
                              "plan": {"best": "m1", "will_image": False, "slots": []}})
    assert B.build(REG, ROWS, CONV, plan, {}, {}, set(), FLAGS, NOW + timedelta(hours=4))["tonight"]["best"] == "m1"


def test_empty_inputs_give_an_empty_board():
    b = B.build({}, [], {}, None, {}, {}, set(), {}, NOW)
    assert b["targets"] == [] and b["counts"] == {} and b["tonight"]["best"] is None


GOALS = {"Ha": 40, "O-III": 40, "S-II": 40}


def _advice(sharp, frames=None, kind="nebula"):
    frames = frames or sharp
    return {"action": "plan", "kind": kind,
            "have": {f: {"frames": frames[f], "usable": frames[f], "sharp": n} for f, n in sharp.items()}}


def test_progress_is_the_worst_filters_share_with_that_filter_named():
    p = B.progress(_advice({"Ha": 30, "O-III": 19, "S-II": 25}, {"Ha": 35, "O-III": 25, "S-II": 25}),
                   curve_done=True, nights=3, cap_factor=2.0, goals=GOALS)
    assert p["pct"] == 48 and p["limiting"] == "O-III"
    assert p["need_frames"] == 10 + 21 + 15 and p["shot"] == 85 and p["cap"] == 240
    assert p["nights"] == 3 and p["per_night"] == 28.3 and p["nights_to_go"] == 2   # 46 / 28.3
    assert p["curve"] is True and p["sharp"]["S-II"] == 25 and p["goals"] == GOALS


def test_progress_complete_and_unknown_pace():
    p = B.progress(_advice({"Ha": 44, "O-III": 41, "S-II": 40}), False, 0, 0.0, goals=GOALS)
    assert p["pct"] == 100 and p["need_frames"] == 0 and p["nights_to_go"] == 0
    assert p["cap"] is None and p["per_night"] is None
    q = B.progress(_advice({"Ha": 4}), False, 0, 2.0, goals=GOALS)
    assert q["pct"] == 0 and q["limiting"] == "O-III" and q["nights_to_go"] is None


def test_progress_is_none_for_monitors_and_without_advice():
    assert B.progress({"action": "monitor", "kind": "galaxy", "have": {}}, True, 5) is None
    assert B.progress(None, True, 5) is None
    assert B.progress({"action": "plan", "kind": "nebula", "have": {}}, True, 5, goals={}) is None


def test_nights_from_paths_counts_sessions():
    paths = [r"C:\Users\iriso\Documents\N.I.N.A\Targets\ic410\cdk17\2026-10-27\LIGHT\a.fits",
             r"C:\x\ic410\cdk17\2026-10-27\LIGHT\b.fits", "/home/t/Targets/ic410/cdk17/2026-10-28/LIGHT/c.fits",
             "/home/t/Targets/ic410/cdk17/2026-10-29/FLAT/d.fits", "nonsense"]
    assert B.nights_from_paths(paths) == 2


def test_build_carries_progress_except_for_monitors():
    rows = ROWS + [{"dso": "v1", "status": "waiting", "lightcurve": True}]
    reg = dict(REG, v1={"state": "ACQUIRING", "priority": 50, "legacy_status": "waiting"})
    calls = []

    def pf(dso, row):
        calls.append(dso)
        return {"pct": 48, "limiting": "O-III"}
    b = B.build(reg, rows, CONV, PLAN, DRAFTS, {}, set(), FLAGS, NOW, progress_fn=pf)
    by = {t["dso"]: t for t in b["targets"]}
    assert by["ic410"]["progress"] == {"pct": 48, "limiting": "O-III"}
    assert by["v1"]["progress"] is None and "v1" not in calls
    assert B.build(REG, ROWS, CONV, PLAN, DRAFTS, {}, set(), FLAGS, NOW)["targets"][0]["progress"] is None
