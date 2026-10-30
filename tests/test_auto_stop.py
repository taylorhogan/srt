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


def test_converged_alone_is_done_and_says_so():
    v = A.verdict("m1", True, PLAN_ADVICE)
    assert v["done"] and v["why"] == ["convergence: every filter at its flat tail (is_dso_done)"]


def test_frame_goals_alone_are_done_with_the_counts():
    v = A.verdict("ic410", False, DONE_ADVICE)
    assert v["done"] and v["why"] == ["frame goals met (filter advisor): Ha 44, O-III 41"]


def test_both_verdicts_keep_both_reasons():
    assert len(A.verdict("m1", True, DONE_ADVICE)["why"]) == 2


def test_neither_is_not_done():
    assert A.verdict("m1", False, PLAN_ADVICE) == {"dso": "m1", "done": False, "why": []}
    assert A.verdict("m1", False, None)["done"] is False


def test_a_light_curve_monitor_is_never_done():
    v = A.verdict("hubblev1", True, MONITOR)
    assert v["done"] is False and "monitor" in v["why"][0]


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

    found = A.candidates(ROWS, lambda d: d == "m1", advise)
    assert [v["dso"] for v in found] == ["m1", "ic410"]
    # held, completed and the monitor were never evaluated
    assert seen == ["m1", "ic410", "trunk"]


def test_a_crashing_verdict_errs_toward_imaging_once_more():
    def boom(_):
        raise RuntimeError("convergence.json unreadable")
    found = A.candidates(ROWS, boom, lambda d, r: DONE_ADVICE if d == "trunk" else None)
    assert [v["dso"] for v in found] == ["trunk"]           # the advisor still counted
    found = A.candidates(ROWS, lambda d: False, lambda d, r: 1 / 0)
    assert found == []


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
