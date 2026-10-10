"""Phase 3 step 1 (2026-10-10): the night's decision in one place, planned in shadow.

control/night_plan is the noon / pre-sunset decision the scheduler (through
best_object_tonight) and the conductor's shadow planner both run; the
planner journals whether the conductor's plan matches the scheduler's.
"""
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from control import night_plan as NP

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=-4))
T0 = datetime(2026, 10, 10, 21, 0, tzinfo=TZ)


def _row(name, good):
    return (name, good, 60.0, "galaxy", T0, "", 1)


def _slot(name, start_h, end_h, good):
    return SimpleNamespace(name=name, start=T0 + timedelta(hours=start_h),
                           end=T0 + timedelta(hours=end_h), good_hours=good)


SLOTS = [_slot("ngc7320", 0, 3, 3), _slot("hubblev1", 4, 5, 1)]
RECORD = [{"dso": "ngc7320", "start": "2026-10-10T21:00-04:00", "end": "2026-10-11T00:00-04:00", "hours": 3},
          {"dso": "hubblev1", "start": "2026-10-11T01:00-04:00", "end": "2026-10-11T02:00-04:00", "hours": 1}]


def test_min_good_hours_is_the_schedulers():
    src = (ROOT / "end_points" / "scheduler_server.py").read_text(encoding="utf-8")
    m = re.search(r"^_MIN_GOOD_HOURS\s*=\s*(\d+)", src, re.M)
    assert m and int(m.group(1)) == NP.MIN_GOOD_HOURS


def test_queue_caps_only_waiting_entries_that_carry_one():
    q = [{"dso": "hubblev1", "status": "waiting", "max_hours": 1},
         {"dso": "m31", "status": "completed", "max_hours": 2},
         {"dso": "ngc7320", "status": "waiting"}]
    assert NP.queue_caps(q) == {"hubblev1": 1.0}
    assert NP.queue_caps(None) == {}


def test_read_queue_caps_survives_a_missing_queue(tmp_path):
    assert NP.read_queue_caps(tmp_path / "nope.json") == {}
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"dso": "hubblev1", "status": "waiting", "max_hours": "1.5"}]))
    assert NP.read_queue_caps(p) == {"hubblev1": 1.5}


def test_slots_record_is_the_scheduler_state_shape():
    assert NP.slots_record(SLOTS) == RECORD


def test_decide():
    assert NP.decide([], [])["will_image"] is False
    on = NP.decide([_row("ngc7320", 3), _row("m33", 5)], SLOTS)
    assert on["best"] == "ngc7320" and on["will_image"] is True and on["slots"] == RECORD
    assert on["best_start"] == "2026-10-10T21:00-04:00"
    off = NP.decide([_row("ngc7320", 2)], SLOTS)
    assert off["will_image"] is False and off["slots"] == [] and off["good_hours"] == 2


def _sched(dso="ngc7320", will=True, slots=RECORD):
    return {"state": "WAITING_FOR_PRE_SUNSET", "dso": dso, "will image tonight": will, "slots": slots}


def test_compare():
    plan = NP.decide([_row("ngc7320", 3)], SLOTS)
    assert NP.compare(plan, _sched()) == []
    assert NP.compare(plan, _sched(will="True")) == []
    assert any("best target" in d for d in NP.compare(plan, _sched(dso="m33")))
    assert any("will image" in d for d in NP.compare(plan, _sched(will=False)))
    assert any("slots" in d for d in NP.compare(plan, _sched(slots=RECORD[:1])))
    # A night off on both sides: stale slots left in the file are not a diff.
    off = NP.decide([_row("ngc7320", 2)], SLOTS)
    assert NP.compare(off, _sched(will=False)) == []
    # The file before any decision ("Unknown") is not a night on.
    assert any("will image" in d for d in NP.compare(plan, _sched(will="Unknown")))


# ------------------------------------------------------------ the planner

from iris.conductor.planner import ShadowPlanner, parse_plan   # noqa: E402
from iris.core.journal import Journal                         # noqa: E402


def _notes(journal, *events):
    return [e for e in journal.replay() if e.kind == "note" and e.event in events]


def test_parse_plan_takes_the_last_json_line():
    plan = NP.decide([_row("ngc7320", 3)], SLOTS)
    out = "cloud cover 21:00: 10%\nNWS grid ok\n" + json.dumps(plan) + "\n"
    assert parse_plan(out) == plan
    with pytest.raises(ValueError):
        parse_plan("no plan here\n{not json\n")


def _planner(tmp_path, runner):
    j = Journal(tmp_path / "journal")
    return ShadowPlanner(tmp_path, j, runner=runner), j


def test_a_matching_plan_is_journaled_as_a_match(tmp_path):
    plan = NP.decide([_row("ngc7320", 3)], SLOTS)
    p, j = _planner(tmp_path, lambda: plan)
    p.on_decision("NOON_CHECK", _sched()).join(5)
    (e,) = _notes(j, "PLAN_MATCH", "PLAN_DIFF")
    assert e.event == "PLAN_MATCH" and e.data["check"] == "NOON_CHECK" and e.data["diffs"] == []
    assert e.data["scheduler"]["slots"] == RECORD


def test_a_different_plan_is_journaled_with_its_diffs(tmp_path):
    plan = NP.decide([_row("m33", 5)], [_slot("m33", 0, 5, 5)])
    p, j = _planner(tmp_path, lambda: plan)
    p.on_decision("PRE_SUNSET_CHECK", _sched()).join(5)
    (e,) = _notes(j, "PLAN_MATCH", "PLAN_DIFF")
    assert e.event == "PLAN_DIFF" and any("best target" in d for d in e.data["diffs"])


def test_a_failed_plan_is_journaled_and_frees_the_planner(tmp_path):
    def boom():
        raise RuntimeError("forecast down")
    p, j = _planner(tmp_path, boom)
    p.on_decision("NOON_CHECK", _sched()).join(5)
    (e,) = _notes(j, "SHADOW_PLAN_FAILED")
    assert "forecast down" in e.data["error"]
    p.runner = lambda: NP.decide([_row("ngc7320", 3)], SLOTS)
    p.on_decision("NOON_CHECK", _sched()).join(5)
    assert _notes(j, "PLAN_MATCH")


def test_a_second_decision_while_planning_is_skipped(tmp_path):
    release = threading.Event()

    def slow():
        release.wait(5)
        return NP.decide([_row("ngc7320", 3)], SLOTS)
    p, j = _planner(tmp_path, slow)
    t = p.on_decision("NOON_CHECK", _sched())
    assert p.on_decision("PRE_SUNSET_CHECK", _sched()) is None
    release.set()
    t.join(5)
    assert [e.event for e in _notes(j, "SHADOW_PLAN_SKIPPED", "PLAN_MATCH")] == ["SHADOW_PLAN_SKIPPED", "PLAN_MATCH"]


# ------------------------------------------------------------ the conductor

from iris.conductor.shadow import ShadowConductor   # noqa: E402


class _FakePlanner:
    def __init__(self):
        self.calls = []

    def on_decision(self, check, scheduler):
        self.calls.append((check, scheduler))


@pytest.fixture
def conductor(tmp_path, monkeypatch):
    monkeypatch.setattr(ShadowConductor, "_nina_running", lambda self: False)
    for name, text in (("scheduler_state.json", json.dumps({"state": "WAITING_FOR_NOON", "dso": "x",
                                                            "will image tonight": "Unknown"})),
                       ("imaging.txt", "IMAGING_STATE NONE"), ("safety.txt", "USER SAFE"),
                       ("mode.txt", "MODE AUTO"), ("iris.log", "")):
        (tmp_path / name).write_text(text)
    (tmp_path / "local").mkdir()
    planner = _FakePlanner()
    c = ShadowConductor(tmp_path, Journal(tmp_path / "local" / "journal"), sun_probe=lambda: None,
                        roof_authority=True, mount_authority=True, planner=planner)
    c.pwi4_probe = lambda: "unreachable"
    c.limits_probe = lambda: ("not_configured", "")
    return c, planner, tmp_path


def _write_sched(root, **d):
    (root / "scheduler_state.json").write_text(json.dumps(d))


def test_leaving_each_check_plans_in_shadow(conductor):
    c, planner, root = conductor
    _write_sched(root, state="NOON_CHECK", dso="x", **{"will image tonight": "Unknown"})
    c.poll()
    assert planner.calls == []                       # entering the check: nothing decided yet
    _write_sched(root, state="WAITING_FOR_PRE_SUNSET", dso="ngc7320", slots=RECORD,
                 **{"will image tonight": True})
    c.poll()
    assert [ck for ck, _ in planner.calls] == ["NOON_CHECK"]
    assert planner.calls[0][1]["slots"] == RECORD and planner.calls[0][1]["dso"] == "ngc7320"
    _write_sched(root, state="PRE_SUNSET_CHECK", dso="ngc7320", slots=RECORD, **{"will image tonight": True})
    c.poll()
    _write_sched(root, state="IMAGING", dso="ngc7320", slots=RECORD, **{"will image tonight": True})
    c.poll()
    assert [ck for ck, _ in planner.calls] == ["NOON_CHECK", "PRE_SUNSET_CHECK"]


def test_a_night_off_is_planned_too(conductor):
    c, planner, root = conductor
    _write_sched(root, state="NOON_CHECK", dso="x", **{"will image tonight": "Unknown"})
    c.poll()
    _write_sched(root, state="WAITING_FOR_NOON", dso="ngc7320", **{"will image tonight": False})
    c.poll()
    assert [ck for ck, _ in planner.calls] == ["NOON_CHECK"]


def test_a_broken_planner_never_stops_the_poll(conductor):
    c, planner, root = conductor

    def boom(check, scheduler):
        raise RuntimeError("planner bug")
    planner.on_decision = boom
    _write_sched(root, state="NOON_CHECK", dso="x", **{"will image tonight": "Unknown"})
    c.poll()
    _write_sched(root, state="WAITING_FOR_PRE_SUNSET", dso="ngc7320", slots=RECORD,
                 **{"will image tonight": True})
    c.poll()                                          # does not raise
    assert c.state == "ARMED"                         # PLAN_GOOD still stepped the machine
