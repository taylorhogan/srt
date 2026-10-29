"""The conductor's clock and planner (iris/conductor/clock.py, ADR 0015).

A whole day on a fake clock: the journal is real (tmp dir), the machine is
the real table stepped permissively, the planner is a stub. What must hold:
each check runs once per date whatever restarts happen, a failure retries
once and then says so, a hold plans nothing, and the scheduler-facing record
(NIGHT_PLAN) is there in every outcome.
"""
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.conductor import clock as C  # noqa: E402
from iris.core.journal import Journal  # noqa: E402
from iris.core.machine import step  # noqa: E402
from iris.core.snapshot import SensorSnapshot, Tri  # noqa: E402

TZ = timezone(timedelta(hours=-4))
DAY = date(2026, 10, 16)
SUNSET = datetime(2026, 10, 16, 18, 10, tzinfo=TZ)
PERMISSIVE = SensorSnapshot(parked_vision=Tri.CONFIRMED, parked_kasa=Tri.CONFIRMED,
                            parked_pwi4=Tri.CONFIRMED, roof=Tri.DENIED, safety_armed=True,
                            mode_auto=True, weather_ok=True, slots_remaining=1, nina_alive=True)


def at(h, m=0, d=DAY):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)


GOOD = {"best": "ngc7790", "good_hours": 6, "best_start": "2026-10-16T19:00-04:00",
        "will_image": True,
        "slots": [{"dso": "ngc7790", "start": "2026-10-16T19:00-04:00",
                   "end": "2026-10-17T01:00-04:00", "hours": 6, "priority": 50}]}
BAD = {"best": "m33", "good_hours": 1, "best_start": None, "will_image": False, "slots": []}


class FakeConductor:
    """The bits of ShadowConductor the driver uses, over the real table."""

    def __init__(self, journal, state="IDLE_DAY"):
        self.journal = journal
        self.state, self.slots = state, 0
        self._lock = threading.RLock()
        self._state_since = 0.0
        self._imaging, self._nina = "NONE", False

    def offer(self, event, source, data=None, evidence=None):
        with self._lock:
            out = step(self.state, event, PERMISSIVE.replace(slots_remaining=self.slots))
            if out.kind == "transition":
                self.journal.append("transition", event, source, from_state=self.state,
                                    to_state=out.state, data=dict(data or {}))
                self.state = out.state
            else:
                self.journal.append("note", event, source,
                                    data=dict(data or {}, ignored_in_state=self.state))
            return out


class Day:
    def __init__(self, tmp_path, state="IDLE_DAY", plans=(GOOD,)):
        self.journal = Journal(tmp_path / "journal")
        self.c = FakeConductor(self.journal, state)
        self.now = at(9)
        self.plans = list(plans)
        self.calls = []
        self.pushes = []
        self.driver = self.new_driver()

    def new_driver(self):
        return C.NightDriver(self.c, self.journal, Path("."), runner=self.runner,
                             now_fn=lambda: self.now, sunset_fn=lambda d: SUNSET,
                             push=lambda m: self.pushes.append(m) or True, retry_after_s=0)

    def runner(self, refresh):
        self.calls.append(refresh)
        p = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        if isinstance(p, Exception):
            raise p
        return p

    def tick(self, h=None, m=0):
        if h is not None:
            self.now = at(h, m)
        t = self.driver.tick()
        if t is not None:
            t.join(5)
        return t

    def recs(self):
        return C.plan_records(self.journal.replay(days=0), DAY)

    def events(self, name):
        return [e for e in self.journal.replay(days=0) if e.event == name]


# --------------------------------------------------------------- due(), pure

def test_due_walks_the_day():
    none = {"noon": None, "sunset": None}
    good = {"noon": {"outcome": "good"}, "sunset": None}
    assert C.due(at(11, 59), SUNSET, none) is None
    assert C.due(at(12, 0), SUNSET, none) == "noon"
    assert C.due(at(17, 59), SUNSET, none) == "noon"          # restarted in the afternoon
    assert C.due(at(18, 0), SUNSET, none) == "missed_noon"
    assert C.due(at(15), SUNSET, good) is None
    assert C.due(at(18, 0), SUNSET, good) == "sunset"
    assert C.due(at(20, 9), SUNSET, good) == "sunset"          # within the 2 h grace
    assert C.due(at(20, 11), SUNSET, good) == "missed_sunset"
    assert C.due(at(18, 0), SUNSET, {"noon": {"outcome": "bad"}, "sunset": None}) is None
    assert C.due(at(19), SUNSET, {"noon": {"outcome": "good"}, "sunset": {"outcome": "good"}}) is None


def test_plan_records_takes_the_latest_per_check_and_this_date_only(tmp_path):
    j = Journal(tmp_path)
    j.append("note", "NIGHT_PLAN", "clock", data={"date": "2026-10-15", "check": "noon", "outcome": "good"})
    j.append("note", "NIGHT_PLAN", "clock", data={"date": "2026-10-16", "check": "noon", "outcome": "bad"})
    j.append("note", "PLAN_MATCH", "planner", data={"date": "2026-10-16", "check": "noon"})
    j.append("note", "NIGHT_PLAN", "clock", data={"date": "2026-10-16", "check": "noon", "outcome": "good"})
    r = C.plan_records(j.replay(days=0), DAY)
    assert r["noon"]["outcome"] == "good" and r["noon"]["seq"] == 4 and r["sunset"] is None


# --------------------------------------------------------------- a whole day

def test_a_good_day_plans_noon_then_sunset_once_each(tmp_path):
    d = Day(tmp_path, plans=(GOOD, GOOD))
    assert d.tick(11, 59) is None and d.calls == []
    d.tick(12, 0)
    assert d.calls == [True]                                   # noon refreshes the queue
    assert d.c.state == "ARMED" and d.c.slots == 1
    assert d.recs()["noon"]["outcome"] == "good"
    assert d.recs()["noon"]["plan"]["best"] == "ngc7790"
    assert d.tick(15) is None and d.tick(17, 59) is None
    d.tick(18, 0)
    assert d.calls == [True, False]                            # sunset does not
    assert d.c.state == "PRE_FLIGHT" and d.recs()["sunset"]["outcome"] == "good"
    assert d.tick(18, 30) is None and len(d.calls) == 2
    assert [e.event for e in d.journal.replay(days=0) if e.kind == "transition"] == [
        "NOON_TICK", "PLAN_GOOD", "PRE_SUNSET_TICK"]


def test_a_restart_never_reruns_a_check_and_runs_a_due_one_once(tmp_path):
    d = Day(tmp_path, plans=(GOOD,))
    d.tick(12, 0)
    d.driver = d.new_driver()                                  # conductor restarted
    assert d.tick(14) is None and d.calls == [True]
    d.driver = d.new_driver()
    d.tick(18, 5)
    assert d.calls == [True, False]
    d.driver = d.new_driver()
    assert d.tick(19) is None and len(d.calls) == 2


def test_down_through_noon_plans_on_return_and_through_sunset_is_missed(tmp_path):
    d = Day(tmp_path, plans=(GOOD,))
    d.tick(15, 30)                                             # came back mid-afternoon
    assert d.calls == [True] and d.recs()["noon"]["outcome"] == "good"
    d.tick(21, 0)                                              # down through sunset + 2 h
    assert d.recs()["sunset"]["outcome"] == "missed" and len(d.calls) == 1


def test_down_until_after_the_sunset_check_records_a_missed_noon(tmp_path):
    d = Day(tmp_path)
    d.tick(18, 30)
    assert d.calls == [] and d.recs()["noon"]["outcome"] == "missed"


def test_a_bad_noon_goes_back_to_idle_and_skips_sunset(tmp_path):
    d = Day(tmp_path, plans=(BAD,))
    d.tick(12, 0)
    assert d.c.state == "IDLE_DAY" and d.recs()["noon"]["outcome"] == "bad"
    assert d.tick(18, 0) is None and d.calls == [True]


def test_a_bad_sunset_calls_the_night_off(tmp_path):
    d = Day(tmp_path, plans=(GOOD, BAD))
    d.tick(12, 0)
    d.tick(18, 0)
    assert d.recs()["sunset"]["outcome"] == "bad" and d.c.state == "IDLE_DAY"


# --------------------------------------------------------------- failures

def test_a_failure_retries_once_then_the_second_success_stands(tmp_path):
    d = Day(tmp_path, plans=(RuntimeError("SIMBAD hung"), GOOD))
    d.tick(12, 0)
    assert d.recs()["noon"] is None and len(d.events("NIGHT_PLAN_RETRY")) == 1
    assert d.c.state == "PLANNING"
    d.tick(12, 5)
    assert d.recs()["noon"]["outcome"] == "good" and d.recs()["noon"]["attempt"] == 2
    assert d.c.state == "ARMED" and d.pushes == []


def test_two_failures_are_failed_no_night_and_one_push(tmp_path):
    d = Day(tmp_path, plans=(TimeoutError("600 s"), TimeoutError("600 s")))
    d.tick(12, 0)
    d.tick(12, 5)
    rec = d.recs()["noon"]
    assert rec["outcome"] == "failed" and "TimeoutError" in rec["why"]
    assert d.c.state == "IDLE_DAY" and len(d.pushes) == 1
    assert d.tick(13) is None and d.tick(18, 0) is None and len(d.calls) == 2


def test_the_retry_waits_its_delay(tmp_path):
    d = Day(tmp_path, plans=(RuntimeError("x"), GOOD))
    d.driver.retry_after_s = 3600
    d.tick(12, 0)
    assert d.tick(12, 1) is None and len(d.calls) == 1


def test_a_sunset_failure_twice_calls_the_night_off(tmp_path):
    d = Day(tmp_path, plans=(GOOD, RuntimeError("a"), RuntimeError("b")))
    d.tick(12, 0)
    d.tick(18, 0)
    d.tick(18, 5)
    assert d.recs()["sunset"]["outcome"] == "failed" and d.c.state == "IDLE_DAY"
    assert len(d.pushes) == 1


# --------------------------------------------------------------- holds, replans, leftovers

def test_a_hold_at_noon_plans_nothing(tmp_path):
    d = Day(tmp_path, state="SAFE_HOLD")
    d.tick(12, 0)
    rec = d.recs()["noon"]
    assert d.calls == [] and rec["outcome"] == "skipped" and "SAFE_HOLD" in rec["why"]
    assert d.c.state == "SAFE_HOLD"


def test_yesterdays_leftover_state_is_reseated_at_noon(tmp_path):
    d = Day(tmp_path, state="NIGHT_DONE")
    d.tick(12, 0)
    assert len(d.events("SHADOW_RESYNC")) == 1 and d.c.state == "ARMED"


def test_a_daytime_run_at_noon_is_left_alone_and_still_planned(tmp_path):
    d = Day(tmp_path, state="SLOT_IMAGING")
    d.c._imaging = "IN_MAIN"
    d.tick(12, 0)
    assert d.c.state == "SLOT_IMAGING" and d.recs()["noon"]["outcome"] == "good"
    assert not d.events("SHADOW_RESYNC")


def test_a_manual_open_at_noon_is_not_reseated(tmp_path):
    d = Day(tmp_path, state="MANUAL_OPEN")
    d.tick(12, 0)
    assert d.c.state == "MANUAL_OPEN" and d.recs()["noon"]["outcome"] == "good"


def test_replan_makes_the_noon_decision_again(tmp_path):
    other = dict(GOOD, best="m33")
    d = Day(tmp_path, plans=(GOOD, other))
    d.tick(12, 0)
    first = d.recs()["noon"]["seq"]
    d.c.offer("REPLAN_REQUESTED", "chat")
    assert d.c.state == "PLANNING"
    d.tick(14, 0)
    rec = d.recs()["noon"]
    assert rec["seq"] > first and rec["plan"]["best"] == "m33" and rec["trigger"] == "replan"
    assert d.calls == [True, True] and d.c.state == "ARMED"


def test_replan_after_the_sunset_check_is_refused(tmp_path):
    d = Day(tmp_path, plans=(GOOD, BAD))
    d.tick(12, 0)
    d.tick(18, 0)                              # sunset bad -> IDLE_DAY
    d.c.offer("REPLAN_REQUESTED", "chat")
    d.tick(19, 0)
    assert len(d.events("REPLAN_REFUSED")) == 1 and d.c.state == "IDLE_DAY"
    assert len(d.calls) == 2


def test_a_planning_state_left_from_yesterday_is_refused_before_noon(tmp_path):
    d = Day(tmp_path, state="PLANNING")
    d.tick(9, 0)
    assert d.calls == [] and d.c.state == "IDLE_DAY"
    assert "planning window" in d.events("REPLAN_REFUSED")[0].data["why"]


def test_a_restart_inside_planning_finishes_the_plan(tmp_path):
    d = Day(tmp_path, state="PLANNING")
    d.tick(12, 3)
    assert d.recs()["noon"]["trigger"] == "recovery" and d.c.state == "ARMED"


# --------------------------------------------------------------- a hold (docs/AUTOMATION_PLAN.md B2)

def test_a_held_machine_gets_one_reminder_a_day_and_no_plan(tmp_path):
    from iris.conductor.alerts import Alerter
    d = Day(tmp_path, state="FAULT_ROOF_UNKNOWN")
    rings = []
    bell = Alerter(tmp_path, push=lambda m, i, p: rings.append((p, m)) or True,
                   picture_fn=lambda day: None, now_fn=lambda: d.now, worker=False)
    d.driver = C.NightDriver(d.c, d.journal, Path("."), runner=d.runner,
                             now_fn=lambda: d.now,
                             sunset_fn=lambda dd: datetime(dd.year, dd.month, dd.day, 18, 10, tzinfo=TZ),
                             push=lambda m: d.pushes.append(m) or True, retry_after_s=0,
                             alerts=bell)
    d.tick(12, 0)
    assert d.calls == [] and d.recs()["noon"]["outcome"] == "skipped"
    assert len(rings) == 1 and rings[0][0] == 1 and "still in FAULT_ROOF_UNKNOWN" in rings[0][1]
    d.tick(12, 5)                                                # same day: quiet
    assert len(rings) == 1
    d.now = at(12, 0, d=date(2026, 10, 17))
    d.tick()
    assert len(rings) == 2 and d.pushes == []                   # the old push path is untouched
