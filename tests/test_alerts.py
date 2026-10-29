"""iris/conductor/alerts -- the bell (docs/AUTOMATION_PLAN.md B1, B2).

What must hold: a transition into a hold is ONE emergency push with the
reason and the day's north frame; a hold entered from a hold, or the same
hold after a restart, pushes nothing more; a blocking event is one emergency;
the noon reminder is one priority-1 line a day while held; disabled is
silent. The conductor and the clock are driven for real on a tmp journal.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris.conductor import alerts as A  # noqa: E402
from iris.conductor.shadow import ShadowConductor  # noqa: E402
from iris.core.journal import Journal  # noqa: E402

NOW = datetime(2026, 10, 29, 22, 12)


class Bell:
    def __init__(self, tmp_path, now=NOW, enabled=True, frame=None):
        self.pushes = []
        self.now = now
        self.frame = frame
        self.a = A.Alerter(tmp_path, push=self._push, picture_fn=lambda day: self.frame,
                           now_fn=lambda: self.now, enabled=enabled, worker=False)

    def _push(self, message, image, priority):
        self.pushes.append((message, image, priority))
        return True


def test_entering_a_hold_is_one_emergency_with_the_picture(tmp_path):
    frame = tmp_path / "north.jpg"
    frame.write_bytes(b"jpg")
    b = Bell(tmp_path, frame=str(frame))
    b.a.on_transition("ROOF_STALL", "CLOSING_ROOF", "FAULT_ROOF_UNKNOWN", {"why": "motor 40 W past 30 s"})
    assert len(b.pushes) == 1
    msg, image, pri = b.pushes[0]
    assert pri == 2 and image == str(frame)
    assert msg.startswith("Iris HOLD: FAULT_ROOF_UNKNOWN after ROOF_STALL (motor 40 W past 30 s)")
    assert "resolve!" in msg


def test_a_hold_from_a_hold_and_a_restart_in_the_hold_stay_quiet(tmp_path):
    b = Bell(tmp_path)
    b.a.on_transition("ESTOP_REQUESTED", "SLOT_IMAGING", "ESTOP", {})
    b.a.on_transition("ESTOP_REQUESTED", "SAFE_HOLD", "ESTOP", {})          # hold -> hold
    assert len(b.pushes) == 1
    again = Bell(tmp_path)                                                   # conductor restarted
    again.a.on_transition("ESTOP_REQUESTED", "SLOT_IMAGING", "ESTOP", {})
    assert again.pushes == []
    state = json.loads((tmp_path / "local" / "alerts_state.json").read_text())
    assert state == {"hold:ESTOP": "2026-10-29"}


def test_a_new_day_rings_again(tmp_path):
    b = Bell(tmp_path)
    b.a.on_transition("ESTOP_REQUESTED", "IDLE_DAY", "ESTOP", {})
    b.now = datetime(2026, 10, 30, 9, 0)
    b.a.on_transition("OPERATOR_RESOLVE", "ESTOP", "IDLE_DAY", {})           # leaving: nothing
    b.a.on_transition("ESTOP_REQUESTED", "IDLE_DAY", "ESTOP", {})
    assert len(b.pushes) == 2


def test_blocking_events_are_emergencies_without_a_picture(tmp_path):
    b = Bell(tmp_path, frame="/nowhere.jpg")
    b.a.on_transition("CHECKS_FAILED", "PRE_FLIGHT", "IDLE_DAY", {"why": "preflight: PWI4 unreachable"})
    b.a.on_transition("PLAN_BAD", "PLANNING", "IDLE_DAY", {"why": "disk: 20 GB free"})
    b.a.on_transition("PLAN_BAD", "PLANNING", "IDLE_DAY", {"dso": "m33"})    # weather: routine
    b.a.on_transition("NOON_TICK", "IDLE_DAY", "PLANNING", {})
    assert [(p, i) for _, i, p in b.pushes] == [(2, None), (2, None)]
    assert "pre-flight checks failed" in b.pushes[0][0] and "PWI4" in b.pushes[0][0]
    assert "called off" in b.pushes[1][0] and "disk" in b.pushes[1][0]


def test_the_noon_reminder_is_one_line_a_day_and_only_in_a_hold(tmp_path):
    b = Bell(tmp_path)
    b.a.noon_reminder("FAULT_ROOF_UNKNOWN", datetime(2026, 10, 28, 6, 1), "Disk: 80 GB free")
    b.a.noon_reminder("FAULT_ROOF_UNKNOWN", datetime(2026, 10, 28, 6, 1))
    b.a.noon_reminder("IDLE_DAY")
    assert len(b.pushes) == 1
    msg, image, pri = b.pushes[0]
    assert pri == 1 and image is None
    assert "still in FAULT_ROOF_UNKNOWN since Wed 28 Oct 06:01" in msg and "Disk: 80 GB" in msg


def test_disabled_is_silent_and_a_failing_push_never_raises(tmp_path):
    b = Bell(tmp_path, enabled=False)
    b.a.on_transition("ESTOP_REQUESTED", "IDLE_DAY", "ESTOP", {})
    assert b.pushes == []

    def boom(*a):
        raise RuntimeError("pushover down")
    a = A.Alerter(tmp_path, push=boom, picture_fn=lambda d: None, now_fn=lambda: NOW, worker=False)
    a.on_transition("ESTOP_REQUESTED", "IDLE_DAY", "ESTOP", {})          # no exception


def test_the_worker_thread_delivers(tmp_path):
    got = []
    a = A.Alerter(tmp_path, push=lambda m, i, p: got.append(p) or True,
                  picture_fn=lambda d: None, now_fn=lambda: NOW)
    a.on_transition("ESTOP_REQUESTED", "IDLE_DAY", "ESTOP", {})
    a.flush()
    a._q.join()
    assert got == [2]


def test_latest_north_frame_is_todays_newest_jpg(tmp_path):
    d = tmp_path / "local" / "north_archive" / "2026-10-29"
    d.mkdir(parents=True)
    (d / "061501_0_SHUT.jpg").write_bytes(b"a")
    (d / "121501_2_OPEN_ir.jpg").write_bytes(b"b")
    (d / "notes.txt").write_text("x")
    assert A.latest_north_frame(tmp_path, NOW.date()).endswith("121501_2_OPEN_ir.jpg")
    assert A.latest_north_frame(tmp_path, datetime(2026, 1, 1).date()) is None


def test_the_conductor_rings_the_bell_on_a_real_estop(tmp_path, monkeypatch):
    (tmp_path / "local").mkdir()
    b = Bell(tmp_path)
    sh = ShadowConductor(tmp_path, Journal(tmp_path / "local" / "journal"),
                         sun_probe=lambda: None, alerter=b.a)
    sh.pwi4_probe = lambda: "unreachable"
    sh.limits_probe = lambda: ("not_configured", "")
    sh.offer("ESTOP_REQUESTED", "operator", {"why": "stop!"})
    assert sh.state == "ESTOP"
    assert len(b.pushes) == 1 and b.pushes[0][2] == 2
    assert "Iris HOLD: ESTOP after ESTOP_REQUESTED (stop!)" in b.pushes[0][0]
    # a conductor built without an alerter is as silent as before
    quiet = ShadowConductor(tmp_path, Journal(tmp_path / "local" / "journal2"), sun_probe=lambda: None)
    quiet.offer("ESTOP_REQUESTED", "operator", {})
    assert quiet.state == "ESTOP"


def test_from_config_reads_the_alerts_block(tmp_path):
    a = A.Alerter.from_config({"alerts": {"enabled": False, "hold_priority": 1}}, tmp_path)
    assert a.enabled is False and a.hold_priority == 1
    assert A.Alerter.from_config({}, tmp_path).enabled is True
