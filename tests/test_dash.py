"""The Core2 desk display's status (cmd_processing/dash.py, 2026-10-22).

Read-only by construction; these pin down what the small screen is told,
above all that it never says SHUT or OPEN on a read the gate would not trust.
"""
from datetime import datetime
from types import SimpleNamespace

from cmd_processing import dash

SUMMARY = "Tonight: m76\nRoof opens ~17:52, closes ~06:05\nm76 22:00"
PLAN = {"noon": {"plan": {"will_image": True, "slots": [
    {"dso": "ngc7790", "start": "2026-10-22T20:00-04:00"},
    {"dso": "m76", "start": "2026-10-22T22:00-04:00"}]}}, "sunset": None}


def _roof(v, at="2026-10-22 09:00"):
    return {"v": v, "at": datetime.fromisoformat(at)}


def _sum(now="2026-10-22 13:00", roof=None, scope="?", state="ARMED", disk=388.0, live=None):
    return dash.summarize(datetime.fromisoformat(now), roof, scope,
                          {"state": state} if state else None, PLAN, SUMMARY, live, disk)


def test_the_newest_read_decides_and_needs_every_frame():
    names = ["084605_0_shut.jpg", "090017_0_shut.jpg", "090017_1_shut_ir.jpg", "090017_2_shut.jpg"]
    r = dash.roof_from_names("2026-10-22", names)
    assert r["v"] == "SHUT" and r["at"] == datetime(2026, 10, 22, 9, 0, 17)
    mixed = dash.roof_from_names("2026-10-22", ["090017_0_shut.jpg", "090017_1_absent.jpg"])
    assert mixed["v"] == "?"                                     # one frame did not see it
    assert dash.roof_from_names("2026-10-22", ["090017_0_absent.jpg"])["v"] == "?"
    assert dash.roof_from_names("2026-10-22", ["notes.txt"]) is None


def test_scope_is_read_never_connected():
    def st(conn, slew=False, track=False):
        return SimpleNamespace(mount=SimpleNamespace(is_connected=conn, is_slewing=slew, is_tracking=track))
    assert dash.scope_from_status(None) == "?"
    assert dash.scope_from_status(st(False)) == "OFF"
    assert dash.scope_from_status(st(True, slew=True)) == "SLEWING"
    assert dash.scope_from_status(st(True, track=True)) == "TRACKING"
    assert dash.scope_from_status(st(True)) == "STOPPED"


def test_a_quiet_afternoon_is_all_green():
    d = _sum(roof=_roof("SHUT"))
    assert d["level"] == "g" and d["roof"] == {"v": "SHUT", "age": 240, "c": "g"}
    assert d["scope"] == {"v": "PWI4 OFF", "c": "g"}            # normal by day
    assert d["next"] == {"dso": "ngc7790", "at": "20:00"}
    assert d["tonight"]["opens"] == "17:52" and d["tonight"]["closes"] == "06:05"


def test_open_inside_the_window_is_fine_outside_it_is_flagged():
    assert _sum("2026-10-22 23:00", _roof("OPEN", "2026-10-22 22:50"), "TRACKING")["roof"]["c"] == "g"
    assert _sum("2026-10-23 02:00", _roof("OPEN", "2026-10-23 01:50"), "TRACKING")["roof"]["c"] == "g"
    d = _sum("2026-10-23 06:20", _roof("OPEN", "2026-10-23 06:15"))
    assert d["roof"]["c"] == "a" and d["level"] == "a" and "roof open off-plan" in d["alerts"]
    d = _sum("2026-10-23 08:00", _roof("OPEN", "2026-10-23 07:55"))     # 2 h past the close
    assert d["roof"]["c"] == "r" and d["level"] == "r" and "roof open past close" in d["alerts"]
    assert _sum("2026-10-22 15:00", _roof("OPEN", "2026-10-22 14:55"))["roof"]["c"] == "r"
    assert _sum("2026-10-22 15:00", _roof("OPEN", "2026-10-22 14:55"), state="MANUAL_OPEN")["roof"]["c"] == "a"


def test_a_hold_promises_no_next_slot():
    assert _sum("2026-10-22 19:00", state="ESTOP")["next"] is None
    assert _sum("2026-10-22 19:00")["next"] == {"dso": "ngc7790", "at": "20:00"}


def test_no_pwi4_at_night_is_worth_a_look():
    assert _sum("2026-10-22 23:00", _roof("OPEN", "2026-10-22 22:50"))["scope"]["c"] == "a"


def test_holds_and_a_dead_conductor_are_red():
    assert _sum(state="ESTOP")["level"] == "r"
    d = _sum(state=None)
    assert d["night"] == {"v": "NO CONDUCTOR", "c": "r"} and d["level"] == "r"


def test_disk_follows_the_morning_check_thresholds():
    assert _sum(disk=150.0)["disk"]["c"] == "g"
    assert _sum(disk=60.0)["disk"]["c"] == "a"
    assert _sum(disk=20.0)["level"] == "r"


def test_after_the_last_slot_starts_there_is_no_next():
    assert _sum("2026-10-22 23:00")["next"] is None


def test_unknown_roof_is_amber():
    assert _sum(roof=None)["roof"] == {"v": "?", "age": None, "c": "a"}
