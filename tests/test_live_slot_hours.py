"""The live panel's "usable hours" is the running target's OWN slot.

2026-10-20: the panel read good_hours from tonight_target.json, which is the
night's best target only, so ngc6946 -- the first of four slots -- showed
"0 usable hours" while it imaged.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
ls = pytest.importorskip("live_skymap")

PLAN = {"noon": {"plan": {"slots": [{"dso": "ngc6946", "hours": 3}, {"dso": "ngc7790", "hours": 1},
                                    {"dso": "hubblev1", "hours": 1}, {"dso": "ngc1961", "hours": 5}]}},
        "sunset": None}


def test_each_slot_reports_its_own_hours():
    assert ls.slot_hours_from_plan(PLAN, "ngc6946") == 3
    assert ls.slot_hours_from_plan(PLAN, "NGC1961") == 5


def test_the_sunset_record_wins_and_unknowns_are_none():
    reply = dict(PLAN, sunset={"plan": {"slots": [{"dso": "ngc6946", "hours": 2}]}})
    assert ls.slot_hours_from_plan(reply, "ngc6946") == 2
    assert ls.slot_hours_from_plan(PLAN, "m33") is None
    assert ls.slot_hours_from_plan(None, "ngc6946") is None


# 2026-10-23: the chart marks every target of the night, not just the running one.
SEQ = {"Items": [{"Target": {"TargetName": "m76", "InputCoordinates": {
    "RAHours": 1, "RAMinutes": 42, "RASeconds": 19.7, "DecDegrees": 51, "DecMinutes": 34,
    "DecSeconds": 31.0, "NegativeDec": False}}},
    {"Target": {"TargetName": "SouthThing", "InputCoordinates": {
        "RAHours": 6, "RAMinutes": 0, "RASeconds": 0.0, "DecDegrees": 5, "DecMinutes": 30,
        "DecSeconds": 0.0, "NegativeDec": True}}}]}
SLOTS = {"noon": {"plan": {"slots": [
    {"dso": "ngc7790", "start": "2026-10-22T20:00-04:00", "end": "2026-10-22T21:00-04:00"},
    {"dso": "m76", "start": "2026-10-22T22:00-04:00", "end": "2026-10-23T04:00-04:00"},
    {"dso": "souththing", "start": "2026-10-23T04:00-04:00", "end": "2026-10-23T06:00-04:00"}]}}}


def test_sequence_targets_reads_where_the_mount_is_sent():
    t = ls.sequence_targets(SEQ)
    assert t["m76"] == pytest.approx((25.582, 51.5753), abs=1e-3)
    assert t["souththing"] == pytest.approx((90.0, -5.5))


def test_slot_marks_skip_unknown_targets_and_mark_finished_slots():
    from datetime import datetime
    marks = ls.slot_marks(SLOTS, ls.sequence_targets(SEQ), datetime(2026, 10, 23, 5, 0))
    assert [(m[0], m[3], m[4]) for m in marks] == [("m76", "22:00", True), ("souththing", "04:00", False)]
    assert ls.slot_marks(None, {}, datetime(2026, 10, 23)) == []
