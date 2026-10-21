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
