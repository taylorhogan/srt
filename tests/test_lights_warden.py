"""end_points/lights_warden: lights off ten minutes before the first exposure (2026-10-09)."""
import json
from datetime import datetime, timedelta, timezone

from end_points import lights_warden as lw

TZ = timezone(timedelta(hours=-4))
SUNSET = datetime(2026, 10, 8, 18, 16, tzinfo=TZ)
DUSK = datetime(2026, 10, 8, 19, 21, tzinfo=TZ)        # nautical, measured night


def test_first_exposure_matches_the_measured_night():
    # 2026-10-08: dusk 19:21, slot wait +5, centering/focus, first frame 19:34:36
    first = lw.first_exposure(DUSK, timedelta(minutes=5))
    assert abs((first - datetime(2026, 10, 8, 19, 34, 36, tzinfo=TZ)).total_seconds()) < 120


def test_due_is_lead_before_first_exposure():
    due = lw.due_at(SUNSET, DUSK, SUNSET - timedelta(hours=1))
    assert due == DUSK + timedelta(minutes=5) + lw.SETUP - lw.LEAD == datetime(2026, 10, 8, 19, 24, tzinfo=TZ)


def test_due_never_before_the_sunset_schedules_have_fired():
    early_dusk = SUNSET - timedelta(minutes=30)             # absurd geometry: dusk before sunset
    assert lw.due_at(SUNSET, early_dusk, SUNSET - timedelta(hours=1)) == SUNSET + lw.AFTER_SUNSET


def test_due_is_now_when_imaging_starts_late():
    late = DUSK + timedelta(hours=2)
    assert lw.due_at(SUNSET, DUSK, late) == late


def test_wait_offset_read_from_the_sequence(tmp_path):
    seq = {"$id": "1", "Items": [
        {"$id": "2", "$type": "NINA.Sequencer.SequenceItem.Utility.WaitForTime, NINA.Sequencer",
         "MinutesOffset": -10, "SelectedProvider": {"$type": "NINA.Sequencer.Utility.DateTimeProvider.SunsetProvider, NINA.Sequencer"}},
        {"$id": "3", "$type": "NINA.Sequencer.SequenceItem.Utility.WaitForTime, NINA.Sequencer",
         "MinutesOffset": 7, "SelectedProvider": {"$type": "NINA.Sequencer.Utility.DateTimeProvider.NauticalDuskProvider, NINA.Sequencer"}}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(seq))
    assert lw.sequence_wait_offset(p) == timedelta(minutes=7)
    assert lw.sequence_wait_offset(tmp_path / "missing.json") == lw.DEFAULT_WAIT_OFFSET


def test_texts():
    due = datetime(2026, 10, 8, 19, 24, tzinfo=TZ); first = datetime(2026, 10, 8, 19, 34, tzinfo=TZ)
    w = lw.warning_text(due, SUNSET, DUSK, first, ("A", "B"))
    assert "19:34" in w and "19:21" in w and "18:16" in w and "19:24" in w and "A, B" in w
    assert lw.result_text({"A": True, "B": True}) == "Lights: all 2 off (verified)"
    assert lw.result_text({"A": True, "B": False}) == "Lights: 1 of 2 off; could not switch B"


def test_start_sequence_uses_the_same_list():
    src = open(lw.__file__.replace("lights_warden.py", "start.py"), encoding="utf-8").read()
    assert "lights_warden" in src and "LIGHTS" in src


def test_lights_list_is_lights_only():
    assert "Telescope mount" not in lw.LIGHTS and "Roof motor" not in lw.LIGHTS
    assert {"SWAN", "Stairs", "Driveway lights"} <= set(lw.LIGHTS)
