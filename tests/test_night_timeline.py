"""control.night_timeline: the noon report as time | event (2026-10-08)."""
from datetime import datetime, timedelta, timezone

from control import night_timeline as nt

TZ = timezone(timedelta(hours=-4))
SUNSET = datetime(2026, 10, 8, 18, 20, tzinfo=TZ)


def _two_slots():
    exp = {"L": 300.0, "R": 300.0, "G": 300.0, "B": 300.0}
    return [nt.SlotPlan("ngc7320", datetime(2026, 10, 8, 20, 0, tzinfo=TZ), datetime(2026, 10, 9, 1, 0, tzinfo=TZ),
                        {"L": 18, "R": 9, "G": 9, "B": 9}, exp),
            nt.SlotPlan("ngc2146", datetime(2026, 10, 9, 3, 0, tzinfo=TZ), datetime(2026, 10, 9, 6, 0, tzinfo=TZ),
                        {"B": 3, "L": 11, "G": 3, "R": 7}, exp)]


def test_rows_in_time_order_with_roof_flats_and_wrap_up():
    rows = nt.build(SUNSET, _two_slots(), TZ)
    events = [e for _t, _a, e in rows]
    assert events[0].startswith("Imaging check")
    assert any(e.startswith("Roof opens") for e in events)
    assert events[-1].startswith("All wrapped up")
    times = [t for t, _a, _e in rows]
    assert times == sorted(times)
    t = {e.split(":")[0].split(" starts")[0]: tt for tt, _a, e in rows}
    assert t["Imaging check (weather and plan re-checked)"] == datetime(2026, 10, 8, 18, 10, tzinfo=TZ)


def test_wrap_up_offsets_follow_the_last_slot_end():
    rows = nt.build(SUNSET, _two_slots(), TZ)
    by = {e: t for t, _a, e in rows}
    end = datetime(2026, 10, 9, 6, 0, tzinfo=TZ)
    assert by["Scope parked, roof closed"] == end + nt.ROOF_CLOSED_AFTER_END
    assert by["Flats done"] == end + nt.ROOF_CLOSED_AFTER_END + nt.FLATS


def test_filters_listed_in_wheel_order_with_total():
    text = nt.render(nt.build(SUNSET, _two_slots(), TZ))
    assert "ngc2146 starts: L 11, R 7, G 3, B 3 x 300s (2.0h)" in text
    assert "~06:05  Scope parked, roof closed" in text
    assert text.splitlines()[1] == "Time    Event"


def test_no_end_time_says_dawn():
    slot = nt.SlotPlan("m33", datetime(2026, 10, 8, 20, 0, tzinfo=TZ), None, {"L": 10}, {"L": 300.0})
    text = nt.render(nt.build(SUNSET, [slot], TZ))
    assert "dawn" in text and "Flats done" not in text
