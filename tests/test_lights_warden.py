"""end_points/lights_warden: the sunset + 15 min lights-off pass (2026-10-09)."""
from datetime import datetime, timedelta, timezone

from end_points import lights_warden as lw

TZ = timezone(timedelta(hours=-4))
SUNSET = datetime(2026, 10, 9, 18, 19, tzinfo=TZ)


def test_due_is_sunset_plus_fifteen_or_now_if_later():
    assert lw.due_at(SUNSET, SUNSET - timedelta(hours=1)) == SUNSET + timedelta(minutes=15)
    late = SUNSET + timedelta(hours=2)
    assert lw.due_at(SUNSET, late) == late


def test_texts():
    w = lw.warning_text(SUNSET + timedelta(minutes=15), SUNSET, ("A", "B"))
    assert "18:34" in w and "18:19" in w and "A, B" in w and "2 outside lights" in w
    assert lw.result_text({"A": True, "B": True}) == "Lights: all 2 off (verified)"
    assert lw.result_text({"A": True, "B": False}) == "Lights: 1 of 2 off; could not switch B"


def test_start_sequence_uses_the_same_list():
    import re
    src = open(lw.__file__.replace("lights_warden.py", "start.py"), encoding="utf-8").read()
    assert "lights_warden" in src and "LIGHTS" in src
    for name in lw.LIGHTS:
        assert name not in src.split("LIGHTS")[0] or name in ("Telescope mount",), name


def test_lights_list_is_lights_only():
    assert "Telescope mount" not in lw.LIGHTS and "Roof motor" not in lw.LIGHTS
    assert {"SWAN", "Stairs", "Driveway lights"} <= set(lw.LIGHTS)
