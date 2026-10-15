"""control/night_plan_status: the plan sidecar the site reads (2026-10-15)."""
from datetime import datetime, timedelta, timezone

from control import night_plan_status as S

TZ = timezone(timedelta(hours=-4))
NOON = datetime(2026, 10, 15, 12, 1, tzinfo=TZ)


def test_write_read_and_age(tmp_path):
    p = tmp_path / "night_plan_status.json"
    S.write(p, "imaging", "noon", "C:/x/night_timeline.jpg", title="Tonight: ngc6946",
            summary="Roof opens ~18:05", when=NOON)
    doc = S.read(p, now=NOON + timedelta(hours=6))
    assert doc["verdict"] == "imaging" and doc["check"] == "noon" and doc["title"] == "Tonight: ngc6946"
    assert S.read(p, now=NOON + timedelta(hours=21)) is None          # yesterday's plan: not tonight's
    assert S.read(p, max_age_h=None)["image"] == "C:/x/night_timeline.jpg"


def test_restamp_keeps_the_plan_and_changes_the_check(tmp_path):
    p = tmp_path / "s.json"
    S.write(p, "imaging", "noon", "img.jpg", title="T", summary="S", when=NOON)
    doc = S.restamp(p, "sunset", when=NOON + timedelta(hours=6))
    assert doc["check"] == "sunset" and doc["title"] == "T" and doc["image"] == "img.jpg"
    assert S.restamp(tmp_path / "missing.json", "sunset") is None


def test_skip_and_unreadable(tmp_path):
    p = tmp_path / "s.json"
    S.write(p, "skip", "sunset", None, reason="conditions degraded", when=NOON)
    assert S.read(p, now=NOON)["reason"] == "conditions degraded"
    p.write_text("{not json")
    assert S.read(p) is None
    assert S.read(tmp_path / "none.json") is None
