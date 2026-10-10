"""Phase 2b (2026-10-10): the two roof cameras and the blind-park evidence in the snapshot.

The merge rule moves to iris/core/guards.roof_by_cameras; it must equal the
rule in force (sentry/north_roof.decide) on every input before the guards may
read it. Until then the conductor only journals it beside its decision.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from iris.core import guards as G
from iris.core.snapshot import SensorSnapshot, Tri, roof_word, unmoved_since_open

WORDS = {"open": Tri.CONFIRMED, "shut": Tri.DENIED, "unknown": Tri.UNKNOWN}


def test_roof_word():
    assert roof_word("open") is Tri.CONFIRMED and roof_word("shut") is Tri.DENIED
    assert roof_word("closed") is Tri.DENIED and roof_word("unread") is Tri.UNKNOWN
    assert roof_word(None) is Tri.UNKNOWN


@pytest.mark.parametrize("north", ["open", "shut", "unknown"])
@pytest.mark.parametrize("cam", ["open", "shut", "unknown"])
def test_roof_by_cameras_equals_the_rule_in_force(north, cam):
    north_roof = pytest.importorskip("sentry.north_roof")
    closed, is_open, _src = north_roof.decide(north, cam == "shut", cam == "open")
    expected = Tri.CONFIRMED if is_open else (Tri.DENIED if closed else Tri.UNKNOWN)
    s = SensorSnapshot(roof_north=WORDS[north], roof_cam=WORDS[cam])
    assert G.roof_by_cameras(s) is expected, (north, cam)


def test_roof_by_cameras_rule_stated_without_the_legacy_module():
    s = SensorSnapshot
    assert G.roof_by_cameras(s(roof_north=Tri.CONFIRMED)) is Tri.CONFIRMED           # north decides
    assert G.roof_by_cameras(s(roof_north=Tri.CONFIRMED, roof_cam=Tri.DENIED)) is Tri.UNKNOWN  # veto
    assert G.roof_by_cameras(s(roof_north=Tri.DENIED, roof_cam=Tri.CONFIRMED)) is Tri.UNKNOWN
    assert G.roof_by_cameras(s(roof_cam=Tri.DENIED)) is Tri.DENIED                    # fallback
    assert G.roof_by_cameras(s()) is Tri.UNKNOWN


NOW = datetime(2026, 10, 10, 0, 33, tzinfo=timezone.utc)


def test_unmoved_since_open():
    opened = NOW - timedelta(hours=5)
    assert unmoved_since_open(None, None, NOW) is Tri.UNKNOWN
    assert unmoved_since_open(opened, opened - timedelta(minutes=2), NOW) is Tri.CONFIRMED   # its own fire
    assert unmoved_since_open(opened, None, NOW) is Tri.CONFIRMED
    assert unmoved_since_open(opened, opened + timedelta(minutes=1), NOW) is Tri.DENIED      # moved since
    assert unmoved_since_open(NOW - timedelta(hours=17), None, NOW) is Tri.DENIED            # too old
    assert G.blind_park_corroborated(SensorSnapshot(roof_unmoved=Tri.CONFIRMED)) is None
    assert "no evidence" in G.blind_park_corroborated(SensorSnapshot(roof_unmoved=Tri.DENIED))


# ------------------------------------------------------------ the conductor

from iris.conductor.shadow import ShadowConductor   # noqa: E402
from iris.core.journal import Journal               # noqa: E402

GOOD = {"parked_vision": "CONFIRMED", "parked_kasa": "CONFIRMED", "roof": "DENIED"}
OPEN = {"parked_vision": "CONFIRMED", "parked_kasa": "CONFIRMED", "roof": "CONFIRMED"}


@pytest.fixture(autouse=True)
def _no_real_nina(monkeypatch):
    monkeypatch.setattr(ShadowConductor, "_nina_running", lambda self: False)


def _conductor(tmp_path):
    for name, text in (("scheduler_state.json", json.dumps({"state": "WAITING_FOR_NOON", "dso": "x",
                                                            "will image tonight": "Unknown"})),
                       ("imaging.txt", "IMAGING_STATE NONE"), ("safety.txt", "USER SAFE"),
                       ("mode.txt", "MODE AUTO"), ("iris.log", "")):
        (tmp_path / name).write_text(text)
    (tmp_path / "local").mkdir()
    c = ShadowConductor(tmp_path, Journal(tmp_path / "local" / "journal"), sun_probe=lambda: None,
                        roof_authority=True, mount_authority=True)
    c.pwi4_probe = lambda: "unreachable"
    c.limits_probe = lambda: ("not_configured", "")
    return c


def _notes(c, event):
    return [e for e in c.journal.replay() if e.kind == "note" and e.event == event]


def test_posted_camera_answers_are_journaled_beside_the_decision(tmp_path):
    c = _conductor(tmp_path)
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {},
                evidence={**GOOD, "roof_north": "DENIED", "roof_cam": "DENIED"})
    assert v.kind == "transition"
    e = [x for x in c.journal.replay() if x.event == "ROOF_OPEN_REQUESTED"][-1]
    assert e.data["roof_cameras"] == {"north": "DENIED", "cam": "DENIED", "by_guards": "DENIED",
                                      "decided": "DENIED", "unmoved_since_open": "UNKNOWN"}
    assert _notes(c, "ROOF_MERGE_DIFF") == []


def test_a_merge_difference_is_its_own_note_and_changes_nothing(tmp_path):
    c = _conductor(tmp_path)
    # decided says shut, the cameras disagree with each other -> the guards' merge is UNKNOWN
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {},
                evidence={**GOOD, "roof_north": "DENIED", "roof_cam": "CONFIRMED"})
    assert v.kind == "transition"                      # verdict still from `roof`
    diff = _notes(c, "ROOF_MERGE_DIFF")
    assert len(diff) == 1 and diff[0].data["by_guards"] == "UNKNOWN" and diff[0].data["decided"] == "DENIED"


def test_the_north_row_in_the_log_feeds_the_snapshot(tmp_path):
    c = _conductor(tmp_path)
    (tmp_path / "iris.log").write_text(
        "10/10/2026 08:25:50 AM north roof: tag says shut, Iris cam says shut -> shut (by north+cam)\n")
    c.poll()
    ev = c._current_evidence()
    assert ev.roof_north is Tri.DENIED and ev.roof_cam is Tri.DENIED


def test_blind_park_corroboration_is_journaled_not_enforced(tmp_path):
    c = _conductor(tmp_path)
    c.offer("ESTOP_REQUESTED", "operator")
    blind = {**OPEN, "roof": "UNKNOWN"}
    v = c.offer("MOUNT_MOVE_REQUESTED", "stop!", {"direction": "park", "blind_park": True}, evidence=blind)
    assert v.accepted                                   # no record on file: still allowed, as before
    note = _notes(c, "MOUNT_PARK_ALLOWED_IN_HOLD")[-1]
    assert note.data["blind_park_corroborated"] is False and "no evidence" in note.data["blind_park_corroboration"]
    opened = datetime.now().astimezone() - timedelta(hours=2)
    (tmp_path / "local" / "roof_evidence.json").write_text(json.dumps(
        {"open_confirmed": opened.isoformat(), "motion_possible": (opened - timedelta(minutes=2)).isoformat()}))
    c.offer("MOUNT_MOVE_REQUESTED", "stop!", {"direction": "park", "blind_park": True}, evidence=blind)
    assert _notes(c, "MOUNT_PARK_ALLOWED_IN_HOLD")[-1].data["blind_park_corroborated"] is True


def test_client_posts_both_roof_cameras_from_a_fresh_vision_read(monkeypatch):
    import sys
    import time
    import types
    from iris import client
    fake = types.SimpleNamespace(last_match={"at": time.time(), "north": {"state": "open"}, "cam_roof": "unknown"})
    monkeypatch.setitem(sys.modules, "sentry.vision_safety", fake)
    e = client.evidence_from_vision(True, False, True)
    assert e["roof_north"] == "CONFIRMED" and e["roof_cam"] == "UNKNOWN" and e["roof"] == "CONFIRMED"
    fake.last_match["at"] = time.time() - 600            # stale: not posted
    assert "roof_north" not in client.evidence_from_vision(True, False, True)
    monkeypatch.delitem(sys.modules, "sentry.vision_safety")
    assert "roof_north" not in client.evidence_from_vision(True, False, True)
