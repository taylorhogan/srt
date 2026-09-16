"""Phase 2: the conductor decides the roof; the actuators ask and report.

Driven the way the live system drives it: a temp repo root with the legacy
files, a conductor constructed on it (roof authority on or off), and the
same POSTs iris/client.py makes -- a request with the evidence just sensed,
then the confirmation or timeout. The pure client decision table is tested
first, on the pytest-only CI interpreter; the conductor tests need only the
stdlib too. The HTTP layer is exercised through FastAPI's TestClient where
it is installed.
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris import client
from iris.conductor.shadow import (FALLBACK_MARKER, ROOF_MOTION_TIMEOUT_S,
                                   ShadowConductor)
from iris.core.journal import Journal

GOOD = {"parked_vision": "CONFIRMED", "parked_kasa": "CONFIRMED", "roof": "DENIED"}
BLIND = {"parked_vision": "UNKNOWN", "parked_kasa": "UNKNOWN", "roof": "DENIED"}
OPEN = {"parked_vision": "CONFIRMED", "parked_kasa": "CONFIRMED", "roof": "CONFIRMED"}


# ------------------------------------------------------------ client (pure)

def test_decide_unreachable_refuses_only_under_authority():
    assert client.decide(None, authority=True)[0] is False
    allowed, reason = client.decide(None, authority=False)
    assert allowed is True and "unreachable" in reason


def test_decide_accepted_is_allowed_and_would_refuse_is_advisory():
    assert client.decide({"accepted": True}, True) == (True, None)
    allowed, reason = client.decide({"accepted": True, "would_refuse": "mount_parked: x"}, False)
    assert allowed is True and reason.startswith("conductor would have refused")


def test_decide_refusal_binds_only_under_authority():
    reply = {"accepted": False, "guard": "mount_parked: scope not confirmed parked"}
    assert client.decide(reply, True) == (False, reply["guard"])
    allowed, reason = client.decide(reply, False)
    assert allowed is True and reply["guard"] in reason


def test_evidence_from_vision_never_denies_on_its_own():
    """visual_status's parked=False is 'could not confirm', not 'off park'."""
    e = client.evidence_from_vision(False, True, False)
    assert e["parked_vision"] == "UNKNOWN" and e["roof"] == "DENIED"
    e = client.evidence_from_vision(True, False, True)
    assert e["parked_vision"] == "CONFIRMED" and e["roof"] == "CONFIRMED"
    assert client.evidence_from_vision(True, False, False)["roof"] == "UNKNOWN"
    assert client.asserted_evidence("open")["roof"] == "DENIED"
    assert client.asserted_evidence("close")["asserted"] is True


# ------------------------------------------------------------ harness

def _mkroot(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scheduler_state.json").write_text(
        json.dumps({"state": "WAITING_FOR_NOON", "dso": "Unknown",
                    "will image tonight": "Unknown"}))
    (tmp_path / "imaging.txt").write_text("IMAGING_STATE NONE")
    (tmp_path / "safety.txt").write_text("USER SAFE")
    (tmp_path / "mode.txt").write_text("MODE AUTO")
    (tmp_path / "iris.log").write_text("")
    (tmp_path / "local").mkdir()
    return tmp_path


@pytest.fixture(autouse=True)
def _no_real_nina(monkeypatch):
    monkeypatch.setattr(ShadowConductor, "_nina_running", lambda self: False)


def _conductor(root, authority=True):
    c = ShadowConductor(root, Journal(root / "local" / "journal"),
                        sun_probe=lambda: None, roof_authority=authority)
    c.pwi4_probe = lambda: "unreachable"
    c.limits_probe = lambda: ("not_configured", "")
    return c


def _entries(c, kind=None):
    return [e for e in c.journal.replay() if kind is None or e.kind == kind]


# ------------------------------------------------------------ authority on

def test_manual_open_and_close_walk_the_machine_on_posted_evidence(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD)
    assert v.kind == "transition" and v.state == "MANUAL_OPENING"
    e = _entries(c, "transition")[-1]
    assert e.data["guards"] == "enforced"
    assert e.data["evidence_posted"]["parked_vision"] == "CONFIRMED"
    v = c.offer("ROOF_OPEN_CONFIRMED", "operator", {"confirmed": True}, evidence=OPEN)
    assert v.state == "MANUAL_OPEN"
    v = c.offer("ROOF_CLOSE_REQUESTED", "operator", {}, evidence=OPEN)
    assert v.kind == "transition" and v.state == "MANUAL_CLOSING"
    v = c.offer("ROOF_CLOSE_CONFIRMED", "operator", {"confirmed": True}, evidence=GOOD)
    assert v.state == "IDLE_DAY"


def test_blind_evidence_is_refused_and_journaled(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=BLIND)
    assert v.kind == "rejected" and v.state == "IDLE_DAY"
    assert v.guard.startswith("mount_parked:")
    rej = _entries(c, "rejected")
    assert len(rej) == 1 and rej[0].guard == v.guard
    assert rej[0].data["evidence_posted"]["parked_vision"] == "UNKNOWN"


def test_a_kasa_denial_vetoes_even_with_vision_confirmed(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {},
                evidence={**GOOD, "parked_kasa": "DENIED"})
    assert v.kind == "rejected" and "OFF park" in v.guard


def test_unknown_roof_position_refuses_a_toggle(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {"direction": "unknown"},
                evidence={**GOOD, "roof": "UNKNOWN"})
    assert v.kind == "rejected" and v.guard.startswith("roof_state_known:")


def test_a_request_with_no_row_is_a_refusal_not_a_move(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    c.state = "SLOT_IMAGING"
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD)
    assert v.kind == "ignored" and c.state == "SLOT_IMAGING"
    assert "no transition for ROOF_OPEN_REQUESTED in state SLOT_IMAGING" == v.guard


def test_the_nights_own_close_is_decided_on_the_posted_park(tmp_path):
    """end.py: NIGHT_END_REQUESTED (unguarded) then MOUNT_PARK_CONFIRMED with
    the vision read; the close is refused when the read cannot confirm."""
    c = _conductor(_mkroot(tmp_path))
    c.state, c.slots = "SLOT_IMAGING", 0
    assert c.offer("NIGHT_END_REQUESTED", "end.py").state == "PARKING"
    v = c.offer("MOUNT_PARK_CONFIRMED", "end.py", {}, evidence={**OPEN, "parked_vision": "UNKNOWN"})
    assert v.kind == "rejected" and c.state == "PARKING"
    v = c.offer("MOUNT_PARK_CONFIRMED", "end.py", {}, evidence=OPEN)
    assert v.state == "CLOSING_ROOF"
    assert c.offer("ROOF_CLOSE_CONFIRMED", "end.py", {"confirmed": True}).state == "FLATS"


def test_unplanned_operator_run_opens_from_idle_with_its_own_weather(tmp_path):
    root = _mkroot(tmp_path)
    (root / "scheduler_state.json").write_text(
        json.dumps({"state": "WAITING_FOR_NOON", "will image tonight": "false", "slots": []}))
    c = _conductor(root)
    v = c.offer("CHECKS_PASSED", "operator", {}, evidence=GOOD)
    assert v.kind == "transition" and v.state == "OPENING_ROOF"
    e = _entries(c, "transition")[-1]
    assert e.data["unplanned"] is True and e.data["weather"] == "operator's call"
    assert c.slots == 1
    # The scheduler's own run on such a day is NOT its own weather verdict.
    c2 = _conductor(_mkroot(tmp_path / "b"))
    c2.state = "ARMED"
    assert c2.offer("CHECKS_PASSED", "scheduler", {}, evidence=GOOD).kind == "rejected"


def test_live_posts_make_the_synthesized_steps_unnecessary(tmp_path):
    """With the run posting CHECKS_PASSED and ROOF_OPEN_CONFIRMED itself, the
    IN_PRELUDE edge must not journal a second, ignored copy of each."""
    root = _mkroot(tmp_path)
    c = _conductor(root)
    c.offer("CHECKS_PASSED", "operator", {}, evidence=GOOD)
    c.offer("ROOF_OPEN_CONFIRMED", "operator", {"confirmed": True}, evidence=OPEN)
    assert c.state == "PRELUDE"
    (root / "imaging.txt").write_text("IMAGING_STATE ACTIVE"); c.poll()
    (root / "imaging.txt").write_text("IMAGING_STATE IN_PRELUDE"); c.poll()
    assert c.state == "PRELUDE"
    ignored = [e for e in _entries(c, "note") if e.data.get("ignored_in_state")]
    assert not ignored
    assert len(_entries(c, "transition")) == 2


# ------------------------------------------------------------ faults + holds

def test_motion_that_outlives_the_confirm_loop_faults_and_holds(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD)
    c._state_since = time.time() - ROOF_MOTION_TIMEOUT_S - 1
    c.poll()
    assert c.state == "FAULT_ROOF_UNKNOWN"
    tr = _entries(c, "transition")[-1]
    assert tr.event == "ROOF_TIMEOUT" and tr.source == "watchdog"
    # Held: opening is refused until an operator resolves; closing (the safe
    # direction) is answered by the guards alone and the hold is kept.
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD)
    assert v.kind == "ignored" and c.state == "FAULT_ROOF_UNKNOWN"
    v = c.offer("ROOF_CLOSE_REQUESTED", "operator", {}, evidence=OPEN)
    assert v.accepted and c.state == "FAULT_ROOF_UNKNOWN"
    v = c.offer("OPERATOR_RESOLVE", "operator", {"account": "t"})
    assert v.state == "IDLE_DAY"


def test_an_unconfirmed_move_reported_by_the_actuator_faults(tmp_path):
    c = _conductor(_mkroot(tmp_path))
    c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD)
    v = c.offer("ROOF_TIMEOUT", "operator", {"direction": "open", "confirmed": False})
    assert v.state == "FAULT_ROOF_UNKNOWN"


def test_a_restart_mid_move_faults_and_the_fault_survives_restart(tmp_path):
    root = _mkroot(tmp_path)
    c = _conductor(root)
    c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD)
    assert c.state == "MANUAL_OPENING"
    c2 = _conductor(root)
    assert c2.state == "FAULT_ROOF_UNKNOWN"
    assert _entries(c2, "transition")[-1].event == "ROOF_TIMEOUT"
    c3 = _conductor(root)
    assert c3.state == "FAULT_ROOF_UNKNOWN"      # persisted, not re-derived


def test_fallback_marker_confirmed_is_a_note_unconfirmed_is_a_fault(tmp_path):
    root = _mkroot(tmp_path)
    marker = root / FALLBACK_MARKER
    marker.write_text(json.dumps({"who": "end.py", "confirmed_closed": True}))
    c = _conductor(root)
    assert c.state == "IDLE_DAY"
    assert any(e.event == "ROOF_MOVED_WITHOUT_CONDUCTOR" for e in _entries(c, "note"))
    assert not marker.exists() and list(root.glob("local/roof_fallback_marker.ingested-*.json"))

    root2 = _mkroot(tmp_path / "b")
    (root2 / FALLBACK_MARKER).write_text(json.dumps({"who": "end.py", "confirmed_closed": False}))
    c2 = _conductor(root2)
    assert c2.state == "FAULT_ROOF_UNKNOWN"
    assert _entries(c2, "transition")[-1].event == "VISION_CONTRADICTION"


# ------------------------------------------------------------ authority off

def test_without_authority_a_bad_request_moves_and_carries_the_verdict(tmp_path):
    c = _conductor(_mkroot(tmp_path), authority=False)
    v = c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=BLIND)
    assert v.kind == "transition" and v.state == "MANUAL_OPENING"
    assert v.would_refuse and v.would_refuse.startswith("mount_parked:")
    e = _entries(c, "transition")[-1]
    assert e.data["guard_would"] == v.would_refuse and "guards" not in e.data
    # The client turns that reply into "allowed, advisory".
    reply = {"accepted": True, "would_refuse": v.would_refuse}
    allowed, reason = client.decide(reply, False)
    assert allowed and v.would_refuse in reason


# ------------------------------------------------------------ HTTP surface

def test_api_reports_kind_guard_and_authority(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from iris.core.api import build_app
    c = _conductor(_mkroot(tmp_path))
    app = build_app(c, c.journal, lambda: {})
    t = TestClient(app)
    st = t.get("/v1/state").json()
    assert st["authority"] == {"roof": True} and st["shadow"] is False
    r = t.post("/v1/events", json={"event": "ROOF_OPEN_REQUESTED", "source": "operator",
                                   "evidence": BLIND}).json()
    assert r["accepted"] is False and r["kind"] == "rejected" and r["guard"].startswith("mount_parked")
    r = t.post("/v1/events", json={"event": "ROOF_OPEN_REQUESTED", "source": "operator",
                                   "evidence": GOOD}).json()
    assert r["accepted"] is True and r["state"] == "MANUAL_OPENING" and r["was"] == "IDLE_DAY"


# ------------------------------------------------------------ end.py fallback writer

def test_end_py_fallback_marker_round_trips(tmp_path):
    end = pytest.importorskip("end_points.end")
    root = _mkroot(tmp_path)
    p = end.write_fallback_marker(False, {"parked_vision": "CONFIRMED", "ts": 1.0}, root=str(root))
    info = json.loads(Path(p).read_text())
    assert info["confirmed_closed"] is False and "ts" not in info["evidence"]
    c = _conductor(root)
    assert c.state == "FAULT_ROOF_UNKNOWN"


# ------------------------------------------------------------ closing in a hold

def test_close_in_a_hold_is_decided_by_the_guards_and_keeps_the_hold(tmp_path):
    """stop! -> SAFE_HOLD -> the emergency close asks. Allowed on good
    evidence, refused on blind evidence, and the hold is never left."""
    c = _conductor(_mkroot(tmp_path))
    c.state = "SAFE_HOLD"
    v = c.offer("ROOF_CLOSE_REQUESTED", "end.py", {}, evidence=OPEN)
    assert v.accepted and v.kind == "allowed_in_hold" and c.state == "SAFE_HOLD"
    assert _entries(c, "note")[-1].event == "ROOF_CLOSE_ALLOWED_IN_HOLD"
    v = c.offer("ROOF_CLOSE_REQUESTED", "end.py", {}, evidence={**OPEN, "parked_vision": "UNKNOWN"})
    assert not v.accepted and v.kind == "rejected" and c.state == "SAFE_HOLD"
    # Opening in a hold stays refused: that needs resolve! / safe! first.
    assert not c.offer("ROOF_OPEN_REQUESTED", "operator", {}, evidence=GOOD).accepted
    c.state = "FAULT_ROOF_UNKNOWN"
    assert c.offer("ROOF_CLOSE_REQUESTED", "operator", {}, evidence=OPEN).accepted
    assert c.state == "FAULT_ROOF_UNKNOWN"


def test_end_py_retry_closes_a_night_the_machine_lost(tmp_path):
    """end.py posts MOUNT_PARK_CONFIRMED; with the machine idle that has no
    row, so it asks for a plain close, which IDLE_DAY answers."""
    c = _conductor(_mkroot(tmp_path))
    v = c.offer("MOUNT_PARK_CONFIRMED", "end.py", {}, evidence=OPEN)
    assert v.kind == "ignored" and not v.accepted
    v = c.offer("ROOF_CLOSE_REQUESTED", "end.py", {"why": "not in PARKING"}, evidence=OPEN)
    assert v.accepted and v.state == "MANUAL_CLOSING"
    assert c.offer("ROOF_CLOSE_CONFIRMED", "end.py", {"confirmed": True}).state == "IDLE_DAY"
