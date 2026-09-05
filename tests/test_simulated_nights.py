"""Simulated failure nights — the scenarios CI can run that the sky rarely provides on demand.

Each test is a story told in events, walking the machine through a night that
goes wrong in a specific way. These are the nights the cutover drills will
later reproduce against fakes with real actuator plumbing; here they pin the
TABLE's behaviour, which is the part that must be right first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.core.machine import INITIAL_STATE, step
from iris.core.snapshot import SensorSnapshot, Tri

GO = SensorSnapshot(parked_vision=Tri.CONFIRMED, parked_kasa=Tri.CONFIRMED,
                    parked_pwi4=Tri.CONFIRMED,
                    roof=Tri.DENIED, safety_armed=True, mode_auto=True,
                    weather_ok=True, slots_remaining=1, nina_alive=True)
# Mid-slot reality: scope TRACKING (not parked), roof confirmed open.
MID_SLOT = GO.replace(parked_vision=Tri.UNKNOWN, parked_kasa=Tri.UNKNOWN,
                      parked_pwi4=Tri.DENIED,
                      roof=Tri.CONFIRMED)


def _walk(script, start=INITIAL_STATE):
    s = start
    for event, snap, expected in script:
        out = step(s, event, snap)
        assert out.kind == "transition", (s, event, out)
        assert out.state == expected, (s, event, out.state, expected)
        s = out.state
    return s


def _to_mid_slot():
    return _walk([
        ("NOON_TICK", GO, "PLANNING"), ("PLAN_GOOD", GO, "ARMED"),
        ("PRE_SUNSET_TICK", GO, "PRE_FLIGHT"),
        ("CHECKS_PASSED", GO, "OPENING_ROOF"),
        ("ROOF_OPEN_CONFIRMED", GO, "PRELUDE"),
        ("NINA_PRELUDE_DONE", GO, "SLOT_SETUP"),
        ("SLOT_STARTED", GO, "SLOT_IMAGING"),
    ])


def test_nina_dies_mid_slot_and_the_night_closes_safely():
    """The scenario the old system handles worst (doit_cmd's poll loop spins
    forever). Here: CAPTURE_LOST is the close decision (unguarded -- deciding
    to go home is always allowed), the park confirmation is the pivot, and
    Invariant A sits between the confirmation and the relay."""
    s = _to_mid_slot()
    # capture dies while the scope is tracking: entering PARKING needs no guard
    out = step(s, "CAPTURE_LOST", MID_SLOT)
    assert (out.kind, out.state) == ("transition", "PARKING")
    # a park CLAIM with the scope still not confirmed parked is REFUSED --
    # this is the exact edge Invariant A exists for
    out2 = step("PARKING", "MOUNT_PARK_CONFIRMED", MID_SLOT)
    assert out2.kind == "rejected"
    assert "parked" in out2.guard
    # once both sensors confirm, the close proceeds
    parked = MID_SLOT.replace(parked_vision=Tri.CONFIRMED,
                                     parked_kasa=Tri.CONFIRMED,
                                     parked_pwi4=Tri.CONFIRMED)
    _walk([
        ("MOUNT_PARK_CONFIRMED", parked, "CLOSING_ROOF"),
        ("ROOF_CLOSE_CONFIRMED", parked, "FLATS"),
        ("NINA_FLATS_DONE", parked, "SHUTDOWN"),
        ("SHUTDOWN_DONE", parked, "NIGHT_DONE"),
    ], start="PARKING")


def test_weather_abort_mid_imaging_parks_closes_then_takes_flats():
    """Weather ends the imaging, not the flats: the roof shuts first and the
    flats are shot against a panel behind it, which is the order end.py runs
    and the order a real night was measured in on 2026-09-05."""
    s = _to_mid_slot()
    s = _walk([
        ("WEATHER_BAD", MID_SLOT, "PARKING"),
    ], start=s)
    parked = MID_SLOT.replace(parked_vision=Tri.CONFIRMED,
                                     parked_kasa=Tri.CONFIRMED,
                                     parked_pwi4=Tri.CONFIRMED)
    _walk([("MOUNT_PARK_CONFIRMED", parked, "CLOSING_ROOF"),
           ("ROOF_CLOSE_CONFIRMED", parked, "FLATS"),
           ("NINA_FLATS_DONE", parked, "SHUTDOWN")], start=s)


def test_estop_mid_slot_holds_until_resolved():
    s = _to_mid_slot()
    out = step(s, "ESTOP_REQUESTED", MID_SLOT)     # nothing confirmed; still accepted
    assert out.state == "ESTOP"
    # everything but the operator bounces off
    for ev in ("NOON_TICK", "ROOF_CLOSE_CONFIRMED", "SAFETY_ARMED", "DAY_TICK"):
        assert step("ESTOP", ev, GO).kind == "ignored"
    assert step("ESTOP", "OPERATOR_RESOLVE", GO).state == "IDLE_DAY"


def test_open_roof_discovered_at_idle_faults():
    """After an ESTOP resolve (or any anomaly) the machine sits IDLE_DAY,
    which CLAIMS the roof is closed. If vision then reports it open, that
    contradiction must land in the persistent fault, not be shrugged off --
    IDLE_DAY quietly coexisting with an open roof is how rain meets optics."""
    out = step("IDLE_DAY", "VISION_CONTRADICTION", GO.replace(roof=Tri.CONFIRMED))
    assert (out.kind, out.state) == ("transition", "FAULT_ROOF_UNKNOWN")
    # same claim in ARMED and NIGHT_DONE
    assert step("ARMED", "VISION_CONTRADICTION", GO).state == "FAULT_ROOF_UNKNOWN"
    assert step("NIGHT_DONE", "VISION_CONTRADICTION", GO).state == "FAULT_ROOF_UNKNOWN"


def test_two_slot_night_with_window_end_reslew():
    """Slot 1 ends because its target set behind the trees (SLOT_WINDOW_END),
    not because frames completed -- the small-horizon case the multi-slot
    design exists for."""
    go2 = GO.replace(slots_remaining=2)
    go1 = GO.replace(slots_remaining=1)
    go0 = GO.replace(slots_remaining=0)
    s = _walk([
        ("NOON_TICK", go2, "PLANNING"), ("PLAN_GOOD", go2, "ARMED"),
        ("PRE_SUNSET_TICK", go2, "PRE_FLIGHT"),
        ("CHECKS_PASSED", go2, "OPENING_ROOF"),
        ("ROOF_OPEN_CONFIRMED", go2, "PRELUDE"),
        ("NINA_PRELUDE_DONE", go2, "SLOT_SETUP"),
        ("SLOT_STARTED", go2, "SLOT_IMAGING"),
        ("SLOT_WINDOW_END", go1, "SLOT_SETUP"),      # target set; one slot left
        ("SLOT_STARTED", go1, "SLOT_IMAGING"),
        ("NINA_SLOT_DONE", go0, "PARKING"),
        ("MOUNT_PARK_CONFIRMED", go0, "CLOSING_ROOF"),
        ("ROOF_CLOSE_CONFIRMED", go0, "FLATS"),
        ("NINA_FLATS_DONE", go0, "SHUTDOWN"),
        ("SHUTDOWN_DONE", go0, "NIGHT_DONE"),
        ("DAY_TICK", go0, "IDLE_DAY"),
    ])
    assert s == "IDLE_DAY"


def test_safety_cleared_mid_slot_holds_and_never_reenters_the_night():
    s = _to_mid_slot()
    out = step(s, "SAFETY_CLEARED", MID_SLOT)
    assert out.state == "SAFE_HOLD"
    # re-arming returns to IDLE_DAY -- NOT back into the night. The roof may
    # physically still be open at that moment; the IDLE_DAY contradiction row
    # (tested above) is what catches that, and in live operation the resolve
    # procedure closes the roof first. The machine refuses to pretend the
    # night can resume.
    out = step("SAFE_HOLD", "SAFETY_ARMED", GO)
    assert out.state == "IDLE_DAY"


def test_roof_timeout_on_open_is_a_persistent_fault_not_a_retry():
    s = _walk([
        ("NOON_TICK", GO, "PLANNING"), ("PLAN_GOOD", GO, "ARMED"),
        ("PRE_SUNSET_TICK", GO, "PRE_FLIGHT"),
        ("CHECKS_PASSED", GO, "OPENING_ROOF"),
    ])
    out = step(s, "ROOF_TIMEOUT", GO.replace(roof=Tri.UNKNOWN))
    assert out.state == "FAULT_ROOF_UNKNOWN"
    # the fault does not clear itself when a later vision read looks fine
    assert step("FAULT_ROOF_UNKNOWN", "ROOF_CLOSE_CONFIRMED", GO).kind == "ignored"
