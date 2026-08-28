"""Exhaustive coverage of the Night machine.

Two layers. STRUCTURAL: every (state x event) pair is stepped and its outcome
classified, so nothing the machine could ever be asked is untested. INVARIANT:
the safety and liveness properties the architecture plan promises are asserted
as facts about the table, so a table edit that breaks one fails here before it
reaches the observatory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.core import guards as G
from iris.core.machine import (EVENTS, HOLD_STATES, INITIAL_STATE, STATES,
                               TRANSITIONS, Outcome, step)
from iris.core.snapshot import SensorSnapshot, Tri

# A snapshot in which every guard passes: night is go, scope parked and
# confirmed both ways, roof confirmed CLOSED (known), plan has slots.
ALL_GO = SensorSnapshot(parked_vision=Tri.CONFIRMED, parked_pwi4=Tri.CONFIRMED,
                        roof=Tri.DENIED, safety_armed=True, mode_auto=True,
                        weather_ok=True, slots_remaining=2, nina_alive=True)
# Its complement: nothing confirmed, nothing permitted.
ALL_STOP = SensorSnapshot()


# ---------------------------------------------------------------- table sanity

def test_table_references_only_declared_states_and_events():
    for row in TRANSITIONS:
        assert row.src == "*" or row.src in STATES, row
        assert row.dst in STATES, row
        assert row.event in EVENTS, row


def test_every_declared_event_appears_in_the_table():
    used = {r.event for r in TRANSITIONS}
    unused = set(EVENTS) - used
    assert not unused, f"declared but unused events: {unused}"


def test_states_have_no_duplicate_unguarded_rows():
    """Two unguarded rows for the same (state, event) would make the second
    unreachable — dead table weight that reads as a real transition."""
    seen = set()
    for row in TRANSITIONS:
        if row.guards:
            continue
        key = (row.src, row.event)
        assert key not in seen, f"duplicate unguarded row {key}"
        seen.add(key)


# ------------------------------------------------------- exhaustive stepping

def test_every_state_event_pair_is_deterministic_and_classified():
    """The exhaustive sweep: |STATES| x |EVENTS| pairs, under both the all-go
    and all-stop snapshots. Every outcome must be one of the three kinds, the
    resulting state must be a declared state, and stepping must be pure
    (same inputs -> same outcome)."""
    for state in STATES:
        for event in EVENTS:
            for snap in (ALL_GO, ALL_STOP):
                a = step(state, event, snap)
                b = step(state, event, snap)
                assert a == b, f"non-deterministic: {state} x {event}"
                assert a.kind in ("transition", "rejected", "ignored")
                assert a.state in STATES
                if a.kind == "rejected":
                    assert a.guard, "rejection without a reason"
                    assert a.state == state, "rejection must not move the machine"
                if a.kind == "ignored":
                    assert a.state == state


# ------------------------------------------------------------- safety facts

def test_estop_reaches_estop_from_every_non_hold_state():
    for state in STATES:
        out = step(state, "ESTOP_REQUESTED", ALL_STOP)   # even with nothing confirmed
        if state in HOLD_STATES:
            assert out.kind == "ignored", f"{state} must not wildcard out"
        else:
            assert out == Outcome("transition", "ESTOP"), state


def test_safe_hold_exits_only_by_operator_arming():
    """The restart-re-arm bug, made impossible: from SAFE_HOLD, the ONLY event
    that moves the machine at all is SAFETY_ARMED (an operator act)."""
    for event in EVENTS:
        out = step("SAFE_HOLD", event, ALL_GO)
        if event == "SAFETY_ARMED":
            assert out == Outcome("transition", "IDLE_DAY")
        else:
            assert out.kind == "ignored", f"SAFE_HOLD leaked via {event}"


def test_fault_states_exit_only_via_operator_resolve():
    for hold in ("FAULT_ROOF_UNKNOWN", "ESTOP"):
        for event in EVENTS:
            out = step(hold, event, ALL_GO)
            if event == "OPERATOR_RESOLVE":
                assert out.kind == "transition", (hold, event)
            else:
                assert out.kind == "ignored", f"{hold} leaked via {event}"


ROOF_MOVING_STATES = {"OPENING_ROOF", "CLOSING_ROOF"}


def test_invariant_a_every_entry_into_roof_motion_is_park_guarded():
    """No row may enter a roof-moving state without BOTH the parked guard and
    the roof-state-known guard. This is Invariant A as a property of the
    table itself, independent of any caller's discipline."""
    for row in TRANSITIONS:
        if row.dst in ROOF_MOVING_STATES:
            assert G.mount_parked in row.guards, f"{row} enters roof motion unparked"
            assert G.roof_state_known in row.guards, f"{row} enters roof motion blind"


def test_invariant_a_dynamic_no_snapshot_unparked_reaches_roof_motion():
    """Dynamic complement of the structural check: sweep every source state and
    every event under snapshots where the scope is NOT confirmed parked, and
    assert the machine never lands in a roof-moving state."""
    unparked = [ALL_GO.replace(parked_vision=v, parked_pwi4=p)
                for v in Tri for p in Tri
                if not (v is Tri.CONFIRMED and p is Tri.CONFIRMED)]
    for snap in unparked:
        for state in STATES:
            if state in ROOF_MOVING_STATES:
                continue          # already moving; entry is what is guarded
            for event in EVENTS:
                out = step(state, event, snap)
                assert not (out.kind == "transition" and out.state in ROOF_MOVING_STATES), \
                    f"unparked snapshot reached {out.state} via {state}+{event}"


def test_roof_unknown_never_enters_roof_motion():
    snap = ALL_GO.replace(roof=Tri.UNKNOWN)
    for state in STATES:
        if state in ROOF_MOVING_STATES:
            continue
        for event in EVENTS:
            out = step(state, event, snap)
            assert not (out.kind == "transition" and out.state in ROOF_MOVING_STATES), \
                f"roof-unknown reached {out.state} via {state}+{event}"


# ------------------------------------------------------------ liveness facts

def _successors(state, snaps):
    out = set()
    for event in EVENTS:
        for snap in snaps:
            r = step(state, event, snap)
            if r.kind == "transition":
                out.add(r.state)
    return out


def test_every_state_reachable_from_idle_day():
    snaps = (ALL_GO, ALL_STOP, ALL_GO.replace(slots_remaining=0))
    seen, frontier = {INITIAL_STATE}, [INITIAL_STATE]
    while frontier:
        nxt = []
        for s in frontier:
            for d in _successors(s, snaps):
                if d not in seen:
                    seen.add(d)
                    nxt.append(d)
        frontier = nxt
    assert seen == set(STATES), f"unreachable states: {set(STATES) - seen}"


def test_every_state_has_a_path_back_to_idle_day():
    """No black holes. The holds get back via operator events, which is the
    designed shape; what this forbids is a state nothing can ever leave."""
    snaps = (ALL_GO, ALL_STOP, ALL_GO.replace(slots_remaining=0))
    # reverse reachability: which states can reach IDLE_DAY?
    can_reach = {"IDLE_DAY"}
    changed = True
    while changed:
        changed = False
        for s in STATES:
            if s in can_reach:
                continue
            if _successors(s, snaps) & can_reach:
                can_reach.add(s)
                changed = True
    assert can_reach == set(STATES), f"black holes: {set(STATES) - can_reach}"


# ---------------------------------------------------------------- golden night

def test_a_full_two_slot_night_walks_the_table():
    """The canonical night, two slots, as a plain sequence of events. This is
    the story the whole machine exists to tell; if a table edit changes it,
    that edit should have to look this test in the eye."""
    s = INITIAL_STATE
    go2 = ALL_GO                              # 2 slots remaining
    go1 = ALL_GO.replace(slots_remaining=1)
    go0 = ALL_GO.replace(slots_remaining=0)
    script = [
        ("NOON_TICK", go2, "PLANNING"),
        ("PLAN_GOOD", go2, "ARMED"),
        ("PRE_SUNSET_TICK", go2, "PRE_FLIGHT"),
        ("CHECKS_PASSED", go2, "OPENING_ROOF"),
        ("ROOF_OPEN_CONFIRMED", go2, "PRELUDE"),
        ("NINA_PRELUDE_DONE", go2, "SLOT_SETUP"),
        ("SLOT_STARTED", go2, "SLOT_IMAGING"),        # slot 1 running
        ("NINA_SLOT_DONE", go1, "SLOT_SETUP"),        # slot 1 done, one left
        ("SLOT_STARTED", go1, "SLOT_IMAGING"),        # slot 2 running
        ("NINA_SLOT_DONE", go0, "FLATS"),             # plan exhausted
        ("NINA_FLATS_DONE", go0, "CLOSING_ROOF"),
        ("ROOF_CLOSE_CONFIRMED", go0, "SHUTDOWN"),
        ("SHUTDOWN_DONE", go0, "NIGHT_DONE"),
        ("DAY_TICK", go0, "IDLE_DAY"),
    ]
    for event, snap, expected in script:
        out = step(s, event, snap)
        assert out.kind == "transition", (s, event, out)
        assert out.state == expected, (s, event, out.state, expected)
        s = out.state


def test_stall_night_lands_in_persistent_fault():
    s = "PRE_FLIGHT"
    out = step(s, "CHECKS_PASSED", ALL_GO)
    assert out.state == "OPENING_ROOF"
    out = step(out.state, "ROOF_STALL", ALL_STOP)
    assert out.state == "FAULT_ROOF_UNKNOWN"
    # and nothing but the operator gets it out (covered exhaustively above,
    # restated here in story form)
    assert step("FAULT_ROOF_UNKNOWN", "NOON_TICK", ALL_GO).kind == "ignored"
    assert step("FAULT_ROOF_UNKNOWN", "OPERATOR_RESOLVE", ALL_GO).state == "IDLE_DAY"
