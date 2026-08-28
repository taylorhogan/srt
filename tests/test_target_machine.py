"""Exhaustive coverage of the Target machine (the DSO lifecycle)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.core.target_machine import (TARGET_EVENTS, TARGET_INITIAL_STATE,
                                      TARGET_STATES, TARGET_TRANSITIONS,
                                      target_step)


def test_table_references_only_declared_states_and_events():
    for row in TARGET_TRANSITIONS:
        assert row.src in TARGET_STATES, row
        assert row.dst in TARGET_STATES, row
        assert row.event in TARGET_EVENTS, row


def test_every_pair_deterministic_and_closed():
    for state in TARGET_STATES:
        for event in TARGET_EVENTS:
            a = target_step(state, event)
            assert a == target_step(state, event)
            assert a.kind in ("transition", "ignored")
            assert a.state in TARGET_STATES


def test_no_duplicate_rows():
    seen = set()
    for row in TARGET_TRANSITIONS:
        key = (row.src, row.event)
        assert key not in seen, key
        seen.add(key)


def test_auto_stop_convergence_leaves_acquiring():
    """The owner's auto-stop: convergence moves the target OUT of the pool the
    planner draws from. The planner's rule is 'schedule only ACQUIRING/QUEUED',
    so this transition IS the stop."""
    assert target_step("ACQUIRING", "CONVERGENCE_MET").state == "CONVERGED"
    # and more frames on a converged target do NOT reopen it silently
    assert target_step("CONVERGED", "FRAMES_ADDED").kind == "ignored"


def test_auto_publish_path_is_complete():
    """WISHED -> PUBLISHED with no operator event anywhere on the path."""
    operator_events = {"REOPENED", "RETIRE"}
    s = TARGET_INITIAL_STATE
    for event in ("RESOLVED", "FRAMES_ADDED", "CONVERGENCE_MET",
                  "RENDER_STARTED", "RENDER_DONE", "PUBLISH_DONE"):
        assert event not in operator_events
        out = target_step(s, event)
        assert out.kind == "transition", (s, event)
        s = out.state
    assert s == "PUBLISHED"


def test_render_failure_returns_to_converged_for_retry():
    assert target_step("RENDERING", "RENDER_FAILED").state == "CONVERGED"


def test_published_and_retired_are_reopenable():
    assert target_step("PUBLISHED", "REOPENED").state == "ACQUIRING"
    assert target_step("RETIRED", "REOPENED").state == "ACQUIRING"


def test_every_state_reachable_and_none_terminal_forever():
    # forward reachability from WISHED
    seen, frontier = {TARGET_INITIAL_STATE}, [TARGET_INITIAL_STATE]
    while frontier:
        nxt = []
        for s in frontier:
            for e in TARGET_EVENTS:
                r = target_step(s, e)
                if r.kind == "transition" and r.state not in seen:
                    seen.add(r.state)
                    nxt.append(r.state)
        frontier = nxt
    assert seen == set(TARGET_STATES), set(TARGET_STATES) - seen
    # nothing is a black hole: every state has at least one exit
    for s in TARGET_STATES:
        exits = {target_step(s, e).state for e in TARGET_EVENTS
                 if target_step(s, e).kind == "transition"} - {s}
        assert exits, f"{s} has no exit"
