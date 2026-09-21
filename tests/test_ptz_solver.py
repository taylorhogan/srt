"""The pan/tilt step solver: can the camera always be driven home?

2026-09-21: Iris North sat at y=-123 with its home at y=-9. The old solver
searched products of up to 5 step sizes and found nothing for +114, so
ensure_at returned None, the camera could not be driven to its reference pose,
and every reading from it had to refuse -- a safety sensor disabled by
arithmetic rather than by anything physical.

The step sizes share no common factor, so every delta IS reachable; the
search just has to be willing to look past five moves.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ptz = pytest.importorskip("scripts.kasa_ptz")


def total(plan):
    return sum(ptz.STEP_UNITS[speed] * sign for speed, sign in plan)


def test_the_delta_that_stranded_iris_north():
    plan = ptz.solve(114)
    assert plan is not None and total(plan) == 114
    assert len(plan) == 6        # why the old 5-deep search missed it


@pytest.mark.parametrize("delta", [0, 1, -1, 4, -7, 50, -50, 114, 378, 999, -1234])
def test_every_delta_is_reachable_and_exact(delta):
    plan = ptz.solve(delta)
    assert plan is not None, delta
    assert total(plan) == delta
    assert all(speed in ptz.STEP_UNITS for speed, _ in plan)
    assert all(sign in (1, -1) for _, sign in plan)


def test_zero_asks_for_no_movement():
    assert ptz.solve(0) == []


def test_shortest_plan_is_returned():
    assert len(ptz.solve(50)) == 1
    assert len(ptz.solve(333)) == 1
    assert len(ptz.solve(100)) == 2
    assert len(ptz.solve(114)) <= len(ptz.solve(114, max_moves=12))


def test_plans_stay_inside_the_axis():
    """A plan is no use if it drives into an end stop half way and lands wrong."""
    span = 387
    origin = -123
    plan = ptz.solve(114, origin=origin, span=span)
    assert plan is not None and total(plan) == 114
    at = origin
    for speed, sign in plan:
        at += ptz.STEP_UNITS[speed] * sign
        assert abs(at) <= span, (plan, at)


def test_unreachable_within_a_tight_axis_returns_none():
    # Hard against the stop with no room to manoeuvre in either direction.
    assert ptz.solve(7, origin=0, span=10) is None
