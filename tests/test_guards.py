"""The safety invariants, asserted over the ENTIRE snapshot space.

enumerate_snapshots() yields every behaviourally distinct world (~1,944).
These tests sweep all of them, so the invariants hold not for chosen examples
but for every situation a guard can ever face. That is the strongest claim the
no-TLA+ decision allows, and it is strong enough: a guard consults nothing but
the snapshot, so this sweep IS the model check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.core import guards as G
from iris.core.snapshot import SensorSnapshot, Tri, enumerate_snapshots


def test_space_is_the_size_the_docstring_claims():
    n = sum(1 for _ in enumerate_snapshots())
    assert n == 3 * 3 * 3 * 2 * 2 * 2 * 3 * 2, n


def test_invariant_a_mount_parked_requires_both_confirmations():
    """No snapshot passes mount_parked unless BOTH park sensors say CONFIRMED.
    UNKNOWN counts as refusal — a sensor declining to answer is exactly when
    the roof must not move."""
    for s in enumerate_snapshots():
        passed = G.mount_parked(s) is None
        should = (s.parked_vision is Tri.CONFIRMED
                  and s.parked_pwi4 is Tri.CONFIRMED)
        assert passed == should, s


def test_invariant_b_roof_open_requires_positive_confirmation():
    for s in enumerate_snapshots():
        passed = G.roof_open(s) is None
        assert passed == (s.roof is Tri.CONFIRMED), s


def test_roof_state_known_refuses_only_unknown():
    for s in enumerate_snapshots():
        passed = G.roof_state_known(s) is None
        assert passed == (s.roof is not Tri.UNKNOWN), s


def test_guards_are_pure_and_stateless():
    """Same snapshot twice -> same answer; and evaluating one guard must not
    change another's answer (no hidden coupling)."""
    s = SensorSnapshot(parked_vision=Tri.CONFIRMED)
    for g in (G.mount_parked, G.roof_open, G.roof_closed, G.safety_armed,
              G.auto_mode, G.weather_ok, G.slots_remaining, G.plan_exhausted,
              G.roof_state_known):
        assert g(s) == g(s), g.__name__
    before = G.mount_parked(s)
    G.roof_open(s); G.safety_armed(s)
    assert G.mount_parked(s) == before


def test_every_refusal_names_its_guard_and_reads_as_a_sentence():
    """The reason string is user-facing: evaluate() prefixes the guard name,
    and the text must be non-empty prose, not a code."""
    s = SensorSnapshot()          # everything refuses on the empty world
    for g in (G.mount_parked, G.roof_open, G.roof_closed, G.safety_armed,
              G.auto_mode, G.weather_ok, G.slots_remaining, G.roof_state_known):
        reason = G.evaluate([g], s)
        assert reason and reason.startswith(g.__name__ + ": "), (g.__name__, reason)
        assert len(reason.split(":", 1)[1].strip()) > 10, reason


def test_evaluate_returns_first_refusal_in_order():
    s = SensorSnapshot()          # both would refuse
    reason = G.evaluate([G.safety_armed, G.mount_parked], s)
    assert reason.startswith("safety_armed:")
    reason = G.evaluate([G.mount_parked, G.safety_armed], s)
    assert reason.startswith("mount_parked:")


def test_slots_guards_partition_the_space():
    """slots_remaining and plan_exhausted are complements: exactly one passes
    for every snapshot — this is what makes the NINA_SLOT_DONE fan-out rows
    total (one of the two always fires)."""
    for s in enumerate_snapshots():
        a = G.slots_remaining(s) is None
        b = G.plan_exhausted(s) is None
        assert a != b, s
