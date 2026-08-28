"""The safety guards — pure functions of a SensorSnapshot.

Every guard returns None to pass, or a human-readable reason string to refuse.
The reason string is what the chat renders to a user and what the journal
records for a rejected event, so it is written for a person standing in front
of a refused roof, not for a stack trace.

These functions subsume the six scattered enforcement sites catalogued in
docs/ARCHITECTURE_PLAN.md. Invariant A ("the roof moves only when the scope is
confirmed parked") and Invariant B ("the mount moves only when the roof is
confirmed open") live HERE and nowhere else; the property tests in
tests/test_guards.py enumerate the entire snapshot space and assert them.
"""
from iris.core.snapshot import SensorSnapshot, Tri


def mount_parked(s: SensorSnapshot):
    """Invariant A's evidence rule: BOTH park sensors must positively confirm.

    Both, not either: the two fail differently. PWI4 loses its home reference
    on a power cycle and reports not-parked on a genuinely parked scope
    (fail-safe); the visual marker can be occluded or the camera dark
    (fail-safe as UNKNOWN). Requiring both keeps every failure on the refusing
    side. A softer merge is a policy change to make deliberately, here, with
    the property tests updated in the same commit.
    """
    if s.parked_vision is not Tri.CONFIRMED:
        return ("scope not confirmed parked by vision (%s)"
                % s.parked_vision.value.lower())
    if s.parked_pwi4 is not Tri.CONFIRMED:
        return ("scope not confirmed parked by the mount (%s) — a power cycle "
                "loses PWI4's home reference; re-home and park"
                % s.parked_pwi4.value.lower())
    return None


def roof_state_known(s: SensorSnapshot):
    """The relay is a toggle: firing it with the roof position unknown turns
    an unknown into a coin flip. Any confirmed position passes."""
    if s.roof is Tri.UNKNOWN:
        return "roof position unknown — resolve before any roof motion"
    return None


def roof_open(s: SensorSnapshot):
    """Invariant B: mount motion only under a confirmed-open roof."""
    if s.roof is not Tri.CONFIRMED:
        return ("roof not confirmed open (%s) — mount stays put"
                % s.roof.value.lower())
    return None


def roof_closed(s: SensorSnapshot):
    if s.roof is not Tri.DENIED:
        return ("roof not confirmed closed (%s)" % s.roof.value.lower())
    return None


def safety_armed(s: SensorSnapshot):
    """The operator's standing permission. Cleared or never-set both refuse —
    and note there is deliberately NO guard that re-arms this; only an
    operator event does (see SAFE_HOLD in the machine)."""
    if not s.safety_armed:
        return "observatory not marked safe — issue safe! first"
    return None


def auto_mode(s: SensorSnapshot):
    if not s.mode_auto:
        return "scheduler is in manual mode"
    return None


def weather_ok(s: SensorSnapshot):
    if not s.weather_ok:
        return "weather is not acceptable"
    return None


def slots_remaining(s: SensorSnapshot):
    if s.slots_remaining <= 0:
        return "night plan exhausted"
    return None


def plan_exhausted(s: SensorSnapshot):
    if s.slots_remaining > 0:
        return "%d slot(s) still planned" % s.slots_remaining
    return None


def evaluate(guards, s: SensorSnapshot):
    """First refusal wins; None means all passed. Order is meaningful and the
    tables list the most fundamental guard first, so the reason a user sees is
    the most fundamental problem, not an incidental one."""
    for g in guards:
        reason = g(s)
        if reason is not None:
            return "%s: %s" % (g.__name__, reason)
    return None
