"""SensorSnapshot — the world as the guards are allowed to see it.

Guards are pure functions of exactly this object. That is the whole testing
strategy for the safety invariants: because a guard can consult nothing else,
enumerating this object's small value space IS enumerating every situation a
guard can ever face, and "no snapshot with parked != CONFIRMED permits roof
motion" becomes a checkable statement rather than a hope.

Fields use three-valued readings (CONFIRMED / DENIED / UNKNOWN) wherever a
sensor can fail to answer, because the failure to answer is the safety-relevant
case: the roof relay is a toggle, so acting on UNKNOWN is how telescopes get
crushed. Collapsing UNKNOWN into False would hide the distinction the guards
exist to enforce.
"""
from dataclasses import dataclass, field
from enum import Enum


class Tri(str, Enum):
    """A sensor reading that can decline to answer."""
    CONFIRMED = "CONFIRMED"
    DENIED = "DENIED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SensorSnapshot:
    """Everything a guard may consult, at one instant.

    parked_vision / parked_kasa / parked_pwi4: the three independent park
        sensors. parked_vision is the scope-top webcam's template ladder;
        parked_kasa is the indoor camera's AprilTag comparison against the
        recorded park pose; parked_pwi4 is the mount's own opinion. They can
        legitimately disagree — a mount power cycle destroys PWI4's home
        reference while both markers still sit at park — which is why all
        three are carried rather than pre-merged: the MERGE POLICY is a
        guard's decision, visible in guards.py, not buried in a collector.

        Each is genuinely three-valued. A camera that can see the scope and
        judges it off park says DENIED; a camera that cannot see (dark frame,
        occlusion, stale reading) says UNKNOWN. Collapsing those two into one
        "false" is what made the old bool unable to tell a broken camera from
        a moved scope.
    roof: CONFIRMED means confirmed OPEN; DENIED means confirmed CLOSED;
        UNKNOWN means exactly that (mid-travel, post-stall, camera down).
    safety_armed: the operator's standing permission (safety.txt today,
        journal context later). False is both "cleared" and "never set".
    mode_auto: scheduler may act unattended.
    weather_ok: the planner's current weather verdict.
    slots_remaining: number of night-plan slots not yet run (0 = plan done).
    nina_alive: the capture process exists (liveness probe, never authority).

    roof_north / roof_cam / roof_unmoved (2026-10-10, Phase 2b): the roof's
        sensors kept apart, as the park sensors are, so the merge policy can
        live in guards.py (roof_by_cameras) instead of inside the vision code.
        roof_north is the north camera's tag (id 2) against its OPEN / SHUT
        references; roof_cam is Iris cam (tag 1 at shut, the gold star for
        open); same meaning as `roof` (CONFIRMED open, DENIED closed).
        roof_unmoved is the blind-park evidence (sentry/roof_evidence):
        CONFIRMED = a gating read saw the roof OPEN within 16 h and nothing
        has powered or fired the roof since; DENIED = something could have
        moved it since, or that open is older than 16 h; UNKNOWN = no record.
        Observed, not yet consulted by any table guard: until they are, they
        stay out of enumerate_snapshots (multiplying the space by 27 for
        fields no guard reads would only slow the sweeps); they join it on the
        day a guard reads them.
    """
    parked_vision: Tri = Tri.UNKNOWN
    parked_kasa: Tri = Tri.UNKNOWN
    parked_pwi4: Tri = Tri.UNKNOWN
    roof: Tri = Tri.UNKNOWN
    safety_armed: bool = False
    mode_auto: bool = False
    weather_ok: bool = False
    slots_remaining: int = 0
    nina_alive: bool = False
    roof_north: Tri = Tri.UNKNOWN
    roof_cam: Tri = Tri.UNKNOWN
    roof_unmoved: Tri = Tri.UNKNOWN

    def replace(self, **kw) -> "SensorSnapshot":
        from dataclasses import replace as _replace
        return _replace(self, **kw)


BLIND_PARK_OPEN_MAX_AGE_H = 16.0     # the same bound stop!'s blind_park_refusal uses


def roof_word(word) -> Tri:
    """'open' / 'shut' / anything else -> the roof Tri. Pure."""
    return {"open": Tri.CONFIRMED, "shut": Tri.DENIED, "closed": Tri.DENIED}.get(
        str(word or "").lower(), Tri.UNKNOWN)


def unmoved_since_open(open_confirmed, motion_possible, now,
                       max_age_h: float = BLIND_PARK_OPEN_MAX_AGE_H) -> Tri:
    """roof_unmoved from sentry/roof_evidence's two timestamps. Pure.

    CONFIRMED: an OPEN was confirmed no more than *max_age_h* ago and nothing
    that could move the roof came after it (the open's own fire comes before
    its confirmation, so it does not count). DENIED: motion after the open,
    or the open too old. UNKNOWN: no open on record.
    """
    if open_confirmed is None:
        return Tri.UNKNOWN
    if (now - open_confirmed).total_seconds() > max_age_h * 3600.0:
        return Tri.DENIED
    if motion_possible is not None and motion_possible > open_confirmed:
        return Tri.DENIED
    return Tri.CONFIRMED


# The full enumerable space, for property tests. Kept beside the dataclass so
# adding a field without extending the space is a visible diff in one place.
TRI_VALUES = (Tri.CONFIRMED, Tri.DENIED, Tri.UNKNOWN)
BOOL_VALUES = (False, True)
SLOT_VALUES = (0, 1, 3)          # zero / last / several — the behavioural classes


def enumerate_snapshots():
    """Yield every behaviourally distinct SensorSnapshot (3,888)."""
    for pv in TRI_VALUES:
        for pk in TRI_VALUES:
            for pp in TRI_VALUES:
                for roof in TRI_VALUES:
                    for safe in BOOL_VALUES:
                        for auto in BOOL_VALUES:
                            for wx in BOOL_VALUES:
                                for slots in SLOT_VALUES:
                                    for nina in BOOL_VALUES:
                                        yield SensorSnapshot(
                                            parked_vision=pv, parked_kasa=pk,
                                            parked_pwi4=pp,
                                            roof=roof, safety_armed=safe,
                                            mode_auto=auto, weather_ok=wx,
                                            slots_remaining=slots,
                                            nina_alive=nina)
