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

    parked_vision / parked_pwi4: the two independent park sensors. They can
        legitimately disagree — a mount power cycle destroys PWI4's home
        reference while the visual marker still sits at park — which is why
        both are carried rather than pre-merged: the MERGE POLICY is a guard's
        decision, visible in guards.py, not buried in a collector.
    roof: CONFIRMED means confirmed OPEN; DENIED means confirmed CLOSED;
        UNKNOWN means exactly that (mid-travel, post-stall, camera down).
    safety_armed: the operator's standing permission (safety.txt today,
        journal context later). False is both "cleared" and "never set".
    mode_auto: scheduler may act unattended.
    weather_ok: the planner's current weather verdict.
    slots_remaining: number of night-plan slots not yet run (0 = plan done).
    nina_alive: the capture process exists (liveness probe, never authority).
    """
    parked_vision: Tri = Tri.UNKNOWN
    parked_pwi4: Tri = Tri.UNKNOWN
    roof: Tri = Tri.UNKNOWN
    safety_armed: bool = False
    mode_auto: bool = False
    weather_ok: bool = False
    slots_remaining: int = 0
    nina_alive: bool = False

    def replace(self, **kw) -> "SensorSnapshot":
        from dataclasses import replace as _replace
        return _replace(self, **kw)


# The full enumerable space, for property tests. Kept beside the dataclass so
# adding a field without extending the space is a visible diff in one place.
TRI_VALUES = (Tri.CONFIRMED, Tri.DENIED, Tri.UNKNOWN)
BOOL_VALUES = (False, True)
SLOT_VALUES = (0, 1, 3)          # zero / last / several — the behavioural classes


def enumerate_snapshots():
    """Yield every behaviourally distinct SensorSnapshot (~1,944)."""
    for pv in TRI_VALUES:
        for pp in TRI_VALUES:
            for roof in TRI_VALUES:
                for safe in BOOL_VALUES:
                    for auto in BOOL_VALUES:
                        for wx in BOOL_VALUES:
                            for slots in SLOT_VALUES:
                                for nina in BOOL_VALUES:
                                    yield SensorSnapshot(
                                        parked_vision=pv, parked_pwi4=pp,
                                        roof=roof, safety_armed=safe,
                                        mode_auto=auto, weather_ok=wx,
                                        slots_remaining=slots,
                                        nina_alive=nina)
