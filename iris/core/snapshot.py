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

    def replace(self, **kw) -> "SensorSnapshot":
        from dataclasses import replace as _replace
        return _replace(self, **kw)


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
