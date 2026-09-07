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


def parked_by_cameras(s: SensorSnapshot) -> Tri:
    """The park reading the cameras support, as one Tri.

    Since 2026-09-07 there is ONE park camera: the inside Kasa camera reading
    the scope's AprilTag against the recorded park pose, pose-verified,
    several frames per read. parked_vision is that gating read (the
    "vision parked=" line every roof decision writes). parked_kasa is the
    same camera's raw line, which every read of it writes, verified or not;
    it cannot confirm on its own but a DENIED from it -- an off-park scope
    decoded in any frame -- vetoes. Before that date the two fields were two
    cameras that had to agree; the scope-top webcam that supplied the first
    was retired with the 2026-09-06 bump that cost a night.
    """
    if s.parked_kasa is Tri.DENIED:
        return Tri.DENIED
    return s.parked_vision


def mount_parked(s: SensorSnapshot):
    """Invariant A's evidence rule: the park camera confirms, and neither the
    camera's raw line nor PWI4 contradicts it.

    The mount is UNPOWERED at every roof open. start.py switches it on during
    the prelude, which runs AFTER the roof is already open, so PWI4 is
    structurally UNKNOWN at the exact moment the most dangerous move is
    decided. This is not a rare fault to design around; it is every single
    night. Measured on 2026-09-04: both roof fires carried
    parked_pwi4=UNKNOWN while the cameras read the scope parked, and the
    earlier rule ("BOTH park sensors must positively confirm", where the
    second sensor was PWI4) would have refused the open AND the close. In a
    live machine that is a roof left open at dawn.

    So PWI4 is a VETO, not a requirement. DENIED refuses, because a POWERED
    mount saying "I am not at park" is real information that outranks a
    template match. UNKNOWN abstains, because an unpowered mount has no
    opinion and pretending otherwise is what broke the rule.

    The positive evidence is the pose-verified, multi-frame AprilTag read
    (parked_vision). The independent second confirmation the retired webcam
    used to give is gone; what replaces it is inside the read (the camera is
    driven to its reference pose before it looks, and OPEN/parked need the
    frames to agree) and the two vetoes here. A fixed fiducial tag on the
    wall would let the camera prove its own pose without the cloud; until
    then a pose that cannot be verified reads UNKNOWN and refuses.

    Every failure still lands on the refusing side: a dark camera, a stale
    reading, an unverified pose, a raw-line UNSAFE and a mount veto all
    refuse.
    """
    if s.parked_kasa is Tri.DENIED:
        return ("the park camera read the scope OFF park in at least one "
                "frame — nothing moves until it reads parked")
    if s.parked_vision is not Tri.CONFIRMED:
        return ("scope not confirmed parked by the park camera (reads %s) — "
                "it must positively see the scope tag at the park pose"
                % s.parked_vision.value.lower())
    if s.parked_pwi4 is Tri.DENIED:
        return ("the mount reports it is NOT parked, contradicting the "
                "camera — re-home and park, or resolve the disagreement, "
                "before any roof motion")
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
