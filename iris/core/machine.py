"""The Night machine — the whole machine IS the table below.

Nothing else in the system may change the night's state: actuators run as
transition ACTIONS (from Phase 2 on), consoles render the journal, and every
offered event either transitions, is rejected by a guard (journaled with the
reason), or is ignored (no row). There is no other code path.

Design rules, enforced by tests/test_machine.py:
  * ESTOP_REQUESTED is accepted from every state (wildcard row).
  * SAFE_HOLD is left ONLY by an operator event — this is how the old
    restart-re-arms-safety bug is impossible rather than merely fixed.
  * FAULT_ROOF_UNKNOWN and ESTOP are left ONLY via OPERATOR_RESOLVE.
  * Every state is reachable from IDLE_DAY, and every state has a path back
    (no black holes except through an operator, which is the point of one).
  * Any transition INTO a roof-moving state carries mount_parked and
    roof_state_known; any transition INTO a mount-moving state carries
    roof_open. The guards, not the callers, are the safety system.

The interpreter is ~60 lines because the machine is data. Rows are matched in
table order; the first row whose guards all pass wins, which is how one event
can fan out (NINA_SLOT_DONE -> next slot or flats) without a special case.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from iris.core import guards as G
from iris.core.snapshot import SensorSnapshot

# ---------------------------------------------------------------- states

STATES = (
    "IDLE_DAY",          # nothing planned; roof closed, mount parked
    "PLANNING",          # noon check running
    "ARMED",             # night plan exists — survives restart
    "PRE_FLIGHT",        # pre-sunset re-check
    "OPENING_ROOF",      # relay fired, awaiting confirmed open
    "PRELUDE",           # once per night: cooling, initial focus
    "SLOT_SETUP",        # slew / sequence launch for the current slot
    "SLOT_IMAGING",      # main sequence for the current slot
    "FLATS",             # once per night
    "CLOSING_ROOF",      # park confirmed, relay fired, awaiting confirmed closed
    "SHUTDOWN",          # dehumidifier, summary, handoff marker
    "NIGHT_DONE",        # terminal for the night
    "SAFE_HOLD",         # operator cleared safety; only an operator exits
    "FAULT_ROOF_UNKNOWN",  # roof position untrusted; only OPERATOR_RESOLVE exits
    "ESTOP",             # emergency stop ran; only OPERATOR_RESOLVE exits
)

# ---------------------------------------------------------------- events

EVENTS = (
    # clock / planner
    "NOON_TICK", "PLAN_GOOD", "PLAN_BAD", "PRE_SUNSET_TICK",
    "CHECKS_PASSED", "CHECKS_FAILED", "REPLAN_REQUESTED",
    # roof sensing / motion outcomes
    "ROOF_OPEN_CONFIRMED", "ROOF_CLOSE_CONFIRMED", "ROOF_STALL",
    "ROOF_TIMEOUT", "VISION_CONTRADICTION",
    # capture (NINA today) cooperative signals.
    # SLOT_STARTED and NINA_SLOT_DONE are deliberately DISTINCT events. An
    # early draft used NINA_SLOT_DONE for both "sequence launched" (in
    # SLOT_SETUP) and "sequence finished" (in SLOT_IMAGING), disambiguated
    # only by state -- and the historical replay caught it: on the real
    # 2026-05-18 night NINA restarted mid-run, the state desynced by one, the
    # two meanings crossed, and the machine ended the night stranded in
    # SLOT_IMAGING. One name, one meaning.
    "NINA_PRELUDE_DONE", "SLOT_STARTED", "NINA_SLOT_DONE", "SLOT_WINDOW_END",
    "NINA_FLATS_DONE", "CAPTURE_LOST",
    # night lifecycle
    "NIGHT_END_REQUESTED", "SHUTDOWN_DONE", "DAY_TICK",
    # weather
    "WEATHER_BAD",
    # operator
    "SAFETY_CLEARED", "SAFETY_ARMED", "ESTOP_REQUESTED", "OPERATOR_RESOLVE",
)

# ---------------------------------------------------------------- table


@dataclass(frozen=True)
class T:
    """One row: in state `src`, event `event` moves to `dst` if guards pass."""
    src: str            # a state, or "*" for any non-hold state
    event: str
    dst: str
    guards: Sequence[Callable] = field(default_factory=tuple)


# States a wildcard row does NOT apply to: the holds are left only by their
# own explicit operator rows, and a wildcard that could yank the machine out
# of SAFE_HOLD would reintroduce the bug SAFE_HOLD exists to kill.
HOLD_STATES = frozenset({"SAFE_HOLD", "FAULT_ROOF_UNKNOWN", "ESTOP"})

TRANSITIONS = (
    # --- the day
    T("IDLE_DAY",     "NOON_TICK",           "PLANNING"),
    T("PLANNING",     "PLAN_GOOD",           "ARMED"),
    T("PLANNING",     "PLAN_BAD",            "IDLE_DAY"),
    T("ARMED",        "PRE_SUNSET_TICK",     "PRE_FLIGHT"),
    T("ARMED",        "WEATHER_BAD",         "IDLE_DAY"),

    # --- arming and opening. The whole of Invariant A rides on this row.
    T("PRE_FLIGHT",   "CHECKS_PASSED",       "OPENING_ROOF",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known, G.weather_ok)),
    T("PRE_FLIGHT",   "CHECKS_FAILED",       "IDLE_DAY"),
    T("PRE_FLIGHT",   "WEATHER_BAD",         "IDLE_DAY"),
    T("OPENING_ROOF", "ROOF_OPEN_CONFIRMED", "PRELUDE"),
    T("OPENING_ROOF", "ROOF_STALL",          "FAULT_ROOF_UNKNOWN"),
    T("OPENING_ROOF", "ROOF_TIMEOUT",        "FAULT_ROOF_UNKNOWN"),

    # --- the slots. One event, two rows: table order + guards do the fan-out.
    T("PRELUDE",      "NINA_PRELUDE_DONE",   "SLOT_SETUP",
      guards=(G.slots_remaining,)),
    T("PRELUDE",      "NINA_PRELUDE_DONE",   "FLATS",
      guards=(G.plan_exhausted,)),
    T("SLOT_SETUP",   "SLOT_STARTED",        "SLOT_IMAGING"),
    T("SLOT_IMAGING", "NINA_SLOT_DONE",      "SLOT_SETUP",
      guards=(G.slots_remaining,)),
    T("SLOT_IMAGING", "NINA_SLOT_DONE",      "FLATS",
      guards=(G.plan_exhausted,)),
    T("SLOT_IMAGING", "SLOT_WINDOW_END",     "SLOT_SETUP",
      guards=(G.slots_remaining,)),
    T("SLOT_IMAGING", "SLOT_WINDOW_END",     "FLATS",
      guards=(G.plan_exhausted,)),
    T("SLOT_IMAGING", "REPLAN_REQUESTED",    "SLOT_SETUP"),
    T("SLOT_IMAGING", "CAPTURE_LOST",        "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("SLOT_IMAGING", "WEATHER_BAD",         "FLATS"),

    # --- closing out the night. Invariant A again, on the way down.
    T("FLATS",        "NINA_FLATS_DONE",     "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("FLATS",        "CAPTURE_LOST",        "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("CLOSING_ROOF", "ROOF_CLOSE_CONFIRMED", "SHUTDOWN"),
    T("CLOSING_ROOF", "ROOF_STALL",          "FAULT_ROOF_UNKNOWN"),
    T("CLOSING_ROOF", "ROOF_TIMEOUT",        "FAULT_ROOF_UNKNOWN"),
    T("SHUTDOWN",     "SHUTDOWN_DONE",       "NIGHT_DONE"),
    T("NIGHT_DONE",   "DAY_TICK",            "IDLE_DAY"),

    # --- an early end request from any active night state
    T("PRELUDE",      "NIGHT_END_REQUESTED", "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("SLOT_SETUP",   "NIGHT_END_REQUESTED", "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("SLOT_IMAGING", "NIGHT_END_REQUESTED", "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("FLATS",        "NIGHT_END_REQUESTED", "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),

    # --- contradiction between roof sensors, noticed at rest
    T("PRELUDE",      "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("SLOT_SETUP",   "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("SLOT_IMAGING", "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("FLATS",        "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),

    # --- operator: the holds
    T("*",            "SAFETY_CLEARED",      "SAFE_HOLD"),
    T("SAFE_HOLD",    "SAFETY_ARMED",        "IDLE_DAY"),
    T("*",            "ESTOP_REQUESTED",     "ESTOP"),
    T("FAULT_ROOF_UNKNOWN", "OPERATOR_RESOLVE", "IDLE_DAY"),
    T("ESTOP",        "OPERATOR_RESOLVE",    "IDLE_DAY"),
)

# ---------------------------------------------------------------- interpreter


@dataclass(frozen=True)
class Outcome:
    """What offering one event to the machine produced."""
    kind: str                     # "transition" | "rejected" | "ignored"
    state: str                    # the (possibly new) current state
    guard: Optional[str] = None   # refusal reason, for "rejected"


def step(state: str, event: str, snapshot: SensorSnapshot,
         table: Sequence[T] = TRANSITIONS) -> Outcome:
    """Offer one event. Pure: no I/O, no clock, no globals.

    Matching rows are tried in table order; the first whose guards all pass
    fires. If rows matched but every one was guarded off, the event is
    REJECTED with the first row's refusal (the most specific complaint). If no
    row matched at all, the event is IGNORED — which is not an error: sensors
    report unconditionally and most reports are irrelevant to most states.
    """
    first_refusal = None
    for row in table:
        if row.event != event:
            continue
        if row.src == "*":
            if state in HOLD_STATES:
                continue
            if state == row.dst:
                continue          # already there; re-entering is noise
        elif row.src != state:
            continue
        refusal = G.evaluate(row.guards, snapshot)
        if refusal is None:
            return Outcome("transition", row.dst)
        if first_refusal is None:
            first_refusal = refusal
    if first_refusal is not None:
        return Outcome("rejected", state, guard=first_refusal)
    return Outcome("ignored", state)


INITIAL_STATE = "IDLE_DAY"
