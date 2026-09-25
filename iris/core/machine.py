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
  * The mount does not change the night's state, so a mount request is a
    PERMISSION row (dst == src, Outcome "allowed"), and every
    MOUNT_MOVE_REQUESTED row carries roof_open -- Invariant B, decided here
    and nowhere else since 2026-09-25. No hold has such a row: after a
    stop! nothing moves or powers the mount until resolve!.

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
    "PARKING",           # mount parking before any roof close. Exists because
                         #   reality parks the scope AS PART OF closing (end.py
                         #   does exactly this); an entry guard of "already
                         #   parked" on CLOSING_ROOF contradicted every
                         #   mid-night close, where the scope is tracking when
                         #   the close is decided. Invariant A's guard sits on
                         #   the PARKING -> CLOSING_ROOF edge: between the park
                         #   CONFIRMATION and the relay fire.
    "CLOSING_ROOF",      # park confirmed, relay fired, awaiting confirmed closed
    "FLATS",             # once per night, AFTER the roof is shut. This order
                         #   is not a preference, it is what end.py does and
                         #   what the 2026-09-04 night did: relay fired closed
                         #   at 02:26:38, flats ran 02:29:46 to 04:02:47
                         #   against a panel. The machine used to place FLATS
                         #   before PARKING, which no real night could match,
                         #   and that inversion is also what made the close
                         #   cascade fire twice -- imaging.txt legitimately
                         #   passes through NONE between the main sequence and
                         #   the flats, and the old mapping read that NONE as
                         #   the end of the night.
    "SHUTDOWN",          # dehumidifier, summary, handoff marker
    "NIGHT_DONE",        # terminal for the night
    "SAFE_HOLD",         # operator cleared safety; only an operator exits
    "FAULT_ROOF_UNKNOWN",  # roof position untrusted; only OPERATOR_RESOLVE exits
    "ESTOP",             # emergency stop ran; only OPERATOR_RESOLVE exits
    # --- roof moved by hand, outside a night (Phase 2). roof!! open/close and
    # scripts/cycle_roof.py ask the conductor for the roof; these are where
    # the machine holds that motion so it is guarded, journaled and timed out
    # exactly as the night's own roof moves are, without pretending a night
    # has begun. Not stages of a night: excluded from NIGHT_PATH.
    "MANUAL_OPENING",    # relay fired on an operator's request, awaiting open
    "MANUAL_OPEN",       # roof open by hand; the night machine is otherwise idle
    "MANUAL_CLOSING",    # relay fired on an operator's request, awaiting closed
)

# ---------------------------------------------------------------- events

EVENTS = (
    # clock / planner
    "NOON_TICK", "PLAN_GOOD", "PLAN_BAD", "PRE_SUNSET_TICK",
    "CHECKS_PASSED", "CHECKS_FAILED", "REPLAN_REQUESTED",
    # roof sensing / motion outcomes. ROOF_FIRE_FAILED: the relay command
    # never reached the relay (2026-09-16 11:11, the Shelly did not answer
    # for the ten seconds the run needed it) and vision re-read the roof at
    # its start position -- sense and expectation AGREE, so it is not a
    # fault; the guard on its rows demands that fresh read.
    "ROOF_OPEN_CONFIRMED", "ROOF_CLOSE_CONFIRMED", "ROOF_STALL",
    "ROOF_TIMEOUT", "ROOF_FIRE_FAILED", "VISION_CONTRADICTION",
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
    # mount. The two REQUESTED events are permissions (Phase 2b): a slew /
    # home / park / unpark launch, and switching the mount's power on. They
    # are answered by rows whose dst == src -- the caller may act, the night
    # does not move.
    "MOUNT_PARK_CONFIRMED", "MOUNT_MOVE_REQUESTED", "MOUNT_POWER_REQUESTED",
    # night lifecycle
    "NIGHT_END_REQUESTED", "SHUTDOWN_DONE", "DAY_TICK",
    # weather
    "WEATHER_BAD",
    # operator
    "SAFETY_CLEARED", "SAFETY_ARMED", "ESTOP_REQUESTED", "OPERATOR_RESOLVE",
    # operator roof requests outside a night (roof!!, cycle_roof). The night's
    # own roof moves are CHECKS_PASSED (open) and MOUNT_PARK_CONFIRMED (close).
    "ROOF_OPEN_REQUESTED", "ROOF_CLOSE_REQUESTED",
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

# Roof moved by an operator outside a night. Roof motion is guarded and timed
# out here exactly as in the night's own OPENING_ROOF / CLOSING_ROOF.
MANUAL_STATES = frozenset({"MANUAL_OPENING", "MANUAL_OPEN", "MANUAL_CLOSING"})

# Every state in which the relay has fired and the roof has not yet been
# confirmed at rest. Entry into any of these is Invariant A's territory, and
# the conductor's watchdog turns a stay here that outlives the confirm loop
# into ROOF_TIMEOUT.
ROOF_MOVING_STATES = frozenset({"OPENING_ROOF", "CLOSING_ROOF",
                                "MANUAL_OPENING", "MANUAL_CLOSING"})

# The mount's requests: answered by permission rows (dst == src) below, and
# under conductor.mount_authority stepped with the real evidence.
MOUNT_DECISION_EVENTS = frozenset({"MOUNT_MOVE_REQUESTED", "MOUNT_POWER_REQUESTED"})

# The night's progression, in order, for consoles that show "where the night
# stands" (the website's Live tab). DERIVED from STATES' declaration order --
# which is the night's order, and is asserted to be by tests -- so a console
# rendering this list can never show a stage the machine does not have, and a
# state added to the table appears without touching any console. The holds
# and the manual roof states are excluded: they are not stages of a night,
# they are where a night stops or where the roof moves without one.
NIGHT_PATH = tuple(s for s in STATES
                   if s not in HOLD_STATES and s not in MANUAL_STATES)

TRANSITIONS = (
    # --- the day
    T("IDLE_DAY",     "NOON_TICK",           "PLANNING"),
    T("PLANNING",     "PLAN_GOOD",           "ARMED"),
    T("PLANNING",     "PLAN_BAD",            "IDLE_DAY"),
    T("ARMED",        "PRE_SUNSET_TICK",     "PRE_FLIGHT"),
    T("ARMED",        "WEATHER_BAD",         "IDLE_DAY"),

    # --- arming and opening. The whole of Invariant A rides on these rows.
    T("PRE_FLIGHT",   "CHECKS_PASSED",       "OPENING_ROOF",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known, G.weather_ok)),
    # A manual `image!!` opens the roof without waiting for the pre-sunset
    # tick, so the night starts from ARMED rather than PRE_FLIGHT. On
    # 2026-09-04 that run began at 17:45 and the machine, having no row here,
    # ignored the whole night: four events dropped before the scheduler even
    # woke, and it sat in PRE_FLIGHT until noon the next day.
    #
    # IDENTICAL GUARDS, deliberately. The operator choosing the moment does
    # not change what has to be true before a roof moves, and the evidence
    # rule is the guards' job in both cases. What the operator replaces is the
    # CLOCK, not the safety case.
    #
    # Only from ARMED: a manual run with no plan at all still lands in
    # IDLE_DAY as an ignored note, which is Phase 3's dataset for modelling
    # unplanned runs and is asserted by tests/test_shadow.py.
    T("ARMED",        "CHECKS_PASSED",       "OPENING_ROOF",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known, G.weather_ok)),
    # An UNPLANNED manual run: image!! on a day the noon check planned
    # nothing (or was never run). With the conductor deciding the roof
    # (Phase 2) a missing row is a refusal, and refusing every operator night
    # that the planner did not foresee is a lost night, not safety. The
    # safety case is identical; only the weather guard is absent, because an
    # operator starting a run IS the weather verdict for that run -- the
    # planner had none to offer. The conductor sizes the slot count from the
    # sequence on disk when this row fires (see shadow.offer).
    T("IDLE_DAY",     "CHECKS_PASSED",       "OPENING_ROOF",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known)),
    T("PRE_FLIGHT",   "CHECKS_FAILED",       "IDLE_DAY"),
    T("PRE_FLIGHT",   "WEATHER_BAD",         "IDLE_DAY"),
    T("OPENING_ROOF", "ROOF_OPEN_CONFIRMED", "PRELUDE"),
    T("OPENING_ROOF", "ROOF_STALL",          "FAULT_ROOF_UNKNOWN"),
    T("OPENING_ROOF", "ROOF_TIMEOUT",        "FAULT_ROOF_UNKNOWN"),
    # The fire never happened and the roof is sensed still shut: back to the
    # day the open was attempted from (a standing plan means ARMED). Without
    # the fresh sense there is no row, and the motion watchdog's ROOF_TIMEOUT
    # takes it to FAULT as before.
    T("OPENING_ROOF", "ROOF_FIRE_FAILED",    "ARMED",
      guards=(G.roof_closed, G.slots_remaining)),
    T("OPENING_ROOF", "ROOF_FIRE_FAILED",    "IDLE_DAY",
      guards=(G.roof_closed, G.plan_exhausted)),

    # --- the slots. One event, two rows: table order + guards do the fan-out.
    T("PRELUDE",      "NINA_PRELUDE_DONE",   "SLOT_SETUP",
      guards=(G.slots_remaining,)),
    T("PRELUDE",      "NINA_PRELUDE_DONE",   "FLATS",
      guards=(G.plan_exhausted,)),
    T("SLOT_SETUP",   "SLOT_STARTED",        "SLOT_IMAGING"),
    T("SLOT_IMAGING", "NINA_SLOT_DONE",      "SLOT_SETUP",
      guards=(G.slots_remaining,)),
    T("SLOT_IMAGING", "NINA_SLOT_DONE",      "PARKING",
      guards=(G.plan_exhausted,)),
    T("SLOT_IMAGING", "SLOT_WINDOW_END",     "SLOT_SETUP",
      guards=(G.slots_remaining,)),
    T("SLOT_IMAGING", "SLOT_WINDOW_END",     "PARKING",
      guards=(G.plan_exhausted,)),
    T("SLOT_IMAGING", "REPLAN_REQUESTED",    "SLOT_SETUP"),
    T("SLOT_IMAGING", "CAPTURE_LOST",        "PARKING"),
    T("SLOT_IMAGING", "WEATHER_BAD",         "PARKING"),
    # NINA can die in the prelude too: 2026-09-10 the Pegasus switch connect
    # failed 16 s after mount power-up and NINA quit with the roof open and
    # nothing driving. Without this row the machine sat in PRELUDE, every
    # later event of the night (two retries, one of them clean) was ignored,
    # and the site showed idle through a night of imaging. Same answer as
    # mid-slot: the close decision is unguarded, the motion after it is not.
    T("PRELUDE",      "CAPTURE_LOST",        "PARKING"),

    # --- closing out the night. The close DECISION is unguarded (deciding to
    # go home must always be possible); the roof MOTION is where Invariant A
    # bites, on the single PARKING -> CLOSING_ROOF edge.
    T("PARKING",      "MOUNT_PARK_CONFIRMED", "CLOSING_ROOF",
      guards=(G.mount_parked, G.roof_state_known)),
    T("PARKING",      "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("CLOSING_ROOF", "ROOF_CLOSE_CONFIRMED", "FLATS"),
    T("CLOSING_ROOF", "ROOF_STALL",          "FAULT_ROOF_UNKNOWN"),
    T("CLOSING_ROOF", "ROOF_TIMEOUT",        "FAULT_ROOF_UNKNOWN"),
    # Close never fired, roof sensed still open: back to PARKING, where the
    # close is asked for again (end.py retries, or an operator closes).
    T("CLOSING_ROOF", "ROOF_FIRE_FAILED",    "PARKING",
      guards=(G.roof_open,)),
    # Flats run shut, so nothing here can move the roof and every way out of
    # FLATS leads to SHUTDOWN. A capture failure during flats costs the flats,
    # not the night's safety: the observatory is already closed.
    T("FLATS",        "NINA_FLATS_DONE",     "SHUTDOWN"),
    T("FLATS",        "CAPTURE_LOST",        "SHUTDOWN"),
    T("SHUTDOWN",     "SHUTDOWN_DONE",       "NIGHT_DONE"),
    T("NIGHT_DONE",   "DAY_TICK",            "IDLE_DAY"),

    # --- an early end request from any active night state: the decision to
    # end is never guarded; the motion after parking is.
    T("PRELUDE",      "NIGHT_END_REQUESTED", "PARKING"),
    T("SLOT_SETUP",   "NIGHT_END_REQUESTED", "PARKING"),
    T("SLOT_IMAGING", "NIGHT_END_REQUESTED", "PARKING"),
    # From FLATS the roof is already shut, so ending early is just skipping
    # the rest of the flats.
    T("FLATS",        "NIGHT_END_REQUESTED", "SHUTDOWN"),

    # --- contradiction between roof sensors, noticed at rest. IDLE_DAY,
    # ARMED and NIGHT_DONE are included: those states CLAIM the roof is
    # closed, so an open roof seen there is precisely the contradiction.
    T("IDLE_DAY",     "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("ARMED",        "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("NIGHT_DONE",   "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("PRELUDE",      "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("SLOT_SETUP",   "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("SLOT_IMAGING", "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("FLATS",        "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),

    # --- the roof moved by hand, outside a night: roof!! open/close and
    # scripts/cycle_roof.py. Same guards as the night's own roof moves
    # (Invariant A on every entry into motion); a request from any other
    # state -- mid-night, or from a hold -- has no row and is refused.
    # Closing is allowed from the resting states too, not only from
    # MANUAL_OPEN: a roof found open where the machine believed it closed
    # (a restart that lost MANUAL_OPEN, a roof opened by hand) must always be
    # closeable, and closing is the safe direction.
    T("IDLE_DAY",     "ROOF_OPEN_REQUESTED", "MANUAL_OPENING",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known)),
    T("ARMED",        "ROOF_OPEN_REQUESTED", "MANUAL_OPENING",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known)),
    T("NIGHT_DONE",   "ROOF_OPEN_REQUESTED", "MANUAL_OPENING",
      guards=(G.safety_armed, G.mount_parked, G.roof_state_known)),
    T("MANUAL_OPENING", "ROOF_OPEN_CONFIRMED", "MANUAL_OPEN"),
    T("MANUAL_OPENING", "ROOF_STALL",        "FAULT_ROOF_UNKNOWN"),
    T("MANUAL_OPENING", "ROOF_TIMEOUT",      "FAULT_ROOF_UNKNOWN"),
    T("MANUAL_OPENING", "ROOF_FIRE_FAILED",  "ARMED",
      guards=(G.roof_closed, G.slots_remaining)),
    T("MANUAL_OPENING", "ROOF_FIRE_FAILED",  "IDLE_DAY",
      guards=(G.roof_closed, G.plan_exhausted)),
    T("MANUAL_OPEN",  "ROOF_CLOSE_REQUESTED", "MANUAL_CLOSING",
      guards=(G.mount_parked, G.roof_state_known)),
    T("MANUAL_OPEN",  "VISION_CONTRADICTION", "FAULT_ROOF_UNKNOWN"),
    T("IDLE_DAY",     "ROOF_CLOSE_REQUESTED", "MANUAL_CLOSING",
      guards=(G.mount_parked, G.roof_state_known)),
    T("ARMED",        "ROOF_CLOSE_REQUESTED", "MANUAL_CLOSING",
      guards=(G.mount_parked, G.roof_state_known)),
    T("NIGHT_DONE",   "ROOF_CLOSE_REQUESTED", "MANUAL_CLOSING",
      guards=(G.mount_parked, G.roof_state_known)),
    # Closed again: back to the day the roof was opened from. A standing
    # night plan (slots) means ARMED, else IDLE_DAY.
    T("MANUAL_CLOSING", "ROOF_CLOSE_CONFIRMED", "ARMED",
      guards=(G.slots_remaining,)),
    T("MANUAL_CLOSING", "ROOF_CLOSE_CONFIRMED", "IDLE_DAY",
      guards=(G.plan_exhausted,)),
    T("MANUAL_CLOSING", "ROOF_STALL",        "FAULT_ROOF_UNKNOWN"),
    T("MANUAL_CLOSING", "ROOF_TIMEOUT",      "FAULT_ROOF_UNKNOWN"),
    T("MANUAL_CLOSING", "ROOF_FIRE_FAILED",  "MANUAL_OPEN",
      guards=(G.roof_open,)),

    # --- the mount asks (Phase 2b, 2026-09-25). Permission rows: dst == src,
    # the machine stays put, the caller may act. A MOVE (slew, home, park,
    # unpark, tracking on) only under a confirmed-open roof -- Invariant B --
    # and only in the states where a night or an operator legitimately
    # drives the scope. POWER additionally where the roof is confirmed shut
    # and the scope confirmed parked (flats after the close, a daytime
    # doflats): powering does not move the mount, and that is the safe
    # geometry. No row in a hold or in any other state, so the request is
    # refused: on 2026-09-17 a run that missed a stop! powered the mount and
    # launched flats while the conductor sat in SAFE_HOLD. A park inside a
    # hold is the one exception, decided by the conductor on the evidence
    # (shadow._park_in_hold), because stop! parks before it closes.
    T("PRELUDE",      "MOUNT_MOVE_REQUESTED",  "PRELUDE",      guards=(G.roof_open,)),
    T("SLOT_SETUP",   "MOUNT_MOVE_REQUESTED",  "SLOT_SETUP",   guards=(G.roof_open,)),
    T("SLOT_IMAGING", "MOUNT_MOVE_REQUESTED",  "SLOT_IMAGING", guards=(G.roof_open,)),
    T("PARKING",      "MOUNT_MOVE_REQUESTED",  "PARKING",      guards=(G.roof_open,)),
    T("MANUAL_OPEN",  "MOUNT_MOVE_REQUESTED",  "MANUAL_OPEN",  guards=(G.roof_open,)),
    T("PRELUDE",      "MOUNT_POWER_REQUESTED", "PRELUDE",      guards=(G.roof_open,)),
    T("SLOT_SETUP",   "MOUNT_POWER_REQUESTED", "SLOT_SETUP",   guards=(G.roof_open,)),
    T("SLOT_IMAGING", "MOUNT_POWER_REQUESTED", "SLOT_IMAGING", guards=(G.roof_open,)),
    T("PARKING",      "MOUNT_POWER_REQUESTED", "PARKING",      guards=(G.roof_open,)),
    T("MANUAL_OPEN",  "MOUNT_POWER_REQUESTED", "MANUAL_OPEN",  guards=(G.roof_open,)),
    T("IDLE_DAY",     "MOUNT_POWER_REQUESTED", "IDLE_DAY",
      guards=(G.mount_parked, G.roof_closed)),
    T("ARMED",        "MOUNT_POWER_REQUESTED", "ARMED",
      guards=(G.mount_parked, G.roof_closed)),
    T("NIGHT_DONE",   "MOUNT_POWER_REQUESTED", "NIGHT_DONE",
      guards=(G.mount_parked, G.roof_closed)),
    T("FLATS",        "MOUNT_POWER_REQUESTED", "FLATS",
      guards=(G.mount_parked, G.roof_closed)),

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
    kind: str                     # "transition" | "rejected" | "ignored" | "allowed"
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
    A row whose dst is its src is a PERMISSION: the outcome is ALLOWED and
    the state is unchanged (the mount's requests).
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
            if row.dst == row.src:
                return Outcome("allowed", state)
            return Outcome("transition", row.dst)
        if first_refusal is None:
            first_refusal = refusal
    if first_refusal is not None:
        return Outcome("rejected", state, guard=first_refusal)
    return Outcome("ignored", state)


INITIAL_STATE = "IDLE_DAY"
