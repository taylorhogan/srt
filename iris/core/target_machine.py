"""The Target machine — one DSO's life, from wish to published picture.

This unifies the split brain the exploration found: my_instructions.json's
status field (waiting/completed) and convergence.json's is-it-actually-done
verdict were the same fact in two files with a written rule that one must
never overwrite the other. Here the fact has one owner and seven states, and
the two behaviours the owner asked for fall out of the table rather than
being features:

  AUTO-STOP:    ACQUIRING -> CONVERGED on CONVERGENCE_MET. The night planner
                simply never allocates a slot to a target past ACQUIRING.
  AUTO-PUBLISH: CONVERGED -> RENDERING -> PUBLISHED, driven by the Spark's
                RENDER_DONE and the publisher's PUBLISH_DONE events. No human
                in the loop; every step journaled (kind: "target").

Events arrive from four sources: the operator (chat), nightly frame
accounting, the convergence evaluation (existing snr math, run post-night),
and the Spark/publisher jobs posting to /v1/events.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

TARGET_STATES = (
    "WISHED",      # requested; coordinates may not resolve yet
    "QUEUED",      # eligible: resolved, filter plan set
    "ACQUIRING",   # accumulating frames across nights
    "CONVERGED",   # goodness reached — planner stops scheduling it
    "RENDERING",   # Spark stack+denoise underway
    "PUBLISHED",   # the picture is on the site
    "RETIRED",     # archived; reopenable
)

TARGET_EVENTS = (
    "RESOLVED",           # coordinates + filter plan exist
    "RESOLVE_FAILED",     # name lookup failed — stays WISHED, journaled
    "FRAMES_ADDED",       # nightly: this target gained kept frames
    "CONVERGENCE_MET",    # per-filter thresholds all satisfied
    "CONVERGENCE_LOST",   # re-evaluation says no longer done (e.g. gates changed)
    "RENDER_STARTED",     # Spark picked it up
    "RENDER_DONE",        # products exist
    "RENDER_FAILED",      # back to CONVERGED; retried next morning
    "PUBLISH_DONE",       # gallery entry committed to the site
    "REOPENED",           # operator wants more/other data
    "RETIRE",             # operator or policy archives it
)


@dataclass(frozen=True)
class TT:
    src: str
    event: str
    dst: str
    guards: Sequence[Callable] = field(default_factory=tuple)


TARGET_TRANSITIONS = (
    TT("WISHED",     "RESOLVED",         "QUEUED"),
    TT("QUEUED",     "FRAMES_ADDED",     "ACQUIRING"),
    TT("ACQUIRING",  "FRAMES_ADDED",     "ACQUIRING"),   # explicit self-loop:
    #   the journal gets one target event per contributing night, which is the
    #   per-target history the website's target board renders.
    TT("ACQUIRING",  "CONVERGENCE_MET",  "CONVERGED"),
    TT("CONVERGED",  "CONVERGENCE_LOST", "ACQUIRING"),
    TT("CONVERGED",  "RENDER_STARTED",   "RENDERING"),
    TT("RENDERING",  "RENDER_DONE",      "RENDERING"),   # products recorded;
    #   publication is a separate actor's act, so RENDER_DONE alone does not
    #   move the state — PUBLISH_DONE does.
    TT("RENDERING",  "RENDER_FAILED",    "CONVERGED"),
    TT("RENDERING",  "PUBLISH_DONE",     "PUBLISHED"),
    TT("PUBLISHED",  "RETIRE",           "RETIRED"),
    TT("PUBLISHED",  "REOPENED",         "ACQUIRING"),
    TT("RETIRED",    "REOPENED",         "ACQUIRING"),
    # Operator can retire from anywhere active (abandon a target).
    TT("WISHED",     "RETIRE",           "RETIRED"),
    TT("QUEUED",     "RETIRE",           "RETIRED"),
    TT("ACQUIRING",  "RETIRE",           "RETIRED"),
    TT("CONVERGED",  "RETIRE",           "RETIRED"),
)


@dataclass(frozen=True)
class TargetOutcome:
    kind: str                    # "transition" | "ignored"
    state: str


def target_step(state: str, event: str,
                table: Sequence[TT] = TARGET_TRANSITIONS) -> TargetOutcome:
    """Pure, guardless for now: target transitions are bookkeeping, not
    hardware, so the failure mode of a wrong one is an incorrect record rather
    than a crushed telescope. Guards arrive if a policy ever needs them."""
    for row in table:
        if row.src == state and row.event == event:
            return TargetOutcome("transition", row.dst)
    return TargetOutcome("ignored", state)


TARGET_INITIAL_STATE = "WISHED"
