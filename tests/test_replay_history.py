"""Replay five months of REAL nights through the Night machine.

tests/replay/imaging_state_2026-08-28.log is a committed snapshot of the
observatory's actual imaging_state.log: 58 nights, 2026-03 through 2026-08.
It contains everything synthetic tests do not think to invent — NINA
restarting mid-night (double and quadruple DONE_PRELUDE), aborted nights that
never reach flats, a night where flats fired four times, and one night with
two complete runs back to back.

For each recorded night the test reconstructs the event stream the shadow
conductor would synthesize and walks the machine through it, asserting:

  * the machine NEVER rejects an event on the permissive snapshot (a
    rejection here would mean the table itself is inconsistent), and never
    leaves the declared state set;
  * every structurally clean night (exactly one prelude, ending in flats)
    reaches NIGHT_DONE;
  * every anomalous night is absorbed as IGNOREs — the machine shrugs at a
    duplicate signal rather than derailing — and still ends in a coherent
    state.

If the LIVE imaging_state.log has grown beyond the fixture, the same
assertions run over it too, so every historical night the observatory ever
records keeps validating the table for free.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

from apps.shadow_report import _ISLOG_RE
from iris.core.machine import INITIAL_STATE, STATES, step
from iris.core.snapshot import SensorSnapshot, Tri

FIXTURE = Path(__file__).parent / "replay" / "imaging_state_2026-08-28.log"
LIVE = Path(__file__).resolve().parent.parent / "imaging_state.log"

PERMISSIVE = SensorSnapshot(parked_vision=Tri.CONFIRMED, parked_pwi4=Tri.CONFIRMED,
                            roof=Tri.DENIED, safety_armed=True, mode_auto=True,
                            weather_ok=True, slots_remaining=1, nina_alive=True)


def _nights(path):
    """{night_date: [state, ...]} in recorded order."""
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ISLOG_RE.match(ln.strip())
        if not m:
            continue
        mo, dy, yr, hh, mi, ss, state = m.groups()
        when = datetime(int(yr), int(mo), int(dy), int(hh), int(mi), int(ss))
        out.setdefault((when - timedelta(hours=12)).date(), []).append(state)
    return out


def _night_events(states):
    """The event stream the shadow synthesizes for one recorded night.

    imaging_state.log only records the .bat-written anchors (DONE_PRELUDE,
    DONE_FLATS); the file transitions between them (IN_MAIN, IN_FLATS, NONE)
    are not logged but always happen, so they are interpolated the same way
    the shadow's imaging.txt watcher emits them.
    """
    ev = [("NOON_TICK", 1), ("PLAN_GOOD", 1), ("PRE_SUNSET_TICK", 1),
          ("CHECKS_PASSED", 1), ("ROOF_OPEN_CONFIRMED", 1)]
    for s in states:
        if s == "DONE_PRELUDE":
            ev.append(("NINA_PRELUDE_DONE", 1))
            ev.append(("SLOT_STARTED", 1))        # imaging.txt -> IN_MAIN
        elif s == "DONE_FLATS":
            ev.append(("NINA_SLOT_DONE", 0))      # imaging.txt -> IN_FLATS
            ev.append(("NINA_FLATS_DONE", 0))
            ev.append(("ROOF_CLOSE_CONFIRMED", 0))
            ev.append(("SHUTDOWN_DONE", 0))
    ev.append(("DAY_TICK", 0))
    return ev


def _walk(states):
    """Run one night; return (final_state, outcome_counter)."""
    s = INITIAL_STATE
    outcomes = Counter()
    for event, slots in _night_events(states):
        out = step(s, event, PERMISSIVE.replace(slots_remaining=slots))
        outcomes[out.kind] += 1
        assert out.kind != "rejected", (states, s, event, out.guard)
        assert out.state in STATES
        s = out.state
    return s, outcomes


def _sources():
    yield "fixture", _nights(FIXTURE)
    if LIVE.exists():
        live = _nights(LIVE)
        if len(live) > len(_nights(FIXTURE)):
            yield "live", live


def test_fixture_is_the_real_history_not_a_toy():
    nights = _nights(FIXTURE)
    assert len(nights) >= 58
    patterns = Counter(tuple(v) for v in nights.values())
    # The anomalies that make this dataset worth replaying must be present;
    # if the fixture is ever regenerated and these vanish, the test suite got
    # weaker and should say so.
    assert patterns[("DONE_PRELUDE", "DONE_FLATS")] >= 30
    assert any(p.count("DONE_PRELUDE") >= 2 for p in patterns)   # NINA restarts
    assert any("DONE_FLATS" not in p for p in patterns)          # aborted nights
    assert any(p.count("DONE_FLATS") >= 2 for p in patterns)     # repeated flats


def test_every_recorded_night_replays_without_rejection():
    for name, nights in _sources():
        for night, states in sorted(nights.items()):
            final, outcomes = _walk(states)
            assert outcomes["rejected"] == 0, (name, night)


def test_clean_nights_reach_night_done_then_idle():
    for name, nights in _sources():
        for night, states in sorted(nights.items()):
            if states == ["DONE_PRELUDE", "DONE_FLATS"]:
                final, outcomes = _walk(states)
                assert final == "IDLE_DAY", (name, night, final)
                assert outcomes["ignored"] == 0, (name, night, outcomes)


def test_anomalous_nights_are_absorbed_not_derailed():
    """Duplicate preludes/flats become IGNOREs; the machine still finishes in
    a coherent place. An aborted night (no flats) parks the machine in the
    night states — which is TRUE: that night, the roof close happened outside
    the modelled flow (end.py or by hand), and the shadow's imaging.txt NONE
    edge is what would close it out in live operation."""
    for name, nights in _sources():
        for night, states in sorted(nights.items()):
            if states == ["DONE_PRELUDE", "DONE_FLATS"]:
                continue
            final, outcomes = _walk(states)
            # "Ends in flats -> reaches IDLE_DAY" holds only for nights that
            # actually RAN a prelude first. The replay's first execution found
            # the exception: 2026-03-28 is a DONE_FLATS with no prelude at all
            # — a standalone `doflats` session, not a night — and the machine
            # correctly refuses to pretend it was one (the events absorb as
            # IGNOREs and the walk never leaves the pre-slot states).
            ran_night = "DONE_PRELUDE" in states
            if states and states[-1] == "DONE_FLATS" and ran_night:
                assert final == "IDLE_DAY", (name, night, states, final)
            assert outcomes["ignored"] >= 1, (name, night, states, outcomes)
