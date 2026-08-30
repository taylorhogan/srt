"""The shadow watcher, driven through a synthetic legacy night.

A temp directory stands in for the repo root; the test writes the same files
the real system writes (scheduler_state.json, imaging.txt, safety.txt,
iris.log) in the order a real night writes them, polls the shadow after each,
and asserts the journal tells the night's story. This is the executable
specification of the legacy->event mapping that shadow_report later verifies
against real nights.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.conductor.shadow import ShadowConductor
from iris.core.journal import Journal


def _mkroot(tmp_path):
    (tmp_path / "scheduler_state.json").write_text(
        json.dumps({"state": "WAITING_FOR_NOON", "dso": "Unknown",
                    "will image tonight": "Unknown"}))
    (tmp_path / "imaging.txt").write_text("IMAGING_STATE NONE")
    (tmp_path / "safety.txt").write_text("USER SAFE")
    (tmp_path / "mode.txt").write_text("MODE AUTO")
    (tmp_path / "iris.log").write_text("")
    (tmp_path / "local").mkdir()
    return tmp_path


def _shadow(root):
    sh = ShadowConductor(root, Journal(root / "local" / "journal"))
    # Hermetic: no real mount or Shelly round-trips from tests. Individual
    # tests override these to stage evidence.
    sh.pwi4_probe = lambda: "unreachable"
    sh.limits_probe = lambda: ("not_configured", "")
    return sh


def _sched(root, state, will="yes"):
    (root / "scheduler_state.json").write_text(
        json.dumps({"state": state, "dso": "trunk", "will image tonight": will}))


def _imaging(root, value):
    (root / "imaging.txt").write_text(f"IMAGING_STATE {value}")


def _transitions(sh):
    return [(e.event, e.from_state, e.to_state)
            for e in sh.journal.replay() if e.kind == "transition"]


def test_full_legacy_night_produces_the_canonical_timeline(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    assert sh.state == "IDLE_DAY"

    _sched(root, "NOON_CHECK"); sh.poll()
    _sched(root, "WAITING_FOR_PRE_SUNSET"); sh.poll()
    _sched(root, "PRE_SUNSET_CHECK"); sh.poll()
    _sched(root, "IMAGING"); sh.poll()
    _imaging(root, "ACTIVE"); sh.poll()
    _imaging(root, "IN_PRELUDE"); sh.poll()      # roof opened + prelude begins
    _imaging(root, "DONE_PRELUDE"); sh.poll()
    _imaging(root, "IN_MAIN"); sh.poll()
    _imaging(root, "IN_FLATS"); sh.poll()
    _imaging(root, "DONE_FLATS"); sh.poll()
    _imaging(root, "NONE"); sh.poll(); sh.poll()  # end.py finished; NONE must
    _sched(root, "WAITING_FOR_NOON"); sh.poll()   # survive two polls (debounce)

    got = _transitions(sh)
    expected = [
        ("NOON_TICK",           "IDLE_DAY",     "PLANNING"),
        ("PLAN_GOOD",           "PLANNING",     "ARMED"),
        ("PRE_SUNSET_TICK",     "ARMED",        "PRE_FLIGHT"),
        ("CHECKS_PASSED",       "PRE_FLIGHT",   "OPENING_ROOF"),
        ("ROOF_OPEN_CONFIRMED", "OPENING_ROOF", "PRELUDE"),
        ("NINA_PRELUDE_DONE",   "PRELUDE",      "SLOT_SETUP"),
        ("SLOT_STARTED",        "SLOT_SETUP",   "SLOT_IMAGING"),
        ("NINA_SLOT_DONE",      "SLOT_IMAGING", "FLATS"),
        ("NINA_FLATS_DONE",     "FLATS",        "PARKING"),
        ("MOUNT_PARK_CONFIRMED", "PARKING",     "CLOSING_ROOF"),
        ("ROOF_CLOSE_CONFIRMED", "CLOSING_ROOF", "SHUTDOWN"),
        ("SHUTDOWN_DONE",       "SHUTDOWN",     "NIGHT_DONE"),
        ("DAY_TICK",            "NIGHT_DONE",   "IDLE_DAY"),
    ]
    assert got == expected
    assert sh.state == "IDLE_DAY"


def test_weathered_out_night(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _sched(root, "NOON_CHECK"); sh.poll()
    _sched(root, "WAITING_FOR_NOON", will="no"); sh.poll()
    assert _transitions(sh) == [
        ("NOON_TICK", "IDLE_DAY", "PLANNING"),
        ("PLAN_BAD",  "PLANNING", "IDLE_DAY"),
    ]


def test_safety_edges_are_operator_events_and_hold(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    (root / "safety.txt").write_text("USER UNSAFE"); sh.poll()
    assert sh.state == "SAFE_HOLD"
    # legacy events cannot pull the machine out of the hold
    _sched(root, "NOON_CHECK"); sh.poll()
    assert sh.state == "SAFE_HOLD"
    (root / "safety.txt").write_text("USER SAFE"); sh.poll()
    assert sh.state == "IDLE_DAY"
    kinds = [(e.kind, e.event) for e in sh.journal.replay()]
    assert ("transition", "SAFETY_CLEARED") in kinds
    assert ("note", "NOON_TICK") in kinds          # journaled as ignored-note
    assert ("transition", "SAFETY_ARMED") in kinds


def test_manual_run_outside_the_night_is_journaled_not_lost(tmp_path):
    """A manual image!! run starts with the machine in IDLE_DAY. The events
    must not vanish: they land as notes with ignored_in_state, which is the
    dataset Phase 3 uses to model manual runs."""
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _imaging(root, "ACTIVE"); sh.poll()
    _imaging(root, "IN_PRELUDE"); sh.poll()
    notes = [e for e in sh.journal.replay() if e.kind == "note"]
    assert any(e.event == "CHECKS_PASSED" and
               e.data.get("ignored_in_state") == "IDLE_DAY" for e in notes)


def test_restart_mid_night_resumes_from_journal(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _sched(root, "NOON_CHECK"); sh.poll()
    _sched(root, "WAITING_FOR_PRE_SUNSET"); sh.poll()
    _sched(root, "PRE_SUNSET_CHECK"); sh.poll()
    _imaging(root, "IN_PRELUDE"); sh.poll()
    assert sh.state == "PRELUDE"
    # new process, same directory: recovers state AND does not re-emit events
    sh2 = _shadow(root)
    assert sh2.state == "PRELUDE"
    n_before = sh2.journal.head()
    sh2.poll()                                     # nothing changed on disk
    assert sh2.journal.head() == n_before


def _to_in_main(root, sh):
    """Walk a night to SLOT_IMAGING with imaging.txt at IN_MAIN."""
    _sched(root, "NOON_CHECK"); sh.poll()
    _sched(root, "WAITING_FOR_PRE_SUNSET"); sh.poll()
    _sched(root, "PRE_SUNSET_CHECK"); sh.poll()
    _imaging(root, "IN_PRELUDE"); sh.poll()
    _imaging(root, "DONE_PRELUDE"); sh.poll()
    _imaging(root, "IN_MAIN"); sh.poll()
    assert sh.state == "SLOT_IMAGING"


def test_transient_none_flicker_does_not_fire_the_close_cascade(tmp_path):
    """The 2026-08-28 wedge: imaging.txt read NONE for one poll mid-slot (a
    torn read), flats ran an hour later, and the premature close cascade left
    the machine ignoring events in SLOT_IMAGING for 16 hours."""
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _to_in_main(root, sh)
    _imaging(root, "NONE"); sh.poll()            # one flicker: not believed
    assert sh.state == "SLOT_IMAGING"
    _imaging(root, "IN_FLATS"); sh.poll()        # reality: flats begin
    _imaging(root, "DONE_FLATS"); sh.poll()
    _imaging(root, "NONE"); sh.poll(); sh.poll() # the real end persists
    assert sh.state == "NIGHT_DONE"
    got = [e.event for e in sh.journal.replay() if e.kind == "transition"]
    assert got.count("MOUNT_PARK_CONFIRMED") == 1


def test_main_ending_straight_to_none_still_closes_the_night(tmp_path):
    """A sequence abort skips IN_FLATS/DONE_FLATS entirely; the shadow must
    synthesize the completions the file never showed, tagged as such."""
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _to_in_main(root, sh)
    _imaging(root, "NONE"); sh.poll(); sh.poll()
    assert sh.state == "NIGHT_DONE"
    entries = [e for e in sh.journal.replay() if e.kind == "transition"]
    synth = {e.event for e in entries if e.data.get("synthesized")}
    assert synth == {"NINA_SLOT_DONE", "NINA_FLATS_DONE"}


def test_flats_reached_from_an_unexpected_prev_still_ends_the_slot(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _to_in_main(root, sh)
    _imaging(root, "ACTIVE"); sh.poll()          # an unmapped intermediate
    _imaging(root, "IN_FLATS"); sh.poll()        # prev != IN_MAIN
    assert sh.state == "FLATS"


def test_recovering_a_wedged_state_resyncs_to_idle(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _to_in_main(root, sh)
    _imaging(root, "NONE")                       # night long over on disk
    sh2 = _shadow(root)                          # restart
    assert sh2.state == "IDLE_DAY"
    notes = [e for e in sh2.journal.replay() if e.event == "SHADOW_RESYNC"]
    assert notes and notes[-1].data["from_state"] == "SLOT_IMAGING"


def test_noon_while_mid_night_resyncs_and_tracks_the_new_day(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _to_in_main(root, sh)
    _sched(root, "NOON_CHECK"); sh.poll()        # a NEW day begins
    assert sh.state == "PLANNING"
    assert any(e.event == "SHADOW_RESYNC" for e in sh.journal.replay())


def test_guard_counterfactual_is_recorded(tmp_path):
    """The transition into OPENING_ROOF fires permissively, but the entry must
    carry what the guards WOULD have said given observed evidence — with no
    vision lines seen yet, that is a refusal mentioning vision."""
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _sched(root, "NOON_CHECK"); sh.poll()
    _sched(root, "WAITING_FOR_PRE_SUNSET"); sh.poll()
    _sched(root, "PRE_SUNSET_CHECK"); sh.poll()
    _imaging(root, "IN_PRELUDE"); sh.poll()
    entries = {e.event: e for e in sh.journal.replay() if e.kind == "transition"}
    would = entries["CHECKS_PASSED"].data.get("guard_would")
    assert would and "vision" in would


def _log(root, line):
    with open(root / "iris.log", "a") as fh:
        fh.write(line + "\n")


def _fire_notes(sh):
    return [e for e in sh.journal.replay() if e.event == "ROOF_FIRE_OBSERVED"]


def test_roof_fire_with_full_evidence_would_be_allowed(tmp_path):
    """The decision-diff happy path: vision parked+closed, PWI4 parked --
    Invariant A's guards would have permitted the move legacy made."""
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    sh.pwi4_probe = lambda: "parked"
    _log(root, "08/30/2026 vision parked=True closed=True open=False x")
    _log(root, "08/30/2026 roof relay fire: direction=open")
    sh.poll()
    notes = _fire_notes(sh)
    assert len(notes) == 1
    assert notes[0].data["direction"] == "open"
    assert notes[0].data["guard_would"] is None
    assert notes[0].data["evidence"]["parked_pwi4"] == "CONFIRMED"


def test_roof_fire_with_unparked_mount_would_be_refused(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    sh.pwi4_probe = lambda: "not_parked"
    _log(root, "08/30/2026 vision parked=True closed=True open=False x")
    _log(root, "08/30/2026 roof relay fire: direction=close")
    sh.poll()
    would = _fire_notes(sh)[0].data["guard_would"]
    assert would is not None and "parked" in would


def test_limit_switches_contradicting_vision_read_unknown(tmp_path):
    """Limits say open, vision says closed: a lying sensor must cost a
    refusal (roof UNKNOWN), never a wrong move."""
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    sh.pwi4_probe = lambda: "parked"
    sh.limits_probe = lambda: ("open", "")
    _log(root, "08/30/2026 vision parked=True closed=True open=False x")
    _log(root, "08/30/2026 roof relay fire: direction=open")
    sh.poll()
    note = _fire_notes(sh)[0]
    assert note.data["evidence"]["roof"] == "UNKNOWN"
    assert note.data["guard_would"] is not None


def test_limit_switches_alone_carry_roof_state_when_vision_is_stale(tmp_path):
    import time as _time
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    sh.limits_probe = lambda: ("closed", "")
    _log(root, "08/30/2026 vision parked=True closed=True open=False x")
    sh.poll()
    sh._evidence_ts = _time.time() - 3600          # vision goes stale
    ev = sh._current_evidence()
    from iris.core.snapshot import Tri
    assert ev.parked_vision is Tri.UNKNOWN          # decayed
    assert ev.roof is Tri.DENIED                    # limits still fresh


def test_stale_vision_evidence_decays_to_unknown(tmp_path):
    import time as _time
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    _log(root, "08/30/2026 vision parked=True closed=True open=False x")
    sh.poll()
    from iris.core.snapshot import Tri
    assert sh._current_evidence().parked_vision is Tri.CONFIRMED
    sh._evidence_ts = _time.time() - 3600
    ev = sh._current_evidence()
    assert ev.parked_vision is Tri.UNKNOWN and ev.roof is Tri.UNKNOWN


def test_vision_log_lines_update_evidence(tmp_path):
    root = _mkroot(tmp_path)
    sh = _shadow(root)
    with open(root / "iris.log", "a") as fh:
        fh.write("08/28/2026 vision parked=True closed=True open=False x\n")
    sh.poll()
    from iris.core.snapshot import Tri
    assert sh.evidence.parked_vision is Tri.CONFIRMED
    assert sh.evidence.roof is Tri.DENIED
    with open(root / "iris.log", "a") as fh:
        fh.write("08/28/2026 vision parked=True closed=False open=True x\n")
    sh.poll()
    assert sh.evidence.roof is Tri.CONFIRMED
