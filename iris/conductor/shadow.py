"""The shadow watcher: legacy state observed, events synthesized, nothing touched.

This process may READ: imaging.txt, scheduler_state.json, safety.txt, mode.txt,
the NINA process table, and iris.log. It may WRITE: only the journal. It must
NEVER open a camera (the live vision path owns the webcam behind a lock in the
social-server process; a second reader would contend with the safety system),
and it must never actuate anything -- shadow authority is exactly zero.

Two design points worth understanding before editing:

TRANSITIONS FOLLOW REALITY; GUARDS ARE COUNTERFACTUAL. When the legacy system
opens the roof, the shadow's machine must follow it into the night -- otherwise
one early divergence wedges the shadow and the rest of the night's journal is
garbage. So events are stepped with a permissive snapshot, and the EVIDENCE
snapshot (built from log-observed vision verdicts and file states) is evaluated
separately against the fired row's guards, with the verdict recorded in the
journal entry's data as `guard_would`. That field is Phase 2's dataset: every
place the guards WOULD have refused where legacy proceeded is either a guard
bug or a legacy bug, and the morning report surfaces each one.

EVIDENCE COMES FROM THE LOG, NOT THE SENSORS. Vision verdicts are parsed from
iris.log lines the live system already writes ("vision parked=... closed=...
open=...") rather than re-running vision. This means evidence can be stale or
absent -- which is recorded honestly as UNKNOWN, never guessed.
"""
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from iris.core import guards as G
from iris.core.journal import Journal
from iris.core.machine import HOLD_STATES, INITIAL_STATE, TRANSITIONS, step
from iris.core.snapshot import SensorSnapshot, Tri

_logger = logging.getLogger(__name__)

# The permissive snapshot transitions are stepped with (see module docstring).
# slots_remaining is patched per-event from the shadow's own slot counter.
_PERMISSIVE = SensorSnapshot(parked_vision=Tri.CONFIRMED,
                             parked_pwi4=Tri.CONFIRMED,
                             roof=Tri.DENIED, safety_armed=True,
                             mode_auto=True, weather_ok=True,
                             slots_remaining=1, nina_alive=True)

_VISION_RE = re.compile(
    r"vision parked=(True|False) closed=(True|False) open=(True|False)")


class ShadowConductor:
    def __init__(self, repo_root: Path, journal: Journal):
        self.root = Path(repo_root)
        self.journal = journal
        self.state = INITIAL_STATE
        self.slots = 0
        # last-seen values of the legacy sources, None = not yet read
        self._sched = None            # scheduler_state.json "state"
        self._will_image = None
        self._imaging = None          # imaging.txt value
        self._safety = None           # bool
        self._nina = None             # bool
        self._log_pos = None          # byte offset into iris.log
        self.evidence = SensorSnapshot()
        self._recover()

    # ------------------------------------------------------------ recovery

    def _recover(self):
        """Resume from the journal so a shadow restart mid-night does not
        replay the day's events as if new."""
        last = None
        for e in self.journal.replay():
            if e.kind == "transition":
                last = e
        if last and last.to_state:
            self.state = last.to_state
            self.slots = int(last.data.get("slots_after", 0))
        # Prime last-seen values so the first poll only reacts to CHANGES
        # after restart, not to the standing state.
        self._sched, self._will_image = self._read_sched()
        self._imaging = self._read_imaging()
        self._safety = self._read_safety()
        self._nina = self._nina_running()
        log = self.root / "iris.log"
        self._log_pos = log.stat().st_size if log.exists() else 0
        _logger.info("shadow recovered: state=%s slots=%d journal head=%d",
                     self.state, self.slots, self.journal.head())

    # ------------------------------------------------------------ readers

    def _read_sched(self):
        import json
        try:
            d = json.loads((self.root / "scheduler_state.json").read_text())
            return d.get("state"), str(d.get("will image tonight", "")).lower()
        except Exception:
            return None, None

    def _read_imaging(self):
        try:
            parts = (self.root / "imaging.txt").read_text().strip().split()
            return parts[1] if len(parts) >= 2 else "NONE"
        except Exception:
            return "NONE"

    def _read_safety(self):
        try:
            return (self.root / "safety.txt").read_text().strip() == "USER SAFE"
        except Exception:
            return False

    def _read_mode_auto(self):
        try:
            return (self.root / "mode.txt").read_text().strip() == "MODE AUTO"
        except Exception:
            return False

    def _nina_running(self):
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq NINA.exe"],
                capture_output=True, text=True, timeout=15)
            return "NINA.exe" in out.stdout
        except Exception:
            return self._nina if self._nina is not None else False

    def _read_new_log_lines(self):
        log = self.root / "iris.log"
        try:
            size = log.stat().st_size
        except OSError:
            return []
        if self._log_pos is None or size < self._log_pos:
            self._log_pos = 0         # rotated/truncated: start over
        if size == self._log_pos:
            return []
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self._log_pos)
            chunk = fh.read(size - self._log_pos)
        self._log_pos = size
        return chunk.splitlines()

    # ------------------------------------------------------------ evidence

    def _update_evidence_from_log(self, lines):
        for ln in lines:
            m = _VISION_RE.search(ln)
            if m:
                parked, closed, is_open = (x == "True" for x in m.groups())
                roof = (Tri.CONFIRMED if is_open
                        else Tri.DENIED if closed else Tri.UNKNOWN)
                self.evidence = self.evidence.replace(
                    parked_vision=Tri.CONFIRMED if parked else Tri.UNKNOWN,
                    roof=roof)

    def _current_evidence(self):
        return self.evidence.replace(
            safety_armed=bool(self._safety),
            mode_auto=self._read_mode_auto(),
            slots_remaining=self.slots,
            nina_alive=bool(self._nina))

    # ------------------------------------------------------------ stepping

    def _fired_row(self, state, event, snap):
        for row in TRANSITIONS:
            if row.event != event:
                continue
            if row.src == "*":
                if state in HOLD_STATES or state == row.dst:
                    continue
            elif row.src != state:
                continue
            if G.evaluate(row.guards, snap) is None:
                return row
        return None

    def offer(self, event: str, source: str, data: dict = None):
        """Step the machine with the permissive snapshot; journal everything,
        including the counterfactual guard verdict on the fired row."""
        snap = _PERMISSIVE.replace(slots_remaining=self.slots)
        out = step(self.state, event, snap)
        payload = dict(data or {})
        if out.kind == "transition":
            row = self._fired_row(self.state, event, snap)
            if row is not None and row.guards:
                would = G.evaluate(row.guards, self._current_evidence())
                payload["guard_would"] = would      # None == would have passed
            payload["slots_after"] = self.slots
            self.journal.append("transition", event, source,
                                from_state=self.state, to_state=out.state,
                                data=payload)
            self.state = out.state
        elif out.kind == "rejected":
            self.journal.append("rejected", event, source,
                                from_state=self.state, guard=out.guard,
                                data=payload)
        else:
            # Ignored events are journaled as notes in shadow mode: they are
            # exactly the mismatches between reality and the table that Phase 3
            # needs to know about (e.g. a manual image!! run starting outside
            # the modelled night).
            payload["ignored_in_state"] = self.state
            self.journal.append("note", event, source, data=payload)

    # ------------------------------------------------------------ polling

    def poll(self):
        """One observation pass. Called every few seconds by the runner."""
        lines = self._read_new_log_lines()
        self._update_evidence_from_log(lines)

        # --- scheduler transitions -> planner events
        sched, will = self._read_sched()
        if sched is not None and sched != self._sched:
            prev = self._sched
            if sched == "NOON_CHECK":
                self.offer("NOON_TICK", "shadow", {"sched": sched})
            elif sched == "WAITING_FOR_PRE_SUNSET":
                self.slots = 1
                self.offer("PLAN_GOOD", "shadow", {"sched": sched, "dso": will})
            elif sched == "WAITING_FOR_NOON" and prev == "NOON_CHECK":
                self.offer("PLAN_BAD", "shadow", {"sched": sched})
            elif sched == "PRE_SUNSET_CHECK":
                self.offer("PRE_SUNSET_TICK", "shadow", {"sched": sched})
            elif sched == "WAITING_FOR_NOON" and prev in ("IMAGING", "WAITING_FOR_BOOT"):
                self.offer("DAY_TICK", "shadow", {"sched": sched})
            self._sched, self._will_image = sched, will

        # --- imaging.txt transitions -> capture events
        img = self._read_imaging()
        if img != self._imaging:
            prev = self._imaging
            if img == "IN_PRELUDE":
                # The legacy run opens the roof between the checks and the
                # prelude; the shadow sees only the prelude begin, so the two
                # machine steps are synthesized back to back. Their true
                # relative timing is in iris.log for the report to compare.
                self.offer("CHECKS_PASSED", "shadow", {"imaging": img})
                self.offer("ROOF_OPEN_CONFIRMED", "shadow", {"imaging": img})
            elif img == "DONE_PRELUDE":
                self.offer("NINA_PRELUDE_DONE", "shadow", {"imaging": img})
            elif img == "IN_MAIN":
                self.offer("SLOT_STARTED", "shadow", {"imaging": img})
            elif img == "IN_FLATS":
                if prev == "IN_MAIN":
                    self.slots = 0
                    self.offer("NINA_SLOT_DONE", "shadow", {"imaging": img})
            elif img == "DONE_FLATS":
                self.offer("NINA_FLATS_DONE", "shadow", {"imaging": img})
            elif img == "NONE" and prev in ("DONE_FLATS", "IN_FLATS", "IN_MAIN"):
                # end.py's last act is clearing the state; the close itself is
                # observed via the vision lines it logs on the way.
                self.offer("ROOF_CLOSE_CONFIRMED", "shadow", {"imaging": img})
                self.offer("SHUTDOWN_DONE", "shadow", {"imaging": img})
            self._imaging = img

        # --- safety flag edges -> operator events
        safe = self._read_safety()
        if safe != self._safety:
            self.offer("SAFETY_ARMED" if safe else "SAFETY_CLEARED", "operator",
                       {"safety_txt": safe})
            self._safety = safe

        # --- roof stall, from the log
        for ln in lines:
            if "roof stall watchdog" in ln.lower() and "cut" in ln.lower():
                self.offer("ROOF_STALL", "watchdog", {"log": ln[-160:]})

        # --- NINA process liveness edge
        nina = self._nina_running()
        if self._nina and not nina and self._imaging in ("IN_PRELUDE", "IN_MAIN"):
            self.offer("CAPTURE_LOST", "watchdog", {"imaging": self._imaging})
        self._nina = nina

    def run_forever(self, interval_s: float = 5.0, stop=None):
        _logger.info("shadow conductor watching (read-only), state=%s", self.state)
        while stop is None or not stop.is_set():
            try:
                self.poll()
            except Exception:
                _logger.exception("shadow poll failed (continuing)")
            time.sleep(interval_s)
