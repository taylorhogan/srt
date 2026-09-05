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
                             parked_kasa=Tri.CONFIRMED,
                             parked_pwi4=Tri.CONFIRMED,
                             roof=Tri.DENIED, safety_armed=True,
                             mode_auto=True, weather_ok=True,
                             slots_remaining=1, nina_alive=True)

# The scope-top webcam's verdict line. The trailing vote counts are optional
# so lines written before they existed still parse, but when present they are
# what makes the webcam's park reading three-valued rather than a bool:
# "lit" is how many exposure rungs were bright enough to judge at all, so
# lit == 0 is the camera saying "I cannot see", which is UNKNOWN and not
# "not parked". vision_safety collapses both into parked=False.
_VISION_RE = re.compile(
    r"vision parked=(True|False) closed=(True|False) open=(True|False)"
    r"(?:.*?votes parked (\d+)/(\d+) lit)?")

# The indoor Kasa camera's verdict line (sentry/kasa_state.kasa_status). Its
# scope verdict is already three-valued at the source -- 'safe' / 'UNSAFE' /
# 'unknown' from the AprilTag comparison against the recorded park pose -- and
# was being written to the log and thrown away. Wording is coupled to that
# logger call; change them together.
_KASA_RE = re.compile(r"kasa_status: scope=(\w+) roof=(\w+)")
# The log-only anchor toggle_roof() writes immediately before firing the
# relay -- every roof move from every path passes it. Wording is coupled to
# that line; change them together.
_ROOF_FIRE_RE = re.compile(r"roof relay fire: direction=(\w+)")

# Evidence older than this decays to UNKNOWN. The legacy system runs a vision
# check seconds before any roof move, so at decision moments evidence is
# fresh; a verdict quoted against an hours-old snapshot would be a lie.
EVIDENCE_MAX_AGE_S = 15 * 60


def _read_pwi4_park():
    """'parked' | 'not_parked' | 'unreachable' -- read-only, never raises.

    Uses the same alt/az-vs-configured-park comparison as
    pwi4_utils.get_is_parked, but keeps the three-way distinction that
    function collapses: an unreachable PWI4 (mount powered off, service down)
    is UNKNOWN evidence, not "not parked"."""
    try:
        from configs import config
        from hardware_control.pwi4_client import PWI4
        s = PWI4().status()
        if not s.mount.is_connected:
            # get_is_parked would CONNECT here; the shadow refuses to command
            # anything, even a connect, and reports honest ignorance instead.
            return "unreachable"
        if s.mount.is_slewing or s.mount.is_tracking:
            return "not_parked"
        cfg = config.data()["camera safety"]
        d_alt = abs(cfg["parked altitude deg"] - s.mount.altitude_degs)
        d_az = abs(cfg["parked azimuth deg"] - s.mount.azimuth_degs)
        # Same 1-degree window get_is_parked hard-codes.
        return "parked" if (d_alt < 1.0 and d_az < 1.0) else "not_parked"
    except Exception:
        return "unreachable"


def _read_roof_limits():
    """(state, detail) from the limit switches; NOT_CONFIGURED until the
    hardware exists. Never raises."""
    try:
        from hardware_control import roof_limit_switches as rls
        r = rls.read()
        return r.state, r.detail
    except Exception as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"


class ShadowConductor:
    # States that mean "a night is in progress on the hardware". Recovering
    # into one of these while imaging.txt reads NONE and NINA is gone means
    # the shadow missed the night's end (a mapping gap, a crash) -- reality
    # has moved on and faithfully resuming the wedge just extends it.
    _MID_NIGHT = {"OPENING_ROOF", "PRELUDE", "SLOT_SETUP", "SLOT_IMAGING",
                  "FLATS", "PARKING", "CLOSING_ROOF", "SHUTDOWN"}

    # States the machine can still be sitting in when imaging.txt reaches
    # IN_FLATS -- i.e. the walk to FLATS has not happened yet and the events
    # that would have driven it need synthesizing.
    _BEFORE_FLATS = {"SLOT_IMAGING", "PARKING", "CLOSING_ROOF"}

    # How many consecutive NONE polls to wait, sitting in FLATS, before
    # concluding the flats are not coming. imaging.txt cannot distinguish "the
    # roof just shut and the flats start in a minute" from "the night ended
    # without flats" -- both read NONE. Measured 2026-09-05, the real gap was
    # 54 s (NONE at 02:28:58, IN_FLATS at 02:29:52), so at the 5 s cadence 60
    # polls is five minutes: far past any real launch, and it only ever
    # ADVANCES a night that is already over. Injectable so tests need not
    # spin.
    _flats_grace_polls = 60

    def __init__(self, repo_root: Path, journal: Journal):
        self.root = Path(repo_root)
        self.journal = journal
        self.state = INITIAL_STATE
        self.slots = 0
        self._none_streak = 0         # consecutive polls reading NONE (debounce)
        # Slow sensors (network round-trips) polled every N fast polls.
        # Injectable so tests drive them without a mount or a Shelly.
        self.pwi4_probe = _read_pwi4_park
        self.limits_probe = _read_roof_limits
        self._slow_every = 6          # ~30 s at the 5 s cadence
        self._slow_tick = 0
        self._limits = ("not_configured", "")
        self._evidence_ts = 0.0       # when a vision line last updated evidence
        self._kasa_ts = 0.0           # ditto for the indoor camera, which can
                                      # fail independently and must decay on
                                      # its own clock rather than ride the
                                      # webcam's freshness
        self._cams_split = False      # last known camera (dis)agreement
        self._flats_none_streak = 0   # polls sat in FLATS reading NONE
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
        # Wedge recovery. On 2026-08-28 a torn imaging.txt read fired the close
        # cascade mid-slot; every event was (correctly) ignored and the machine
        # sat in SLOT_IMAGING for 16 hours while reality planned the next
        # night. Resuming that faithfully would resume the wedge, so a
        # recovered mid-night state with no capture activity is re-seated to
        # IDLE_DAY -- journaled as a SHADOW_RESYNC note, because a divergence
        # silently papered over is a divergence the morning report cannot
        # count. Shadow-only behaviour: the authoritative conductor treats the
        # same discrepancy as a FAULT, not a shrug.
        if (self.state in self._MID_NIGHT and self._imaging == "NONE"
                and not self._nina):
            self.journal.append(
                "note", "SHADOW_RESYNC", "shadow",
                data={"from_state": self.state,
                      "reason": "recovered mid-night state with no capture activity"})
            self.state, self.slots = INITIAL_STATE, 0
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
                parked, closed, is_open = (x == "True" for x in m.groups()[:3])
                lit = m.group(5)
                roof = (Tri.CONFIRMED if is_open
                        else Tri.DENIED if closed else Tri.UNKNOWN)
                if parked:
                    pv = Tri.CONFIRMED
                elif lit is None:
                    # Pre-vote log line: cannot tell "saw it off park" from
                    # "could not see", so claim the weaker of the two.
                    pv = Tri.UNKNOWN
                else:
                    # The camera judged it: no lit rung means it could not see
                    # anything, which is ignorance, not a negative verdict.
                    pv = Tri.DENIED if int(lit) > 0 else Tri.UNKNOWN
                self.evidence = self.evidence.replace(parked_vision=pv,
                                                      roof=roof)
                self._evidence_ts = time.time()

            m = _KASA_RE.search(ln)
            if m:
                self.evidence = self.evidence.replace(
                    parked_kasa={"safe": Tri.CONFIRMED,
                                 "UNSAFE": Tri.DENIED}.get(m.group(1),
                                                           Tri.UNKNOWN))
                self._kasa_ts = time.time()

        # Once per batch, never per line. The two cameras log about four
        # seconds apart and the poll runs every five, so a per-line check
        # would report a "split" on every single vision check purely because
        # the webcam's line is read before the indoor camera's.
        self._note_camera_split()

    def _offer_clock(self, event, sched):
        """Offer a SCHEDULER CLOCK event, unless a night is already running.

        The scheduler is a separate state machine that keeps ticking whether or
        not it is the thing driving the night. When a manual `image!!` owns the
        night its ticks are commentary, and translating them into night
        lifecycle events reports a night that is not happening.

        Measured on 2026-09-04, both from that one manual run: the pre-sunset
        tick arrived at 19:09 with the main sequence already imaging since
        17:50, and a DAY_TICK arrived 30 s later because the scheduler had
        touched IMAGING for three seconds and stood down ("Mode is manual --
        skipping auto imaging"). Neither described the observatory. The second
        is the worse of the two: DAY_TICK means "a night ended", and no night
        had ended -- none had been started by the scheduler at all.

        So mid-night these are journaled with their reason and not offered.
        Not silently dropped: a note keeps them in the record, which is what
        Phase 3 will need when the scheduler is absorbed and this ambiguity
        has to be designed away rather than sidestepped.
        """
        if self.state in self._MID_NIGHT:
            self.journal.append(
                "note", event, "shadow",
                data={"sched": sched, "suppressed_in_state": self.state,
                      "why": "a night is already running; the scheduler's "
                             "clock does not describe it"})
            return
        self.offer(event, "shadow", {"sched": sched})

    def _note_camera_split(self):
        """Journal the moment the two park cameras CONTRADICT each other, and
        the moment they stop.

        Contradiction means both cameras have an opinion and the opinions
        differ — one says parked, the other says not parked. That should never
        happen, and it is what "log if they ever disagree" is asking for.

        One camera reading UNKNOWN is deliberately NOT logged here. It is not a
        disagreement, it is a camera declining to answer (dark, occluded, not
        yet reported this cycle, or decayed), and it is already visible in the
        journal as the guard's own refusal on any move attempted while it
        lasts. Logging it here as well would bury the real contradictions in
        thousands of routine lines.

        Edge-triggered: the transition is the event, not the polls either side
        of it. Frequency and duration of these are exactly the evidence a
        later decision to soften the two-camera rule would need.
        """
        e = self.evidence
        split = (e.parked_vision is not e.parked_kasa
                 and Tri.UNKNOWN not in (e.parked_vision, e.parked_kasa))
        if split == self._cams_split:
            return
        self._cams_split = split
        self.journal.append(
            "note", "PARK_CAMERAS_SPLIT" if split else "PARK_CAMERAS_AGREE",
            "shadow",
            data={"webcam": e.parked_vision.name, "kasa": e.parked_kasa.name})

    def _poll_slow_sensors(self):
        """PWI4 park state and roof limit switches, every _slow_every polls.

        Both are strictly read-only network round-trips; keeping them off the
        5 s cadence keeps an unreachable mount from stalling every poll."""
        self._slow_tick += 1
        if self._slow_tick % self._slow_every != 1:
            return
        park = self.pwi4_probe()
        self.evidence = self.evidence.replace(
            parked_pwi4={"parked": Tri.CONFIRMED,
                         "not_parked": Tri.DENIED}.get(park, Tri.UNKNOWN))
        self._limits = self.limits_probe()

    def _roof_from_limits_and_vision(self, vision_roof):
        """Combine the two roof modalities. Agreement wins; contradiction is
        UNKNOWN (a lying sensor must cost a refusal, not a wrong move); an
        unfitted/faulted/unreachable switch pair leaves vision alone."""
        state = self._limits[0]
        if state == "open":
            return Tri.UNKNOWN if vision_roof is Tri.DENIED else Tri.CONFIRMED
        if state == "closed":
            return Tri.UNKNOWN if vision_roof is Tri.CONFIRMED else Tri.DENIED
        if state == "in_transit":
            return Tri.UNKNOWN
        return vision_roof

    def _current_evidence(self):
        e = self.evidence
        # Vision evidence decays: a verdict quoted against an hours-old
        # snapshot is a lie. The legacy system runs vision seconds before any
        # roof move, so at decision moments this is always fresh.
        now = time.time()
        if now - self._evidence_ts > EVIDENCE_MAX_AGE_S:
            e = e.replace(parked_vision=Tri.UNKNOWN, roof=Tri.UNKNOWN)
        # The indoor camera decays separately: it can stop reporting while the
        # webcam keeps going, and a stale AprilTag reading standing in as the
        # second confirmation would defeat the point of having two.
        if now - self._kasa_ts > EVIDENCE_MAX_AGE_S:
            e = e.replace(parked_kasa=Tri.UNKNOWN)
        return e.replace(
            roof=self._roof_from_limits_and_vision(e.roof),
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
        self._poll_slow_sensors()

        # --- scheduler transitions -> planner events
        sched, will = self._read_sched()
        if sched is not None and sched != self._sched:
            prev = self._sched
            if sched == "NOON_CHECK":
                # A new day starting while the machine still thinks a night is
                # in progress means the shadow lost the night's end. Re-seat to
                # IDLE_DAY (journaled) so the new day is tracked instead of a
                # second day of ignored events. Holds are exempt: SAFE_HOLD /
                # FAULT / ESTOP are deliberate and only an operator exits them.
                if self.state != "IDLE_DAY" and self.state not in HOLD_STATES:
                    self.journal.append(
                        "note", "SHADOW_RESYNC", "shadow",
                        data={"from_state": self.state,
                              "reason": "noon tick while machine was mid-night"})
                    self.state, self.slots = INITIAL_STATE, 0
                self.offer("NOON_TICK", "shadow", {"sched": sched})
            elif sched == "WAITING_FOR_PRE_SUNSET":
                self.slots = 1
                self.offer("PLAN_GOOD", "shadow", {"sched": sched, "dso": will})
            elif sched == "WAITING_FOR_NOON" and prev == "NOON_CHECK":
                self.offer("PLAN_BAD", "shadow", {"sched": sched})
            elif sched == "PRE_SUNSET_CHECK":
                self._offer_clock("PRE_SUNSET_TICK", sched)
            elif sched == "WAITING_FOR_NOON" and prev in ("IMAGING", "WAITING_FOR_BOOT"):
                self._offer_clock("DAY_TICK", sched)
            self._sched, self._will_image = sched, will

        # --- imaging.txt transitions -> capture events
        img = self._read_imaging()
        # Debounce NONE: imaging.txt is rewritten in place, and a read can land
        # mid-write on an empty file, which _read_imaging reports as NONE. On
        # 2026-08-28 one such flicker (IN_MAIN -> NONE at 03:10, flats an hour
        # later) fired the close cascade mid-slot and wedged the machine. A
        # real end state persists; a torn read does not survive two polls.
        if img == "NONE" and self._imaging not in (None, "NONE"):
            self._none_streak += 1
            if self._none_streak < 2:
                img = self._imaging          # not yet believed
        else:
            self._none_streak = 0
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
                # The roof is ALREADY SHUT by now -- end.py parks and closes
                # before launching the flats, which run against a panel. So by
                # this point the machine should have been walked to FLATS by
                # the NONE branch below, and there is nothing left to offer.
                #
                # It used to map to NINA_SLOT_DONE, from a model where flats
                # preceded the close. That is why the same night produced the
                # close cascade twice: once on the NONE that really is the
                # end of the main sequence, and again on the NONE that ends
                # the flats.
                #
                # Anything still due gets synthesized, tagged, so a night that
                # skipped straight from IN_MAIN without the intervening NONE
                # (a poll that lands badly, a torn read that survives the
                # debounce) still reaches FLATS rather than stalling.
                self.slots = 0
                for ev in ("NINA_SLOT_DONE", "MOUNT_PARK_CONFIRMED",
                           "ROOF_CLOSE_CONFIRMED"):
                    if self.state in self._BEFORE_FLATS:
                        self.offer(ev, "shadow",
                                   {"imaging": img, "synthesized": True})
            elif img == "DONE_FLATS":
                self.offer("NINA_FLATS_DONE", "shadow", {"imaging": img})
            elif img == "NONE" and prev == "IN_MAIN":
                # The main sequence ended and end.py ran: park, then close,
                # then clear the file. Measured 2026-09-05 -- "Begin End
                # Sequence" 02:24:36, relay closed 02:26:38, imaging.txt NONE
                # 02:28:58, flats launched 02:29:46. So this NONE is the roof
                # CLOSING, not the night ending, and the flats follow it.
                self.slots = 0
                self.offer("NINA_SLOT_DONE", "shadow", {"imaging": img})
                self.offer("MOUNT_PARK_CONFIRMED", "shadow", {"imaging": img})
                self.offer("ROOF_CLOSE_CONFIRMED", "shadow", {"imaging": img})
            elif img == "NONE" and prev in ("IN_FLATS", "DONE_FLATS"):
                # NOW the night is over. A run that never showed DONE_FLATS --
                # last night's flats were killed by the stall watchdog -- has
                # its completion synthesized and tagged.
                if prev == "IN_FLATS":
                    self.offer("NINA_FLATS_DONE", "shadow",
                               {"imaging": img, "synthesized": True})
                self.offer("SHUTDOWN_DONE", "shadow", {"imaging": img})
            self._imaging = img

        # --- flats that never came. Sitting in FLATS with the state file long
        # since cleared means end.py finished without running any, so close the
        # night out rather than leaving the machine parked in a stage reality
        # has already left. Tagged synthesized: nothing observed these.
        if self.state == "FLATS" and img == "NONE":
            self._flats_none_streak += 1
            if self._flats_none_streak >= self._flats_grace_polls:
                self._flats_none_streak = 0
                self.offer("NINA_FLATS_DONE", "shadow",
                           {"imaging": img, "synthesized": True,
                            "why": "no flats after %d polls"
                                   % self._flats_grace_polls})
                self.offer("SHUTDOWN_DONE", "shadow",
                           {"imaging": img, "synthesized": True})
        else:
            self._flats_none_streak = 0

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

        # --- decision-diff: every observed relay fire gets a guard verdict.
        # toggle_roof logs the anchor line immediately before firing; the
        # journal records what Invariant A's guards would have said at that
        # moment, from live evidence. This is Phase 2's dataset: a "would have
        # refused" on a move legacy made is either a guard bug or a legacy
        # bug, and the morning report surfaces each one.
        for ln in lines:
            m = _ROOF_FIRE_RE.search(ln)
            if m:
                ev = self._current_evidence()
                would = G.evaluate((G.mount_parked, G.roof_state_known), ev)
                self.journal.append(
                    "note", "ROOF_FIRE_OBSERVED", "shadow",
                    data={"direction": m.group(1),
                          "guard_would": would,     # None == would have allowed
                          "evidence": {"parked_vision": ev.parked_vision.name,
                                       "parked_kasa": ev.parked_kasa.name,
                                       "parked_pwi4": ev.parked_pwi4.name,
                                       "roof": ev.roof.name,
                                       "limits": self._limits[0]}})

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
