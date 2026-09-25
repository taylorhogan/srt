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

PHASE 2 (roof authority, 2026-09-16). The actuators still fire the relay, but
they ASK first: every roof-moving site posts its request and the evidence it
just sensed (iris/client.py), and this conductor steps the machine with that
evidence. With `conductor.roof_authority` ON the verdict is binding -- a
refusal is journaled with the guard's reason and the caller does not move.
OFF, the request is stepped permissively as any shadow event and the verdict
is returned as `would_refuse`: the same decision-diff, now on the caller's
own sensor read rather than a log line. The shadow watcher keeps running in
both modes; where a live post has already walked the machine, the
synthesized event for the same step is skipped rather than journaled as
ignored. Every roof-moving state is timed out into FAULT_ROOF_UNKNOWN when it
outlives the confirm loop, a restart that lands in a manual move faults the
same way, and end.py's last-resort close leaves a marker that is ingested at
the next start.
"""
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from iris.core import guards as G
from iris.core.journal import Journal
from iris.core.machine import (HOLD_STATES, INITIAL_STATE, ROOF_MOVING_STATES,
                               TRANSITIONS, step)
from iris.core.snapshot import SensorSnapshot, Tri
from iris.core.machine import MOUNT_DECISION_EVENTS

_logger = logging.getLogger(__name__)

# A roof move that has not been confirmed within this long has outlived the
# confirm loop (30 s + 5 x 5 min in super_user_commands.confirm_roof_state)
# by a margin: the actuator died, or forgot to report. The roof's position is
# then untrusted, which is what FAULT_ROOF_UNKNOWN means.
ROOF_MOTION_TIMEOUT_S = 30 * 60

# Where end.py's last-resort close (conductor unreachable) records what it
# did, for ingestion at the next conductor start.
FALLBACK_MARKER = Path("local") / "roof_fallback_marker.json"

# The events an actuator posts BEFORE moving the roof. Under authority these
# are stepped with the real evidence; every other live event (confirmations,
# operator resolves, end requests) has no sensor guard and steps as before.
ROOF_DECISION_EVENTS = frozenset({"ROOF_OPEN_REQUESTED", "ROOF_CLOSE_REQUESTED",
                                  "CHECKS_PASSED", "MOUNT_PARK_CONFIRMED",
                                  # a failed fire returns only on a fresh sense
                                  "ROOF_FIRE_FAILED"})


@dataclass(frozen=True)
class Verdict:
    """What offering one event produced, for the API to report."""
    kind: str                       # transition | rejected | ignored | allowed_in_hold
    state: str
    guard: Optional[str] = None     # the refusal, for rejected / ignored
    would_refuse: Optional[str] = None   # decision-diff verdict on a permissive step
    seq: int = 0

    @property
    def accepted(self) -> bool:
        """May the caller act? A transition, a permission row (the mount's
        requests), or a close / park permitted in a hold."""
        return self.kind in ("transition", "allowed", "allowed_in_hold")


def _read_roof_authority():
    try:
        from configs import config
        return bool(config.data().get("conductor", {}).get("roof_authority", False))
    except Exception:
        return False


def _read_mount_authority():
    try:
        from configs import config
        return bool(config.data().get("conductor", {}).get("mount_authority", False))
    except Exception:
        return False


def _tri(name) -> Tri:
    try:
        return Tri(str(name).upper())
    except ValueError:
        return Tri.UNKNOWN

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


def _read_sun_altitude():
    """Sun altitude in degrees at the observatory, or None. Never raises."""
    try:
        from datetime import datetime
        import pytz
        from astral import LocationInfo
        from astral.sun import elevation
        from configs import config
        loc = config.data()["location"]
        li = LocationInfo("obs", "", loc["timezone"], loc["latitude"], loc["longitude"])
        return float(elevation(li.observer, datetime.now(pytz.timezone(loc["timezone"]))))
    except Exception:
        return None


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

    def __init__(self, repo_root: Path, journal: Journal, sun_probe=None,
                 roof_authority=None, mount_authority=None):
        self.root = Path(repo_root)
        self.journal = journal
        self.state = INITIAL_STATE
        self.slots = 0
        # offer() is entered from the watcher thread AND the API thread; the
        # machine steps under one lock so two events can never interleave.
        self._lock = threading.RLock()
        self._state_since = time.time()
        self.roof_authority = (_read_roof_authority() if roof_authority is None
                               else bool(roof_authority))
        self.mount_authority = (_read_mount_authority() if mount_authority is None
                                else bool(mount_authority))
        self._none_streak = 0         # consecutive polls reading NONE (debounce)
        # Slow sensors (network round-trips) polled every N fast polls.
        # Injectable so tests drive them without a mount or a Shelly.
        self.pwi4_probe = _read_pwi4_park
        self.limits_probe = _read_roof_limits
        # Injectable at construction because _recover() consults it, before
        # a caller can reassign the attribute.
        self.sun_probe = sun_probe or _read_sun_altitude
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
        self._ingest_fallback_marker()

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
            self._reseat("recovered mid-night state with no capture activity")
        # The other way round: the shadow was down while the run moved on, so
        # the journal's last transition is behind the capture state on disk.
        # 2026-09-10 a restart mid-run would have recovered to PRELUDE (the
        # last transition, from an attempt that died) with imaging.txt at
        # IN_MAIN, and then waited in PRELUDE for events that had already
        # happened. Walk the two steps the state file proves, tagged.
        if self._imaging == "IN_MAIN" and self._nina:
            if self.state == "PRELUDE":
                self.offer("NINA_PRELUDE_DONE", "shadow",
                           {"imaging": self._imaging, "synthesized": True,
                            "why": "recovered behind the capture state"})
            if self.state == "SLOT_SETUP":
                self.offer("SLOT_STARTED", "shadow",
                           {"imaging": self._imaging, "synthesized": True,
                            "why": "recovered behind the capture state"})
        # A restart in the middle of a MANUAL roof move: the relay fired, the
        # confirmation never arrived here, and nothing will send it now. The
        # roof may be open, closed or stopped halfway; ADR 0007 says that is
        # FAULT_ROOF_UNKNOWN, resolved by an operator who has looked.
        if self.state in ("MANUAL_OPENING", "MANUAL_CLOSING"):
            self.offer("ROOF_TIMEOUT", "conductor",
                       {"why": "conductor restarted mid-move; the roof "
                               "position is untrusted until an operator resolves"})
        _logger.info("shadow recovered: state=%s slots=%d journal head=%d",
                     self.state, self.slots, self.journal.head())

    def _ingest_fallback_marker(self):
        """end.py closed the roof while this conductor was unreachable.

        The marker says whether the close was CONFIRMED by vision. Confirmed:
        journaled loudly, nothing else -- the roof is where the machine
        believes it is. Not confirmed: sense and expectation disagree, which
        is the contradiction FAULT_ROOF_UNKNOWN exists for.
        """
        marker = self.root / FALLBACK_MARKER
        if not marker.exists():
            return
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            info = {"unreadable": f"{type(exc).__name__}: {exc}"}
        self.journal.append("note", "ROOF_MOVED_WITHOUT_CONDUCTOR", "end.py-fallback",
                            data=info)
        if not info.get("confirmed_closed"):
            self.offer("VISION_CONTRADICTION", "end.py-fallback",
                       {"why": "end.py's fallback close was not confirmed while "
                               "the conductor was down"})
        try:
            marker.rename(marker.with_suffix(".ingested-%d.json" % int(time.time())))
        except OSError:
            _logger.exception("could not retire the fallback marker")

    def _live_plan_slots(self):
        """Slots of the night plan still in force per the journal, or None.

        PLAN_GOOD sets it; any transition back to IDLE_DAY or into PLANNING
        (a new day, a cancelled plan, an operator resolve) clears it. A plan
        older than a day is stale whatever the journal says: the scheduler
        was down through a noon and never re-planned.
        """
        from datetime import datetime, timedelta
        slots, when = None, None
        for e in self.journal.replay():
            if e.kind != "transition":
                continue
            if e.event == "PLAN_GOOD":
                slots = max(1, int(e.data.get("slots") or e.data.get("slots_after") or 1))
                when = e.ts
            elif e.to_state in ("IDLE_DAY", "PLANNING"):
                slots, when = None, None
        if slots is None:
            return None
        try:
            planned = datetime.fromisoformat(when)
            if datetime.now(planned.tzinfo) - planned > timedelta(hours=24):
                return None
        except (TypeError, ValueError):
            return None
        return slots

    def _reseat(self, reason, night=None):
        """Abandon the current state (journaled as a SHADOW_RESYNC note).

        Re-seats to ARMED when the night's plan is still in force and it is
        night -- the state a retry of the run starts from, and the one that
        has a CHECKS_PASSED row -- else to IDLE_DAY. Until 2026-09-10 this
        always went to IDLE_DAY, from which a run has no table path, so the
        retry after a lost prelude was ignored for the whole night and the
        site showed idle while the scope imaged. `night` None consults the
        sun; an unknown sun is treated as day (the conservative old answer).
        """
        slots = self._live_plan_slots()
        if night is None:
            alt = self.sun_probe()
            night = alt is not None and alt < 0.0
        to_state, to_slots = INITIAL_STATE, 0
        if slots is not None and night:
            to_state, to_slots = "ARMED", slots
        self.journal.append(
            "note", "SHADOW_RESYNC", "shadow",
            data={"from_state": self.state, "to_state": to_state,
                  "reason": reason})
        self.state, self.slots = to_state, to_slots

    # ------------------------------------------------------------ readers

    def _read_slot_count(self):
        import json
        try:
            d = json.loads((self.root / "scheduler_state.json").read_text())
            return len(d.get("slots") or [])
        except Exception:
            return 0

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
            # The planner's weather verdict is the scheduler's "will image
            # tonight": set True only when the noon / pre-sunset check found
            # enough good hours (visibility x forecast), False when it did not,
            # "Unknown" between days. Until 2026-09-07 nothing set this and the
            # snapshot default (False) made the weather guard refuse every open
            # in the counterfactuals -- the one blemish on the first CLEAN
            # night. Day-scoped, so no freshness decay.
            weather_ok=(self._will_image in ("true", "yes")),
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

    def _close_in_hold(self, source: str, payload: dict) -> Verdict:
        """Closing the roof inside a hold: decided by Invariant A's guards,
        not by the table, and the hold is not left.

        stop! clears safety (SAFE_HOLD) and then parks and closes; a fault
        or an ESTOP leaves the roof wherever it was. The holds ignore every
        non-operator event by design, but closing is the SAFE direction and
        an emergency close refused for bookkeeping is a roof open at dawn.
        So the request is answered by the two guards that matter -- scope
        confirmed parked, roof position known -- on the posted evidence,
        journaled as a note either way, and the machine stays where it is
        until the operator's own event (safe! / resolve!) releases it.
        """
        guards = (G.mount_parked, G.roof_state_known)
        ev = self._current_evidence()
        refusal = G.evaluate(guards, ev)
        payload["in_hold"] = self.state
        if self.roof_authority:
            payload["guards"] = "enforced"
            if refusal:
                e = self.journal.append("rejected", "ROOF_CLOSE_REQUESTED", source,
                                        from_state=self.state, guard=refusal, data=payload)
                return Verdict("rejected", self.state, guard=refusal, seq=e.seq)
            e = self.journal.append("note", "ROOF_CLOSE_ALLOWED_IN_HOLD", source, data=payload)
            return Verdict("allowed_in_hold", self.state, seq=e.seq)
        payload["guard_would"] = refusal
        e = self.journal.append("note", "ROOF_CLOSE_ALLOWED_IN_HOLD", source, data=payload)
        return Verdict("allowed_in_hold", self.state, would_refuse=refusal, seq=e.seq)

    def _park_in_hold(self, source: str, payload: dict) -> Verdict:
        """Parking inside a hold: decided by Invariant B's guard on the
        evidence, not by the table, and the hold is not left.

        stop! parks before it closes, and since 2026-09-25 it runs in ESTOP
        by design. Parking is the safe direction for the scope exactly as
        closing is for the roof, so it is answered like _close_in_hold:
        roof_open on the posted evidence. The one exception is the blind
        park (CLAUDE.md, operator decision 2026-09-17): the scope hides the
        roof, and stop!'s own rule has proved the roof open earlier and
        unmoved since. That request carries blind_park=True and is journaled
        as an assertion, never as a sensor read.
        """
        ev = self._current_evidence()
        refusal = G.evaluate((G.roof_open,), ev)
        if refusal and payload.get("blind_park"):
            refusal = None
            payload["asserted"] = "blind park rule (stop!)"
        payload["in_hold"] = self.state
        if self.mount_authority:
            payload["guards"] = "enforced"
            if refusal:
                e = self.journal.append("rejected", "MOUNT_MOVE_REQUESTED", source,
                                        from_state=self.state, guard=refusal, data=payload)
                return Verdict("rejected", self.state, guard=refusal, seq=e.seq)
            e = self.journal.append("note", "MOUNT_PARK_ALLOWED_IN_HOLD", source, data=payload)
            return Verdict("allowed_in_hold", self.state, seq=e.seq)
        payload["guard_would"] = refusal
        e = self.journal.append("note", "MOUNT_PARK_ALLOWED_IN_HOLD", source, data=payload)
        return Verdict("allowed_in_hold", self.state, would_refuse=refusal, seq=e.seq)

    def _absorb_evidence(self, evidence: dict, payload: dict):
        """A live sensor read posted with a request replaces the log-derived
        one, fresh as of now. Recorded on the entry so the journal shows what
        the decision was made on."""
        e = self.evidence
        if "parked_vision" in evidence:
            e = e.replace(parked_vision=_tri(evidence["parked_vision"]),
                          roof=_tri(evidence.get("roof", e.roof.value)))
            self._evidence_ts = time.time()
        elif "roof" in evidence:
            e = e.replace(roof=_tri(evidence["roof"]))
            self._evidence_ts = time.time()
        if "parked_kasa" in evidence:
            e = e.replace(parked_kasa=_tri(evidence["parked_kasa"]))
            self._kasa_ts = time.time()
        self.evidence = e
        payload["evidence_posted"] = {k: v for k, v in evidence.items() if k != "ts"}

    def offer(self, event: str, source: str, data: dict = None,
              evidence: dict = None) -> Verdict:
        """Offer one event; journal everything.

        Shadow-synthesized events step with the permissive snapshot and carry
        the counterfactual guard verdict (`guard_would`). A LIVE roof decision
        (posted by an actuator with its evidence) steps with the REAL evidence
        when roof authority is on -- the guards decide, and a refusal is
        journaled as `rejected` with the reason -- and permissively otherwise,
        with the verdict returned as `would_refuse` for the caller to post.
        """
        with self._lock:
            payload = dict(data or {})
            if evidence:
                self._absorb_evidence(evidence, payload)
            live = source != "shadow"
            if live and event == "ROOF_CLOSE_REQUESTED" and self.state in HOLD_STATES:
                return self._close_in_hold(source, payload)
            if (live and event == "MOUNT_MOVE_REQUESTED" and self.state in HOLD_STATES
                    and str(payload.get("direction", "")).lower() == "park"):
                return self._park_in_hold(source, payload)
            enforced = live and (
                (self.roof_authority and event in ROOF_DECISION_EVENTS)
                or (self.mount_authority and event in MOUNT_DECISION_EVENTS))
            # An operator starting a run is the weather verdict for that run:
            # the planner's "will image tonight" describes the planner's plan,
            # not the operator's decision to image anyway.
            operator_weather = (event == "CHECKS_PASSED" and source == "operator")
            # An unplanned run (image!! with nothing planned) opens from
            # IDLE_DAY; size its slot count from the sequence on disk so the
            # prelude does not walk straight to FLATS on "plan exhausted".
            if event == "CHECKS_PASSED" and self.state == "IDLE_DAY":
                self.slots = max(1, self._read_slot_count())
                payload["unplanned"] = True
                self._unplanned = True
            # A failed fire returns to "the day the open was attempted from";
            # for an unplanned run that day had no plan, and the slot count
            # synthesized above must not read as one (ARMED) on the way back.
            if event == "ROOF_FIRE_FAILED" and getattr(self, "_unplanned", False):
                self.slots = 0
            evidence_snap = self._current_evidence()
            if operator_weather:
                evidence_snap = evidence_snap.replace(weather_ok=True)
                payload["weather"] = "operator's call"
            if enforced:
                snap = evidence_snap
                payload["guards"] = "enforced"
            else:
                snap = _PERMISSIVE.replace(slots_remaining=self.slots)
            would = None
            if event in MOUNT_DECISION_EVENTS and not enforced:
                # A permission row wants the roof at the position its state
                # implies (open for a move, shut for flats power), so no one
                # permissive roof value fits every row. Step on the evidence;
                # if the guards refuse, step again on the permissive snapshot
                # that fits and carry the refusal as the decision-diff.
                out = step(self.state, event, evidence_snap)
                if out.kind == "rejected":
                    for roof in (Tri.CONFIRMED, Tri.DENIED):
                        alt = step(self.state, event,
                                   _PERMISSIVE.replace(slots_remaining=self.slots, roof=roof))
                        if alt.kind == "allowed":
                            would, out = out.guard, alt
                            break
            else:
                out = step(self.state, event, snap)
            if out.kind == "allowed":
                # A permission row: the mount may act, the night stays put.
                if not enforced:
                    payload["guard_would"] = would
                payload["allowed_in_state"] = self.state
                e = self.journal.append("note", event.replace("_REQUESTED", "_ALLOWED"),
                                        source, data=payload)
                return Verdict("allowed", self.state, would_refuse=would, seq=e.seq)
            if out.kind == "transition":
                if not enforced:
                    row = self._fired_row(self.state, event, snap)
                    if row is not None and row.guards:
                        would = G.evaluate(row.guards, evidence_snap)
                        payload["guard_would"] = would      # None == would have passed
                payload["slots_after"] = self.slots
                e = self.journal.append("transition", event, source,
                                        from_state=self.state, to_state=out.state,
                                        data=payload)
                self.state = out.state
                self._state_since = time.time()
                if self.state in ("IDLE_DAY", "ARMED"):
                    self._unplanned = False
                return Verdict("transition", self.state, would_refuse=would, seq=e.seq)
            if out.kind == "rejected":
                e = self.journal.append("rejected", event, source,
                                        from_state=self.state, guard=out.guard,
                                        data=payload)
                return Verdict("rejected", self.state, guard=out.guard, seq=e.seq)
            # Ignored events are journaled as notes: they are exactly the
            # mismatches between reality and the table that Phase 3 needs to
            # know about. For a live request they are a refusal: no row means
            # the machine does not move the roof from here.
            payload["ignored_in_state"] = self.state
            e = self.journal.append("note", event, source, data=payload)
            return Verdict("ignored", self.state,
                           guard="no transition for %s in state %s" % (event, self.state),
                           seq=e.seq)

    # ------------------------------------------------------------ polling

    def poll(self):
        """One observation pass. Called every few seconds by the runner."""
        with self._lock:
            self._poll_locked()

    def _poll_locked(self):
        lines = self._read_new_log_lines()
        self._update_evidence_from_log(lines)
        self._poll_slow_sensors()

        # --- a roof move that never confirmed. The actuator posts the
        # confirmation (or its own timeout) when its confirm loop ends; if it
        # died first, nothing will. Past the loop's worst case the position
        # is untrusted: FAULT_ROOF_UNKNOWN, until an operator has looked.
        if (self.state in ROOF_MOVING_STATES
                and time.time() - self._state_since > ROOF_MOTION_TIMEOUT_S):
            self.offer("ROOF_TIMEOUT", "watchdog",
                       {"why": "no confirmation within %d s of the relay fire"
                               % ROOF_MOTION_TIMEOUT_S})

        # --- sunrise ends a finished night. NIGHT_DONE's only exit is
        # DAY_TICK, and the scheduler's version of that (IMAGING ->
        # WAITING_FOR_NOON) fires at the wrong moment in manual mode: 19:06 on
        # 2026-09-06, BEFORE the night, where it was rightly suppressed. The
        # real night ended 03:42 with nothing left for the scheduler to say,
        # so the machine sat in NIGHT_DONE until the noon resync flagged a
        # clean night as a divergence. The sun is the honest day clock.
        if self.state == "NIGHT_DONE":
            alt = self.sun_probe()
            if alt is not None and alt > 0.0:
                self.offer("DAY_TICK", "shadow", {"clock": "sunrise",
                                                  "sun_alt": round(alt, 1)})

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
                # One slot unless the plan says otherwise (scheduler_state
                # "slots", written by the noon check since 2026-09-07).
                self.slots = max(1, self._read_slot_count())
                self.offer("PLAN_GOOD", "shadow", {"sched": sched, "dso": will,
                                                   "slots": self.slots})
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
                # A prelude beginning while the machine still holds a night
                # in progress is a RETRY: the previous attempt died (NINA
                # quit in the prelude 2026-09-10, twice) and the operator
                # closed up and ran image!! again. The table has no row for
                # that from any mid-night state, so re-arm first: the plan
                # is unchanged and the roof was shut for the retry to open.
                # ...unless the machine reached PRELUDE moments ago on the
                # run's OWN posts (roof authority: CHECKS_PASSED then
                # ROOF_OPEN_CONFIRMED arrive before NINA writes IN_PRELUDE).
                # That is the same run, not a retry.
                fresh_prelude = (self.state == "PRELUDE"
                                 and time.time() - self._state_since < 15 * 60)
                if self.state in self._MID_NIGHT and not fresh_prelude:
                    self._reseat("prelude restarted while the machine was "
                                 "mid-night: the previous attempt was lost",
                                 night=True)
                # The legacy run opens the roof between the checks and the
                # prelude; the shadow sees only the prelude begin, so the two
                # machine steps are synthesized back to back. Their true
                # relative timing is in iris.log for the report to compare.
                # Each is offered only where the machine still stands before
                # it: with roof authority the run has already posted them
                # itself, and a second copy would only journal as ignored.
                # IDLE_DAY is included since 2026-09-16: an unplanned manual
                # run is tracked as a night, not journaled as noise.
                if self.state in ("ARMED", "PRE_FLIGHT", "IDLE_DAY"):
                    self.offer("CHECKS_PASSED", "shadow", {"imaging": img})
                if self.state == "OPENING_ROOF":
                    self.offer("ROOF_OPEN_CONFIRMED", "shadow", {"imaging": img})
            elif img == "DONE_PRELUDE":
                self.offer("NINA_PRELUDE_DONE", "shadow", {"imaging": img})
            elif img == "IN_MAIN":
                self.offer("SLOT_STARTED", "shadow", {"imaging": img})
            elif img == "DONE_MAIN":
                # The hand-over between two slots: the two-target sequence
                # writes DONE_MAIN as the first target's block ends and IN_MAIN
                # as the second begins imaging (nina_sequence_gen.
                # generate_slots_sequence). One slot consumed; with one left
                # the table walks SLOT_IMAGING -> SLOT_SETUP, and the IN_MAIN
                # that follows is SLOT_STARTED again.
                self.slots = max(0, self.slots - 1)
                self.offer("NINA_SLOT_DONE", "shadow", {"imaging": img})
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
                # Each step only where the machine still stands before it:
                # end.py posts the park and close itself under roof authority.
                if self.state == "SLOT_IMAGING":
                    self.offer("NINA_SLOT_DONE", "shadow", {"imaging": img})
                if self.state == "PARKING":
                    self.offer("MOUNT_PARK_CONFIRMED", "shadow", {"imaging": img})
                if self.state == "CLOSING_ROOF":
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
        # Only while the MACHINE still has a capture running. With roof
        # authority end.py's posts walk the machine to FLATS before the state
        # file catches up, and the deliberate kill of NINA before the flats
        # (2026-09-16 11:43) then read as a capture lost from FLATS.
        if (self._nina and not nina and self._imaging in ("IN_PRELUDE", "IN_MAIN")
                and self.state in ("PRELUDE", "SLOT_SETUP", "SLOT_IMAGING")):
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
