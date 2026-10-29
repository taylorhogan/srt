"""alerts.py -- the one place the conductor tells the owner it cannot continue.

docs/AUTOMATION_PLAN.md B (2026-10-29). Until this module, entering a hold
(SAFE_HOLD, FAULT_ROOF_UNKNOWN, ESTOP) was journaled and nothing more: the
actuator that hit the problem pushed what it knew, the conductor said
nothing, and every following noon was recorded SKIPPED in silence while the
scheduler posted one chat line and noon_watch counted the day a success. A
held observatory could stay down for a week without a phone ringing.

Rules (owner, 2026-10-29: "you, escalating"):

  * a transition INTO a hold from outside one, or one of the BLOCKING events
    (CHECKS_FAILED, PLAN_FAILED, CAPTURE_LOST, a PLAN_BAD whose reason is the
    disk) -> ONE Pushover emergency (priority 2: repeats on the phone until
    acknowledged) with the reason and the picture that decided it (the newest
    north-camera frame of the day) when there is one;
  * while the machine stays in a hold, ONE priority-1 reminder a day at the
    noon tick (NightDriver calls noon_reminder from its SKIPPED branch);
  * nothing else. A hold entered from another hold (SAFE_HOLD -> ESTOP) is
    the same emergency and does not push again; a conductor restarted inside
    a hold does not re-alert, because the (state, day) key is kept on disk.

Pushes run on their own daemon thread: Pushover's emergency path is three
tries with back-off, and offer() holds the machine's lock. The journal is
the record; this is the bell.
"""
import json
import logging
import os
import queue
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

from iris.core.machine import HOLD_STATES

_logger = logging.getLogger(__name__)

BLOCKING_EVENTS = frozenset({"CHECKS_FAILED", "PLAN_FAILED", "CAPTURE_LOST"})
HOLD_PRIORITY = 2
REMINDER_PRIORITY = 1
STATE_FILE = "local/alerts_state.json"
NORTH_ARCHIVE = "local/north_archive"

RESOLVE_HINT = {
    "ESTOP": "stop! is in force: look, then `resolve!` releases it",
    "FAULT_ROOF_UNKNOWN": "the roof's position is unknown: look (status / roof!!), "
                          "close it by hand if open, then `resolve!`",
    "SAFE_HOLD": "the observatory is marked UNSAFE: `safe!` re-arms it",
}


def _default_push(message: str, image: Optional[str], priority: int) -> bool:
    from utils import pushover
    return bool(pushover.push_message(message, image=image, priority=priority))


def latest_north_frame(root, day: Optional[date] = None) -> Optional[str]:
    """The newest archived north-camera frame of *day*, or None.

    sentry/north_roof archives every frame it decides on under
    local/north_archive/<YYYY-MM-DD>/<HHMMSS>_<i>_<verdict>[_ir].jpg; the
    newest is the picture the operator would ask for first.
    """
    d = Path(str(root)) / NORTH_ARCHIVE / (day or date.today()).isoformat()
    try:
        frames = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
    except OSError:
        return None
    return str(frames[-1]) if frames else None


def hold_message(event: str, to_state: str, data: dict, since: Optional[datetime] = None) -> str:
    why = (data or {}).get("why") or (data or {}).get("guard") or (data or {}).get("reason")
    lines = ["Iris HOLD: %s after %s%s." % (to_state, event, " (%s)" % why if why else "")]
    if since is not None:
        lines[0] = lines[0][:-1] + " at %s." % since.strftime("%H:%M")
    lines.append("No night runs until this is resolved. %s."
                 % RESOLVE_HINT.get(to_state, "`resolve!` once it is safe"))
    return "\n".join(lines)


def blocking_message(event: str, data: dict) -> str:
    why = (data or {}).get("why") or (data or {}).get("error") or (data or {}).get("check")
    what = {"CHECKS_FAILED": "the pre-flight checks failed; no roof open tonight",
            "PLAN_FAILED": "the night plan failed twice; no imaging tonight",
            "CAPTURE_LOST": "N.I.N.A disappeared mid-run; the night is ending",
            "PLAN_BAD": "the night is called off"}.get(event, event)
    return "Iris: %s%s. Check the observatory." % (what, " (%s)" % why if why else "")


def reminder_message(state: str, since: Optional[datetime], disk_line: str = "") -> str:
    when = (" since %s" % since.strftime("%a %d %b %H:%M")) if since else ""
    lines = ["Iris is still in %s%s: no plan was made at noon." % (state, when),
             RESOLVE_HINT.get(state, "`resolve!` once it is safe") + "."]
    if disk_line:
        lines.append(disk_line)
    return "\n".join(lines)


class Alerter:
    """One instance per conductor; everything injectable so tests run dry.

    push(message, image, priority) -> bool; picture_fn(day) -> path or None;
    now_fn() -> datetime. worker=False runs pushes inline (tests).
    """

    def __init__(self, root, push: Callable = None, picture_fn: Callable = None,
                 now_fn: Callable[[], datetime] = None, state_path=None,
                 enabled: bool = True, hold_priority: int = HOLD_PRIORITY,
                 worker: bool = True):
        self.root = Path(str(root))
        self.push = push or _default_push
        self.picture_fn = picture_fn or (lambda day: latest_north_frame(self.root, day))
        self.now_fn = now_fn or datetime.now
        self.state_path = Path(state_path) if state_path else self.root / STATE_FILE
        self.enabled = bool(enabled)
        self.hold_priority = int(hold_priority)
        self._sent = self._load()
        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue()
        self._worker = None
        if worker:
            self._worker = threading.Thread(target=self._drain, daemon=True, name="alerts")
            self._worker.start()

    @classmethod
    def from_config(cls, cfg: dict, root) -> "Alerter":
        a = (cfg or {}).get("alerts") or {}
        return cls(root, enabled=a.get("enabled", True),
                   hold_priority=a.get("hold_priority", HOLD_PRIORITY))

    # ------------------------------------------------------------ the rules

    def on_transition(self, event: str, from_state: str, to_state: str, data: dict = None):
        """Called by the conductor after every journaled transition. Never raises."""
        try:
            data = dict(data or {})
            if to_state in HOLD_STATES and from_state not in HOLD_STATES:
                self._once("hold:%s" % to_state,
                           hold_message(event, to_state, data, self.now_fn()),
                           self.hold_priority, picture=True)
            elif event in BLOCKING_EVENTS:
                self._once("event:%s" % event, blocking_message(event, data),
                           self.hold_priority, picture=False)
            elif event == "PLAN_BAD" and str(data.get("why") or "").startswith("disk"):
                self._once("event:PLAN_BAD:disk", blocking_message(event, data),
                           self.hold_priority, picture=False)
        except Exception:  # noqa: BLE001
            _logger.exception("alerts: on_transition failed for %s", event)

    def noon_reminder(self, state: str, since: Optional[datetime] = None, disk_line: str = ""):
        """One priority-1 line a day while the machine is held. Never raises."""
        try:
            if state in HOLD_STATES:
                self._once("reminder:%s" % state, reminder_message(state, since, disk_line),
                           REMINDER_PRIORITY, picture=False)
        except Exception:  # noqa: BLE001
            _logger.exception("alerts: noon_reminder failed")

    # ------------------------------------------------------------ plumbing

    def _once(self, key: str, message: str, priority: int, picture: bool):
        if not self.enabled:
            return
        day = self.now_fn().date().isoformat()
        with self._lock:
            if self._sent.get(key) == day:
                _logger.info("alerts: %s already sent today", key)
                return
            self._sent[key] = day
            self._save()
        image = None
        if picture:
            try:
                image = self.picture_fn(self.now_fn().date())
                if image and not os.path.exists(image):
                    image = None
            except Exception:  # noqa: BLE001
                image = None
        job = (message, image, priority)
        if self._worker is None:
            self._send(*job)
        else:
            self._q.put(job)

    def _send(self, message, image, priority):
        try:
            ok = self.push(message, image, priority)
            _logger.log(logging.INFO if ok else logging.ERROR,
                        "alerts: priority %d %s: %s", priority,
                        "sent" if ok else "NOT ACCEPTED", message.splitlines()[0])
        except Exception:  # noqa: BLE001
            _logger.exception("alerts: push failed")

    def _drain(self):
        while True:
            job = self._q.get()
            self._send(*job)
            self._q.task_done()

    def flush(self, timeout: float = 5.0):
        """Wait for queued pushes (tests, shutdown)."""
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < timeout:
            time.sleep(0.01)

    def _load(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._sent, indent=1), encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception:  # noqa: BLE001
            _logger.warning("alerts: could not save %s", self.state_path, exc_info=True)
