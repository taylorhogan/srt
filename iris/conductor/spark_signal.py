"""The night-end signal to the Spark (2026-11-21).

The PC knows when the night is over: the conductor journals NINA_FLATS_DONE
the moment N.I.N.A's flats finish. Until today the Spark guessed at it
(scripts/morning_trigger: the plan's last slot end plus 85 min, a journal
read every 15 min that was blind to the morning's file until it existed and
then took the previous morning's record for tonight's; on 2026-11-21 the run
fired four minutes before the first flat). Now the PC tells it: on the end
event the conductor runs one ssh command on the Spark that starts the morning
run (the flats poll, the sync, the sentry sync, the render). The command runs
on its own thread, never under the conductor's lock; the outcome is a journal
note (SPARK_SIGNALLED / SPARK_SIGNAL_FAILED) and a log line, nothing more.
The Spark's own 20:00 morning_trigger stays as the fallback and runs
unsignalled when nothing arrives; a run already on record is not run twice.

cfg["spark_signal"]: host, user, key (the PC's own key, authorized on the
Spark: docs/SUCCESSION.md), enabled, and optionally command ({event} is
replaced). An incomplete block is a silent off.
"""
import logging
import subprocess
import threading
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

EVENTS = ("NINA_FLATS_DONE",)
TIMEOUT_S = 90
DEFAULT_COMMAND = ("nohup setsid /home/taylor/Documents/srt/scripts/morning_trigger.py "
                   "--signal {event} >> /tmp/morning_trigger.signal.log 2>&1 < /dev/null &")


def command(block: Optional[dict], event: str) -> list:
    """The ssh argv that starts the Spark's morning run for *event*; [] when
    the block lacks host, user or key. Pure."""
    b = block or {}
    host, user, key = b.get("host"), b.get("user"), b.get("key")
    if not (host and user and key):
        return []
    remote = str(b.get("command") or DEFAULT_COMMAND).replace("{event}", event)
    return ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=accept-new", "%s@%s" % (user, host), remote]


def _run(argv: list) -> tuple:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
    return r.returncode, (r.stdout + r.stderr).strip()


class SparkSignal:
    """Told after every journaled transition, like the Alerter; acts on EVENTS.

    runner(argv) -> (rc, text); note(event, data) journals the outcome;
    worker=False sends inline (tests).
    """

    def __init__(self, block: Optional[dict], runner: Callable = None, note: Callable = None,
                 events=EVENTS, worker: bool = True):
        self.block = dict(block or {})
        self.enabled = bool(self.block.get("enabled", True)) and bool(command(self.block, "x"))
        self.runner = runner or _run
        self.note = note
        self.events = tuple(events)
        self.worker = worker

    @classmethod
    def from_config(cls, cfg: dict, note: Callable = None) -> "SparkSignal":
        return cls((cfg or {}).get("spark_signal") or {}, note=note)

    def wants(self, event: str) -> bool:
        return self.enabled and event in self.events

    def on_transition(self, event: str, from_state: str, to_state: str, data: dict = None) -> bool:
        """True when a signal was (or is being) sent. Never raises."""
        try:
            if not self.wants(event):
                return False
            if self.worker:
                threading.Thread(target=self._send, args=(event,), daemon=True,
                                 name="spark-signal").start()
            else:
                self._send(event)
            return True
        except Exception:  # noqa: BLE001
            _logger.exception("spark signal: on_transition failed for %s", event)
            return False

    def _send(self, event: str) -> None:
        argv = command(self.block, event)
        try:
            rc, text = self.runner(argv)
        except Exception as exc:  # noqa: BLE001
            rc, text = -1, "%s: %s" % (type(exc).__name__, exc)
        ok = rc == 0
        (_logger.info if ok else _logger.warning)(
            "spark signal %s: %s", event, "sent" if ok else "FAILED rc=%s %s" % (rc, text[:200]))
        if self.note is not None:
            try:
                self.note("SPARK_SIGNALLED" if ok else "SPARK_SIGNAL_FAILED",
                          {"event": event, "rc": rc, "text": text[:300], "host": self.block.get("host")})
            except Exception:  # noqa: BLE001
                _logger.exception("spark signal: note failed")
