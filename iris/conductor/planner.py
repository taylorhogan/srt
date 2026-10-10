"""The conductor plans the night in shadow (Phase 3, step 1, 2026-10-10).

Phase 3 of docs/ARCHITECTURE_PLAN.md moves the noon and pre-sunset checks
from the scheduler into the conductor. Before the conductor may decide a
night it has to show it decides the same night the scheduler does, so this
step only WATCHES: each time the scheduler leaves NOON_CHECK or
PRE_SUNSET_CHECK, the decision it wrote to scheduler_state.json (target,
"will image tonight", slots) is set beside the conductor's own plan of the
same night, computed by the same code (control/night_plan), and the
comparison is journaled:

    PLAN_MATCH          the two agree on whether, what and when
    PLAN_DIFF           they do not; "diffs" says where
    SHADOW_PLAN_FAILED  the conductor could not plan (forecast down, crash)
    SHADOW_PLAN_SKIPPED the previous plan was still running

The plan runs in its OWN PROCESS (python -m control.night_plan) from a worker
thread, with a timeout. The planner hits the network (forecasts, SIMBAD) and
imports half the astronomy stack; none of that may be able to stall or crash
the process that answers roof and mount requests. The worker never takes the
machine's lock and never steps the machine: it only appends notes to the
journal, whose append is itself locked.

Triggered on the scheduler's decision rather than on the conductor's own
clock, so both read the same queue (the noon check rewrites its
hours-above-horizon first) and the same forecast hour; what is compared is
the decision, not two clocks. The conductor's own clock comes with the
switch-over (step 3).
"""
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from control import night_plan

_logger = logging.getLogger(__name__)

CHECKS = ("NOON_CHECK", "PRE_SUNSET_CHECK")
PLAN_TIMEOUT_S = 600         # a SIMBAD outage once made a ranking crawl; 24 s is normal


def parse_plan(stdout: str) -> dict:
    """The plan from the CLI's stdout: its last line that is a JSON object.

    The ranking prints progress (cloud cover per hour, NWS notes) before the
    plan; only the final JSON line is the answer."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and "will_image" in d:
                return d
    raise ValueError("no plan in the planner's output")


class ShadowPlanner:
    def __init__(self, repo_root: Path, journal, runner=None,
                 timeout_s: float = PLAN_TIMEOUT_S):
        self.root = Path(repo_root)
        self.journal = journal
        self.timeout_s = timeout_s
        # Injectable: tests plan without a forecast, a queue or a subprocess.
        self.runner = runner or self._run_cli
        self._busy = threading.Lock()
        self._thread = None

    def on_decision(self, check: str, scheduler: dict):
        """The scheduler just left *check* with *scheduler* on disk.

        Plans the same night in a worker thread and journals the comparison.
        Returns the thread (None when skipped). Never raises."""
        if not self._busy.acquire(blocking=False):
            self.journal.append("note", "SHADOW_PLAN_SKIPPED", "planner",
                                data={"check": check, "why": "previous plan still running"})
            return None
        try:
            t = threading.Thread(target=self._plan_and_compare,
                                 args=(check, dict(scheduler or {})),
                                 daemon=True, name="shadow-planner")
            t.start()
        except Exception:
            self._busy.release()
            _logger.exception("shadow planner could not start")
            return None
        self._thread = t
        return t

    def _plan_and_compare(self, check: str, scheduler: dict):
        t0 = time.time()
        try:
            try:
                plan = self.runner()
            except Exception as exc:
                self.journal.append("note", "SHADOW_PLAN_FAILED", "planner",
                                    data={"check": check, "error": str(exc)[-300:],
                                          "took_s": round(time.time() - t0, 1)})
                return
            diffs = night_plan.compare(plan, scheduler)
            self.journal.append(
                "note", "PLAN_DIFF" if diffs else "PLAN_MATCH", "planner",
                data={"check": check, "diffs": diffs,
                      "conductor": plan,
                      "scheduler": {"dso": scheduler.get("dso"),
                                    "will_image": scheduler.get("will image tonight"),
                                    "slots": scheduler.get("slots") or []},
                      "took_s": round(time.time() - t0, 1)})
        except Exception:
            _logger.exception("shadow planner failed")
        finally:
            self._busy.release()

    def _run_cli(self) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "control.night_plan"], cwd=str(self.root),
            capture_output=True, text=True, timeout=self.timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0:
            raise RuntimeError("night_plan exited %d: %s"
                               % (proc.returncode, (proc.stderr or "").strip()[-300:]))
        return parse_plan(proc.stdout)
