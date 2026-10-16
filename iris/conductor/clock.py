"""The conductor plans the night on its own clock (Phase 3 step 2, ADR 0015).

Behind `conductor.drives_checks`. Off, nothing here runs and the scheduler
plans as it always has (with the shadow planner beside it). On:

  12:00              NOON_TICK          IDLE_DAY -> PLANNING, then the noon plan
  sunset - 10 min    PRE_SUNSET_TICK    ARMED -> PRE_FLIGHT, then the sunset plan

Each plan runs `python -m control.night_plan` (the noon form with
--refresh) in its own process from a worker thread, with the planner's
timeout, exactly as the shadow planner did for six clean nights. Never under
the machine's lock: the network and the astronomy stack cannot stall the
process that answers roof and mount requests.

THE RECORD. Every decision is a journal note, NIGHT_PLAN, carrying {date,
check: noon|sunset, outcome, plan, why, trigger, attempt}; the latest per
check per date is the night's plan, served by GET /v1/plan. The machine
events (PLAN_GOOD, PLAN_BAD, PLAN_FAILED, WEATHER_BAD) are offered beside it
and follow it where the table has a row; the record does not depend on
them. The scheduler carries out the record and never plans itself.

  good     the night is on (noon) / confirmed, launch (sunset)
  bad      not enough good hours
  failed   the planner crashed or timed out twice, 5 minutes apart (owner,
           2026-10-16); pushed from here, since the chat may be what hung
  skipped  the machine is in a hold: no plan until an operator resolves
  missed   the conductor was not running while the check was due

IDEMPOTENT BY DATE. What is due comes from the journal, not from a timer:
after a restart the same check never runs twice and one still due runs
once (due()). 12:00 to the sunset check with no noon record plans now, the
job the scheduler's _initial_state() did.
"""
import logging
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from iris.core.machine import HOLD_STATES, INITIAL_STATE, MANUAL_STATES

_logger = logging.getLogger(__name__)

NIGHT_PLAN = "NIGHT_PLAN"
GOOD, BAD, FAILED, SKIPPED, MISSED = "good", "bad", "failed", "skipped", "missed"
NOON_HOUR = 12
SUNSET_LEAD = timedelta(minutes=10)      # the scheduler's pre-sunset check, unchanged
SUNSET_GRACE = timedelta(hours=2)        # a sunset check later than this is missed
RETRY_AFTER_S = 300                      # owner, 2026-10-16: retry once after 5 min
MAX_ATTEMPTS = 2
LOOP_S = 15.0


# ---------------------------------------------------------------- pure parts

def plan_records(entries, day: date) -> dict:
    """{"noon": record or None, "sunset": record or None} for *day*. Pure.

    The latest NIGHT_PLAN note per check (a `replan` supersedes the noon
    one), its data plus the entry's seq and ts."""
    out = {"noon": None, "sunset": None}
    key = day.isoformat()
    for e in entries:
        if e.kind != "note" or e.event != NIGHT_PLAN:
            continue
        d = e.data or {}
        if d.get("date") != key or d.get("check") not in out:
            continue
        rec = dict(d)
        rec["seq"], rec["ts"] = e.seq, e.ts
        out[d["check"]] = rec
    return out


def due(now: datetime, sunset: datetime, records: dict) -> Optional[str]:
    """Which check the clock owes at *now*. Pure.

    "noon" | "sunset" | "missed_noon" | "missed_sunset" | None. *now* and
    *sunset* are tz-aware and on the same local date as *records*."""
    pre = sunset - SUNSET_LEAD
    noon = now.replace(hour=NOON_HOUR, minute=0, second=0, microsecond=0)
    n, s = records.get("noon"), records.get("sunset")
    if s is not None:
        return None
    if n is None:
        if now < noon:
            return None
        return "noon" if now < pre else "missed_noon"
    if n.get("outcome") != GOOD or now < pre:
        return None
    return "sunset" if now < sunset + SUNSET_GRACE else "missed_sunset"


def local_sunset(day: date) -> datetime:
    """Sunset at the observatory on local *day*, tz-aware. Same astral call
    as iris_astronomy.weather.get_sunrise_sunset, without its imports."""
    from astral import LocationInfo
    from astral.sun import sun
    from configs import config
    loc = config.data()["location"]
    li = LocationInfo("obs", "", loc["timezone"], loc["latitude"], loc["longitude"])
    return sun(li.observer, date=day, tzinfo=li.timezone)["sunset"]


def local_now() -> datetime:
    from zoneinfo import ZoneInfo
    from configs import config
    return datetime.now(ZoneInfo(config.data()["location"]["timezone"]))


def _default_push(msg: str) -> bool:
    try:
        from utils import pushover
        return bool(pushover.push_message(msg, priority=1))
    except Exception:
        _logger.exception("night planner: Pushover failed")
        return False


# ---------------------------------------------------------------- the driver

class NightDriver:
    """Owns the noon and sunset checks for one conductor.

    *conductor* is the ShadowConductor (its .state, .slots, ._lock and
    .offer); *runner(refresh)* returns a plan dict or raises. Clock, sunset,
    push and retry delay are injectable so tests drive a whole day in
    milliseconds."""

    def __init__(self, conductor, journal, repo_root: Path,
                 runner: Optional[Callable[[bool], dict]] = None,
                 now_fn: Callable[[], datetime] = None,
                 sunset_fn: Callable[[date], datetime] = None,
                 push: Callable[[str], bool] = None,
                 retry_after_s: float = RETRY_AFTER_S):
        self.c = conductor
        self.journal = journal
        self.root = Path(repo_root)
        self.runner = runner or self._run_cli
        self.now_fn = now_fn or local_now
        self.sunset_fn = sunset_fn or local_sunset
        self.push = push or _default_push
        self.retry_after_s = retry_after_s
        self._busy = threading.Lock()
        self._worker = None
        self._retry_at = 0.0
        self._attempts = {}
        self._sunsets = {}

    # ------------------------------------------------------------ loop

    def run_forever(self):
        while True:
            try:
                self.tick()
            except Exception:
                _logger.exception("night planner tick failed (continuing)")
            time.sleep(LOOP_S)

    def records(self, day: Optional[date] = None) -> dict:
        day = day or self.now_fn().date()
        return plan_records(self.journal.replay(days=2), day)

    def sunset(self, day: date) -> datetime:
        if day not in self._sunsets:
            self._sunsets = {day: self.sunset_fn(day)}
        return self._sunsets[day]

    def tick(self):
        """One pass: start whatever the clock owes, if nothing is running."""
        if self._busy.locked() or time.time() < self._retry_at:
            return None
        now = self.now_fn()
        day = now.date()
        recs = self.records(day)
        # A replan (REPLAN_REQUESTED moved the machine to PLANNING), or a
        # restart that recovered into PLANNING: plan the noon decision now.
        sunset = self.sunset(day)
        if self.c.state == "PLANNING":
            noon = now.replace(hour=NOON_HOUR, minute=0, second=0, microsecond=0)
            if recs["sunset"] is not None or not (noon <= now < sunset - SUNSET_LEAD):
                # After the sunset check has decided, or outside the day's
                # planning window (a PLANNING recovered from yesterday).
                self._refuse_replan(day, "the sunset check has already decided tonight"
                                    if recs["sunset"] is not None
                                    else "outside the planning window (noon to the sunset check)")
                return None
            trigger = "replan" if recs["noon"] is not None else "recovery"
            return self._start("noon", day, trigger)
        what = due(now, sunset, recs)
        if what == "noon":
            return self._tick_noon(day)
        if what == "sunset":
            if self.c.state == "ARMED":
                self.c.offer("PRE_SUNSET_TICK", "clock", {"clock": "sunset - 10 min"})
            return self._start("sunset", day, "clock")
        if what in ("missed_noon", "missed_sunset"):
            check = what.split("_")[1]
            self._record(day, check, MISSED, trigger="clock",
                         why="the conductor was not running while the %s check was due" % check)
        return None

    # ------------------------------------------------------------ checks

    def _tick_noon(self, day: date):
        state = self.c.state
        if state in HOLD_STATES:
            self._record(day, "noon", SKIPPED, trigger="clock",
                         why="the machine is in %s; no plan until an operator resolves" % state)
            return None
        if (state not in ("IDLE_DAY", "PLANNING") and state not in MANUAL_STATES
                and not self._capture_active()):
            # The previous night's end was never seen (NIGHT_DONE waits for
            # sunrise; a lost close leaves a mid-night state). Same re-seat
            # the shadow made at the scheduler's noon, journaled.
            with self.c._lock:
                self.journal.append("note", "SHADOW_RESYNC", "clock",
                                    data={"from_state": self.c.state,
                                          "reason": "noon tick while machine was mid-night"})
                self.c.state, self.c.slots = INITIAL_STATE, 0
                self.c._state_since = time.time()
        if self.c.state == "IDLE_DAY":
            self.c.offer("NOON_TICK", "clock", {"clock": "noon"})
        # Planned whether or not the machine could take the tick (a daytime
        # run still on the hardware at 12:00, a roof opened by hand): the
        # record is the plan.
        return self._start("noon", day, "clock")

    def _refuse_replan(self, day: date, why: str):
        self.journal.append("note", "REPLAN_REFUSED", "clock",
                            data={"date": day.isoformat(), "why": why})
        self.c.offer("PLAN_BAD", "clock", {"why": why})

    def _capture_active(self) -> bool:
        try:
            return (getattr(self.c, "_imaging", None) not in (None, "NONE")
                    or bool(getattr(self.c, "_nina", False)))
        except Exception:
            return False

    # ------------------------------------------------------------ planning

    def _start(self, check: str, day: date, trigger: str):
        if not self._busy.acquire(blocking=False):
            return None
        try:
            t = threading.Thread(target=self._work, args=(check, day, trigger),
                                 daemon=True, name="night-planner")
            t.start()
        except Exception:
            self._busy.release()
            _logger.exception("night planner could not start")
            return None
        self._worker = t
        return t

    def _work(self, check: str, day: date, trigger: str):
        key = (day.isoformat(), check)
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt
        t0 = time.time()
        try:
            try:
                plan = self.runner(check == "noon")
            except Exception as exc:
                self._failed(check, day, trigger, attempt, exc, time.time() - t0)
                return
            self._attempts.pop(key, None)
            outcome = GOOD if plan.get("will_image") else BAD
            self._record(day, check, outcome, plan=plan, trigger=trigger,
                         attempt=attempt, took_s=time.time() - t0)
            self._advance(check, outcome, plan)
        except Exception:
            _logger.exception("night planner worker failed")
        finally:
            self._busy.release()

    def _failed(self, check, day, trigger, attempt, exc, took):
        err = "%s: %s" % (type(exc).__name__, str(exc)[-300:])
        if attempt < MAX_ATTEMPTS:
            self._retry_at = time.time() + self.retry_after_s
            self.journal.append("note", "NIGHT_PLAN_RETRY", "clock",
                                data={"date": day.isoformat(), "check": check,
                                      "attempt": attempt, "error": err,
                                      "retry_in_s": self.retry_after_s,
                                      "took_s": round(took, 1)})
            return
        self._attempts.pop((day.isoformat(), check), None)
        self._record(day, check, FAILED, trigger=trigger, attempt=attempt,
                     took_s=took, why=err)
        self._advance(check, FAILED, None)
        self.push("Iris: the %s plan FAILED twice (%s) -- no imaging tonight. "
                  "Check the conductor's journal; `replan` retries before the sunset check."
                  % (check, err))

    def _advance(self, check: str, outcome: str, plan: Optional[dict]):
        """Walk the machine where the table has a row; the record stands either way."""
        slots = max(1, len((plan or {}).get("slots") or []))
        with self.c._lock:
            state = self.c.state
            if check == "noon" and state == "PLANNING":
                if outcome == GOOD:
                    self.c.slots = slots
                    self.c.offer("PLAN_GOOD", "clock", {"dso": plan.get("best"), "slots": slots})
                elif outcome == BAD:
                    self.c.offer("PLAN_BAD", "clock", {"dso": (plan or {}).get("best")})
                else:
                    self.c.offer("PLAN_FAILED", "clock", {})
            elif check == "sunset":
                if outcome == GOOD and state in ("ARMED", "PRE_FLIGHT"):
                    self.c.slots = slots           # the sunset plan may differ from noon's
                elif outcome == BAD and state in ("ARMED", "PRE_FLIGHT"):
                    self.c.offer("WEATHER_BAD", "clock", {"check": "sunset"})
                elif outcome == FAILED and state == "PRE_FLIGHT":
                    self.c.offer("PLAN_FAILED", "clock", {"check": "sunset"})

    def _record(self, day: date, check: str, outcome: str, plan=None, trigger="clock",
                attempt=1, took_s=None, why=None):
        data = {"date": day.isoformat(), "check": check, "outcome": outcome,
                "plan": plan, "trigger": trigger, "attempt": attempt}
        if took_s is not None:
            data["took_s"] = round(float(took_s), 1)
        if why:
            data["why"] = why
        return self.journal.append("note", NIGHT_PLAN, "clock", data=data)

    def _run_cli(self, refresh: bool) -> dict:
        from iris.conductor.planner import run_night_plan
        return run_night_plan(self.root, refresh=refresh)
