"""Switch the outside lights off AFTER their own sunset schedules have fired.

The start sequence (end_points/start.py) turns the lights off a couple of
minutes before sunset, and four of them -- Driveway, Grill, Iris landscape,
Main landscape -- have Kasa schedules that turn them back ON at sunset. On
2026-10-08 they came on at 18:20 and lit the yard until 22:00-23:00, through
the first four hours of imaging.

So image!! also starts this: a daemon thread that waits until sunset + 15
minutes (owner's choice, 2026-10-09), warns by Pushover one minute before,
then switches every light in LIGHTS off and verifies each. One pass, no
re-check ("let's see how that works"); the start sequence's own pass stays.
If imaging ends while it is still waiting it just stops. Imaging that starts
after that time (a manual run later in the evening) gets the pass at once.

LIGHTS is the one list: start.py builds its switch set from it too.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

_logger = logging.getLogger(__name__)

LIGHTS = ("Iris door light", "Iris inside light", "Driveway lights", "Grill Lights",
          "Iris landscape lights", "Main landscape lights", "SWAN", "Stairs")
DELAY = timedelta(minutes=15)
WARN = timedelta(minutes=1)
_thread = None


def due_at(sunset: datetime, now: datetime) -> datetime:
    """When the pass runs: sunset + DELAY, or now if that has passed. Pure."""
    target = sunset + DELAY
    return target if target > now else now


def warning_text(due: datetime, sunset: datetime, names) -> str:
    return ("Lights: switching off %d outside lights at %s (sunset was %s, their own schedules have fired): %s"
            % (len(names), due.strftime("%H:%M"), sunset.strftime("%H:%M"), ", ".join(names)))


def result_text(results: dict) -> str:
    failed = [n for n, ok in results.items() if not ok]
    if not failed:
        return "Lights: all %d off (verified)" % len(results)
    return "Lights: %d of %d off; could not switch %s" % (len(results) - len(failed), len(results), ", ".join(failed))


def run(post_cb=None, still_imaging=None, sleep=time.sleep) -> dict:
    """The pass itself. Returns the per-light results, or {} if abandoned."""
    import asyncio
    from zoneinfo import ZoneInfo
    from configs import config
    from iris_astronomy import weather
    from hardware_control import kasa_utils as ku
    tz = ZoneInfo(config.data()["location"]["timezone"])
    sunset = weather.get_sunrise_sunset()[1]
    sunset = (sunset if sunset.tzinfo else sunset.replace(tzinfo=timezone.utc)).astimezone(tz)
    due = due_at(sunset, datetime.now(tz))
    _logger.info("lights warden: pass due %s (sunset %s)", due.strftime("%H:%M:%S"), sunset.strftime("%H:%M"))
    while True:
        left = (due - datetime.now(tz)).total_seconds()
        if left <= WARN.total_seconds():
            break
        if still_imaging and not still_imaging():
            _logger.info("lights warden: imaging ended before the pass; nothing done")
            return {}
        sleep(min(30.0, left - WARN.total_seconds()))
    try:
        from utils import pushover
        pushover.push_message(warning_text(due, sunset, LIGHTS))
    except Exception:
        _logger.exception("lights warden: pushover failed")
    left = (due - datetime.now(tz)).total_seconds()
    if left > 0:
        sleep(left)
    dev_map = asyncio.run(ku.make_discovery_map())
    results = asyncio.run(ku.kasa_do(dev_map, {n: "off" for n in LIGHTS}))
    text = result_text(results)
    (_logger.warning if "could not" in text else _logger.info)("lights warden: %s", text)
    if post_cb:
        try:
            post_cb(text)
        except Exception:
            _logger.exception("lights warden: post failed")
    return results


def start(post_cb=None, still_imaging=None) -> bool:
    """Start the warden thread for this run; False if one is already running."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return False

    def _body():
        try:
            run(post_cb, still_imaging)
        except Exception:
            _logger.exception("lights warden failed")

    _thread = threading.Thread(target=_body, name="lights-warden", daemon=True)
    _thread.start()
    return True
