"""Switch the outside lights off AFTER their own sunset schedules have fired.

The start sequence (end_points/start.py) turns the lights off a couple of
minutes before sunset, and four of them -- Driveway, Grill, Iris landscape,
Main landscape -- have Kasa schedules that turn them back ON at sunset. On
2026-10-08 they came on at 18:20 and lit the yard until 22:00-23:00, through
the first four hours of imaging.

So image!! also starts this: a daemon thread that switches every light in
LIGHTS off, verified, ten minutes before the first exposure is expected,
with a Pushover one minute before. One pass, no re-check (owner, 2026-10-09:
"let's see how that works"); the start sequence's own pass stays. If imaging
ends while it is still waiting it just stops; imaging that starts after the
due time (a manual run later in the evening) gets the pass at once.

When is the first exposure? Not at the roof open. Every slot's container
begins with "Wait for object to appear": WaitForTime at nautical dusk plus
the template's offset (5 min), then WaitUntilAboveHorizon, then centering
and autofocus (~8 min). So first exposure ~ nautical dusk + 5 + 8, and the
lights go off LEAD (10 min) before that: nautical dusk + 3 min. Measured
2026-10-08: dusk 19:21, first frame 19:34:36. Never earlier than sunset + 3
min, so the plugs' own sunset schedules have fired before the pass. The
household keeps its lights for an hour after sunset instead of fifteen
minutes.

LIGHTS is the one list: start.py builds its switch set from it too.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

_logger = logging.getLogger(__name__)

LIGHTS = ("Iris door light", "Iris inside light", "Driveway lights", "Grill Lights",
          "Iris landscape lights", "Main landscape lights", "SWAN", "Stairs")
SETUP = timedelta(minutes=8)          # centering + autofocus after the wait ends
LEAD = timedelta(minutes=10)          # lights off this long before the first exposure
AFTER_SUNSET = timedelta(minutes=3)   # the plugs' sunset schedules have fired by then
DEFAULT_WAIT_OFFSET = timedelta(minutes=5)
WARN = timedelta(minutes=1)
_thread = None


def first_exposure(nautical_dusk: datetime, wait_offset: timedelta = DEFAULT_WAIT_OFFSET) -> datetime:
    """Expected first exposure: the slot's wait ends at dusk + offset, then setup. Pure."""
    return nautical_dusk + wait_offset + SETUP


def due_at(sunset: datetime, nautical_dusk: datetime, now: datetime,
           wait_offset: timedelta = DEFAULT_WAIT_OFFSET) -> datetime:
    """When the pass runs: LEAD before the first exposure, never before sunset +
    AFTER_SUNSET, and now if that has already passed. Pure."""
    target = max(first_exposure(nautical_dusk, wait_offset) - LEAD, sunset + AFTER_SUNSET)
    return target if target > now else now


def sequence_wait_offset(path) -> timedelta:
    """The first slot's nautical-dusk WaitForTime offset from the written sequence,
    or the template default if the file cannot be read."""
    import json
    try:
        with open(path, encoding="utf-8-sig") as fh:
            seq = json.load(fh)
    except Exception:
        return DEFAULT_WAIT_OFFSET
    found = []

    def walk(n):
        if isinstance(n, dict):
            t = str(n.get("$type", ""))
            prov = n.get("SelectedProvider") or {}
            if "WaitForTime" in t and "NauticalDusk" in str(prov.get("$type", "")):
                found.append(float(n.get("MinutesOffset") or 0))
            for k, v in n.items():
                if k != "Parent":
                    walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(seq)
    return timedelta(minutes=found[0]) if found else DEFAULT_WAIT_OFFSET


def warning_text(due: datetime, sunset: datetime, nautical_dusk: datetime, first: datetime, names) -> str:
    return ("Lights: first exposure expected about %s (nautical dusk %s, sunset was %s); switching off %d "
            "outside lights at %s: %s" % (first.strftime("%H:%M"), nautical_dusk.strftime("%H:%M"),
                                           sunset.strftime("%H:%M"), len(names), due.strftime("%H:%M"), ", ".join(names)))


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
    from hardware_control import kasa_utils as ku
    cfg = config.data()
    tz = ZoneInfo(cfg["location"]["timezone"])
    sunset, nautical = twilights(cfg, tz)
    offset = sequence_wait_offset(cfg["nina"]["sequence_output"])
    first = first_exposure(nautical, offset)
    due = due_at(sunset, nautical, datetime.now(tz), offset)
    _logger.info("lights warden: pass due %s (sunset %s, nautical dusk %s, wait offset %s, first exposure ~%s)",
                 due.strftime("%H:%M:%S"), sunset.strftime("%H:%M"), nautical.strftime("%H:%M"), offset, first.strftime("%H:%M"))
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
        pushover.push_message(warning_text(due, sunset, nautical, first, LIGHTS))
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


def twilights(cfg: dict, tz):
    """(sunset, nautical dusk) for the coming evening, local, from astroplan --
    the same definitions N.I.N.A's providers use."""
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    from astroplan import Observer
    L = cfg["location"]
    obs = Observer(location=EarthLocation.from_geodetic(L["longitude"] * u.deg, L["latitude"] * u.deg,
                                                        L["elevation"] * u.m), timezone=str(tz))
    now = Time(datetime.now(timezone.utc))
    sunset = obs.sun_set_time(now, which="next").to_datetime(timezone=tz)
    nautical = obs.twilight_evening_nautical(now, which="next").to_datetime(timezone=tz)
    if nautical < sunset:                  # between sunset and dusk: "next" sunset is tomorrow's
        sunset = obs.sun_set_time(now - 1 * u.day, which="next").to_datetime(timezone=tz)
    return sunset, nautical


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
