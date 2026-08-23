"""
iss_watch.py
Predict ISS passes that cross the sky camera's field, and record them.

The sky camera's 104-degree field centred ~4 degrees off zenith reaches down
to roughly 38 degrees altitude, so most passes never enter it, and a pass is
only *seeable* when three independent things line up: the ISS above the
camera's floor, the spacecraft in sunlight, and the site in darkness. That
triple lines up for a high pass only every couple of weeks -- the same cycle
naked-eye watchers know as a "good ISS week".

Three entry points:

  passes_tonight()   what `tonight` calls: visible passes between now and
                     sunrise, and it ARMS the recorder by writing MARKER.
  check_and_spawn()  called by sky_monitor's 5-minute cycle: if an armed
                     pass starts within the next few minutes, spawn the
                     recorder detached and disarm. The 5-minute cadence is
                     why the arm window is 6 minutes: any tick lands in it.
  record()           the detached recorder: sleeps to the start, captures
                     the pass as one H.264 burst, posts the result.

TLE handling: cached in local/iss_tle.json, refetched past 12 hours, stale
cache used on fetch failure -- a day-old TLE still puts pass times within
seconds, and a network blip must not blank the tonight report. Celestrak
asks not to be polled more often than every couple of hours; the cache is
also politeness.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from utils import utils

_logger = utils.set_logger()

TLE_URL = ("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE")
TLE_CACHE = "local/iss_tle.json"
TLE_MAX_AGE_S = 12 * 3600
MARKER = "local/iss_pass_armed.json"
OUT_DIR = "local/iss"

# The camera's usable floor. The plate solution puts the axis ~4 deg off
# zenith with a 104 deg field, so the edge sits near alt 38; 40 keeps the
# pass inside the frame rather than clipping its rise and set.
MIN_ALT_DEG = 40.0
SUN_DARK_DEG = -6.0        # site darkness: civil twilight

# Spawn the recorder when the pass starts within this window. sky_monitor
# ticks every 5 minutes, so 6 guarantees exactly one tick arms it.
SPAWN_AHEAD_S = 6 * 60


def _fetch_tle():
    import requests
    r = requests.get(TLE_URL, timeout=15)
    r.raise_for_status()
    lines = [l.strip() for l in r.text.splitlines() if l.strip()]
    if len(lines) < 3 or not lines[1].startswith("1 "):
        raise ValueError("unexpected TLE payload: %r" % r.text[:80])
    return lines[1], lines[2]


def get_tle():
    """(line1, line2), cached; stale cache beats a failed fetch."""
    cached = None
    if os.path.exists(TLE_CACHE):
        try:
            cached = json.load(open(TLE_CACHE))
        except (OSError, ValueError):
            cached = None
    if cached and time.time() - cached.get("fetched", 0) < TLE_MAX_AGE_S:
        return cached["l1"], cached["l2"]
    try:
        l1, l2 = _fetch_tle()
        os.makedirs("local", exist_ok=True)
        json.dump({"l1": l1, "l2": l2, "fetched": time.time()},
                  open(TLE_CACHE, "w"))
        return l1, l2
    except Exception as e:  # noqa: BLE001
        if cached:
            _logger.warning("ISS TLE fetch failed (%r); using cache from %s",
                            e, datetime.fromtimestamp(cached["fetched"]))
            return cached["l1"], cached["l2"]
        raise


def _sky():
    from skyfield.api import EarthSatellite, Loader, wgs84
    loader = Loader("local/skyfield")
    eph = loader("de421.bsp")
    ts = loader.timescale()
    l1, l2 = get_tle()
    sat = EarthSatellite(l1, l2, "ISS", ts)
    loc = config.data()["location"]
    site = wgs84.latlon(loc["latitude"], loc["longitude"])
    return sat, site, eph, ts


def passes_tonight(hours=16.0, arm=True):
    """Visible-in-frame passes in the next *hours*; optionally arm the recorder.

    Returns a list of dicts {rise, peak, set, peak_alt_deg} in local time
    ISO strings. "Visible in frame" = above MIN_ALT_DEG while sunlit with the
    site dark for at least part of the pass.
    """
    sat, site, eph, ts = _sky()
    sun, earth = eph["sun"], eph["earth"]
    t0 = ts.now()
    t1 = ts.from_datetime(t0.utc_datetime() + timedelta(hours=hours))
    t, events = sat.find_events(site, t0, t1, altitude_degrees=MIN_ALT_DEG)

    out, current = [], {}
    for ti, ev in zip(t, events):
        if ev == 0:
            current = {"rise": ti}
        elif ev == 1:
            current["peak"] = ti
        elif ev == 2 and "peak" in current:
            current["set"] = ti
            # Sample the arc: visible if any point is sunlit over a dark site.
            span = (current["set"].utc_datetime()
                    - current["rise"].utc_datetime()).total_seconds()
            vis = False
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                tx = ts.from_datetime(current["rise"].utc_datetime()
                                      + timedelta(seconds=span * frac))
                sunlit = bool(sat.at(tx).is_sunlit(eph))
                sun_alt = (earth + site).at(tx).observe(sun).apparent()\
                    .altaz()[0].degrees
                if sunlit and sun_alt < SUN_DARK_DEG:
                    vis = True
                    break
            if vis:
                alt, _, _ = (sat - site).at(current["peak"]).altaz()
                out.append({
                    "rise": current["rise"].utc_datetime().astimezone().isoformat(timespec="seconds"),
                    "peak": current["peak"].utc_datetime().astimezone().isoformat(timespec="seconds"),
                    "set": current["set"].utc_datetime().astimezone().isoformat(timespec="seconds"),
                    "peak_alt_deg": round(float(alt.degrees), 1),
                })
            current = {}
    if arm and out:
        json.dump({"passes": out, "armed": datetime.now().astimezone().isoformat(timespec="seconds")},
                  open(MARKER, "w"), indent=1)
        _logger.info("ISS recorder armed for %d pass(es); first peak %s",
                     len(out), out[0]["peak"])
    return out


def check_and_spawn():
    """sky_monitor hook: spawn the detached recorder for an imminent armed pass.

    Never raises; a failure here must not touch the monitor's weather work.
    """
    try:
        if not os.path.exists(MARKER):
            return
        data = json.load(open(MARKER))
        passes = data.get("passes") or []
        now = datetime.now().astimezone()
        keep, launch = [], None
        for p in passes:
            rise = datetime.fromisoformat(p["rise"])
            if rise < now - timedelta(minutes=2):
                continue                      # missed / already handled
            if (rise - now).total_seconds() <= SPAWN_AHEAD_S and launch is None:
                launch = p
            else:
                keep.append(p)
        if launch:
            _logger.info("ISS pass imminent (rise %s) - spawning recorder",
                         launch["rise"])
            subprocess.Popen(
                [sys.executable, "-m", "sentry.iss_watch", "record",
                 json.dumps(launch)],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if keep != passes:
            if keep:
                data["passes"] = keep
                json.dump(data, open(MARKER, "w"), indent=1)
            else:
                with contextlib.suppress(OSError):
                    os.remove(MARKER)
    except Exception:  # noqa: BLE001 -- observer beside the weather loop
        _logger.exception("iss_watch.check_and_spawn failed (ignored)")


def record(p):
    """Capture the pass as one uninterrupted burst and post the outcome.

    One stream for the whole pass rather than repeated short bursts, for the
    same reason the roof-audio recorder works that way: reconnect gaps land
    in the middle of the event. The stream also carries no timestamps, so
    the burst is bracketed with wall-clock times in the sidecar.
    """
    from sentry import sky_camera

    rise = datetime.fromisoformat(p["rise"])
    setts = datetime.fromisoformat(p["set"])
    lead = 15.0
    wait = (rise - datetime.now().astimezone()).total_seconds() - lead
    if wait > 0:
        time.sleep(wait)
    seconds = (setts - datetime.now().astimezone()).total_seconds() + lead
    seconds = max(30.0, min(seconds, 15 * 60))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = rise.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, "iss_%s.h264" % stamp)
    started = datetime.now().astimezone()
    path, n = sky_camera.capture_burst(seconds=seconds, out_path=out)
    ended = datetime.now().astimezone()
    meta = {"pass": p, "burst": str(path), "frames": n,
            "capture_start": started.isoformat(timespec="seconds"),
            "capture_end": ended.isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(OUT_DIR, "iss_%s.json" % stamp), "w"),
              indent=1)

    msg = ("ISS pass recorded: peak alt %.0f deg at %s, %s frames -> %s"
           % (p["peak_alt_deg"], p["peak"][11:19], n, path))
    _logger.info(msg)
    try:
        import requests
        requests.post("http://127.0.0.1:8095/api/post", data={"message": msg},
                      timeout=10)
    except Exception:  # noqa: BLE001
        _logger.warning("could not post ISS capture notice", exc_info=True)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(json.loads(sys.argv[2]))
        return 0
    passes = passes_tonight(arm="--arm" in sys.argv)
    if not passes:
        print("no visible-in-frame ISS passes in the window")
        return 1
    for p in passes:
        print("rise %s  peak %s (alt %.0f)  set %s"
              % (p["rise"][11:19], p["peak"][11:19], p["peak_alt_deg"],
                 p["set"][11:19]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
