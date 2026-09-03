"""
station_watch.py  (born iss_watch.py, renamed when Tiangong joined)
Predict space-station passes that cross the sky camera's field, and record
them. Watches the ISS and Tiangong (see SATELLITES). The DATA paths keep
their historical names on purpose -- local/iss_pass_armed.json and local/iss/
-- because an armed marker must survive a deploy that happens between arming
and the pass, and nothing is gained by orphaning the existing archive.

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

# The stations worth recording. Tiangong (CSS core module TIANHE, NORAD
# 48274) orbits at 41.5 deg inclination against this site's 41.8 deg
# latitude -- the tangent geometry: its passes always culminate toward the
# SOUTH, cluster in runs when the track's northern apex drifts across our
# longitude, and can top 80 deg on the best of them. Same public TLEs, same
# gates, one more catalogue number.
SATELLITES = (
    {"name": "ISS", "catnr": 25544, "cache": "local/iss_tle.json"},
    {"name": "Tiangong", "catnr": 48274, "cache": "local/tiangong_tle.json"},
)
_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=%d&FORMAT=TLE"
TLE_MAX_AGE_S = 12 * 3600
MARKER = "local/iss_pass_armed.json"
OUT_DIR = "local/iss"

# Rolling cap on stored VIDEO, same shape as the roof-audio library's cap: a
# pass is 10-40 MB and a few arrive per week, so 20 is under a GB and months
# of lookback. Sidecar .json files are ~1 KB and are deliberately NOT pruned:
# they remain a permanent log of every pass ever recorded, video or not.
KEEP_VIDEOS = 20

# The camera's usable floor. The plate solution puts the axis ~4 deg off
# zenith with a 104 deg field, so the edge sits near alt 38; 40 keeps the
# pass inside the frame rather than clipping its rise and set.
MIN_ALT_DEG = 40.0
SUN_DARK_DEG = -6.0        # site darkness: civil twilight

# Spawn the recorder when the pass starts within this window. sky_monitor
# ticks every 5 minutes, so 6 guarantees exactly one tick arms it.
SPAWN_AHEAD_S = 6 * 60

# The observatory's own horizon: az -> minimum visible altitude, from the
# same file the DSO scheduler uses (configs/my.hrz). The tree line runs
# 37-60 deg here, so "above the Iris horizon" is a genuinely different
# question from "in the sky camera's frame" -- the camera floor is a flat
# 40, the horizon is azimuth-dependent and mostly higher.
HORIZON_FILE = "configs/my.hrz"


def _horizon():
    """(az_list, alt_list) from my.hrz, wrapped so interpolation closes 360."""
    az, alt = [], []
    with open(HORIZON_FILE) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                az.append(float(parts[0])); alt.append(float(parts[1]))
    if az and az[0] != 0.0:
        az.insert(0, 0.0); alt.insert(0, alt[0])
    if az and az[-1] != 360.0:
        az.append(360.0); alt.append(alt[0])
    return az, alt


def _horizon_alt(azimuth, hz):
    import numpy as np
    return float(np.interp(azimuth % 360.0, hz[0], hz[1]))


def _fetch_tle(catnr):
    import requests
    r = requests.get(_TLE_URL % catnr, timeout=15)
    r.raise_for_status()
    lines = [l.strip() for l in r.text.splitlines() if l.strip()]
    if len(lines) < 3 or not lines[1].startswith("1 "):
        raise ValueError("unexpected TLE payload: %r" % r.text[:80])
    return lines[1], lines[2]


def get_tle(sat_def):
    """(line1, line2) for one satellite, cached; stale cache beats a failed fetch."""
    cache = sat_def["cache"]
    cached = None
    if os.path.exists(cache):
        try:
            cached = json.load(open(cache))
        except (OSError, ValueError):
            cached = None
    if cached and time.time() - cached.get("fetched", 0) < TLE_MAX_AGE_S:
        return cached["l1"], cached["l2"]
    try:
        l1, l2 = _fetch_tle(sat_def["catnr"])
        os.makedirs("local", exist_ok=True)
        json.dump({"l1": l1, "l2": l2, "fetched": time.time()},
                  open(cache, "w"))
        return l1, l2
    except Exception as e:  # noqa: BLE001
        if cached:
            _logger.warning("%s TLE fetch failed (%r); using cache from %s",
                            sat_def["name"], e,
                            datetime.fromtimestamp(cached["fetched"]))
            return cached["l1"], cached["l2"]
        raise


def _sky():
    from skyfield.api import Loader, wgs84
    loader = Loader("local/skyfield")
    eph = loader("de421.bsp")
    ts = loader.timescale()
    loc = config.data()["location"]
    site = wgs84.latlon(loc["latitude"], loc["longitude"])
    return site, eph, ts


def passes_tonight(hours=18.5, arm=True):
    """Visible-in-frame passes in the next *hours*; optionally arm the recorder.

    Returns a list of dicts {sat, rise, peak, set, peak_alt_deg} in local time
    ISO strings, all satellites merged and sorted by rise. "Visible in frame"
    = above MIN_ALT_DEG while sunlit with the site dark for at least part of
    the pass. One station's TLE failing must not blank the other's passes, so
    each is computed under its own try.

    18.5 hours, not 16: arming happens at the NOON check, and noon+16h only
    reaches 04:00 -- a real 04:35 ISS pass on 2026-09-04 fell outside it.
    Noon+18.5h reaches 06:30, past astronomical dawn year-round here, so the
    window means what the docstring above always claimed: "until sunrise".
    """
    from skyfield.api import EarthSatellite
    site, eph, ts = _sky()
    out = []
    for sat_def in SATELLITES:
        try:
            l1, l2 = get_tle(sat_def)
            sat = EarthSatellite(l1, l2, sat_def["name"], ts)
            out.extend(_passes_for(sat_def["name"], sat, site, eph, ts, hours))
        except Exception:  # noqa: BLE001
            _logger.warning("%s pass prediction failed (skipped)",
                            sat_def["name"], exc_info=True)
    out.sort(key=lambda p: p["rise"])
    camera_passes = [p for p in out if p["in_camera"]]
    if arm and camera_passes:
        json.dump({"passes": camera_passes,
                   "armed": datetime.now().astimezone().isoformat(timespec="seconds")},
                  open(MARKER, "w"), indent=1)
        _logger.info("recorder armed for %d camera pass(es); first peak %s (%s)",
                     len(camera_passes), camera_passes[0]["peak"],
                     camera_passes[0]["sat"])
    return out


def _passes_for(name, sat, site, eph, ts, hours):
    """One satellite's qualifying passes, each tagged with its name."""
    sun, earth = eph["sun"], eph["earth"]
    hz = _horizon()
    t0 = ts.now()
    t1 = ts.from_datetime(t0.utc_datetime() + timedelta(hours=hours))
    # 35, not MIN_ALT_DEG: the horizon dips to 36.9 in places, so a pass can
    # clear the tree line without ever reaching the camera's 40-deg floor.
    t, events = sat.find_events(site, t0, t1, altitude_degrees=35.0)

    out, current = [], {}
    for ti, ev in zip(t, events):
        if ev == 0:
            current = {"rise": ti}
        elif ev == 1:
            current["peak"] = ti
        elif ev == 2 and "peak" in current:
            current["set"] = ti
            # Sample the arc every ~15 s. A pass counts for a category if
            # ANY sampled point satisfies it while the ISS is sunlit over a
            # dark site; the categories are judged independently because the
            # camera floor is flat and the horizon is azimuth-shaped.
            span = (current["set"].utc_datetime()
                    - current["rise"].utc_datetime()).total_seconds()
            in_camera = above_horizon = False
            n = max(2, int(span // 15))
            for i in range(n + 1):
                tx = ts.from_datetime(current["rise"].utc_datetime()
                                      + timedelta(seconds=span * i / n))
                sunlit = bool(sat.at(tx).is_sunlit(eph))
                sun_alt = (earth + site).at(tx).observe(sun).apparent()\
                    .altaz()[0].degrees
                if not (sunlit and sun_alt < SUN_DARK_DEG):
                    continue
                alt, az, _ = (sat - site).at(tx).altaz()
                if alt.degrees >= MIN_ALT_DEG:
                    in_camera = True
                if alt.degrees > _horizon_alt(az.degrees, hz):
                    above_horizon = True
                if in_camera and above_horizon:
                    break
            if in_camera or above_horizon:
                alt, _, _ = (sat - site).at(current["peak"]).altaz()
                out.append({
                    "sat": name,
                    "rise": current["rise"].utc_datetime().astimezone().isoformat(timespec="seconds"),
                    "peak": current["peak"].utc_datetime().astimezone().isoformat(timespec="seconds"),
                    "set": current["set"].utc_datetime().astimezone().isoformat(timespec="seconds"),
                    "peak_alt_deg": round(float(alt.degrees), 1),
                    "in_camera": in_camera,
                    "above_horizon": above_horizon,
                })
            current = {}
    return out


def armed_tracks(step_s=15.0):
    """Alt/az tracks for armed passes still in the future, for sky charts.

    Returns [{sat, peak, pts: [(az_deg, alt_deg), ...]}, ...] recomputed from
    the cached TLEs across each pass's rise..set. Empty list on any trouble --
    a chart must never fail because of an optional overlay.
    """
    try:
        passes = (json.load(open(MARKER)) or {}).get("passes") or []
    except Exception:
        return []
    if not passes:
        return []
    out = []
    try:
        from skyfield.api import EarthSatellite
        site, eph, ts = _sky()
        now = datetime.now().astimezone()
        for p in passes:
            try:
                t0 = datetime.fromisoformat(p["rise"])
                t1 = datetime.fromisoformat(p["set"])
                if t1 < now:
                    continue
                sat_def = next((s for s in SATELLITES
                                if s["name"] == p.get("sat", "ISS")), None)
                if sat_def is None:
                    continue
                l1, l2 = get_tle(sat_def)
                sat = EarthSatellite(l1, l2, sat_def["name"], ts)
                span = (t1 - t0).total_seconds()
                n = max(2, int(span // step_s))
                pts = []
                for i in range(n + 1):
                    tx = ts.from_datetime(t0 + timedelta(seconds=span * i / n))
                    alt, az, _ = (sat - site).at(tx).altaz()
                    pts.append((float(az.degrees), float(alt.degrees)))
                out.append({"sat": sat_def["name"], "peak": p["peak"], "pts": pts})
            except Exception:  # noqa: BLE001
                _logger.warning("armed track skipped", exc_info=True)
    except Exception:  # noqa: BLE001
        _logger.warning("armed_tracks unavailable", exc_info=True)
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
            _logger.info("%s pass imminent (rise %s) - spawning recorder",
                         launch.get("sat", "ISS"), launch["rise"])
            subprocess.Popen(
                [sys.executable, "-m", "sentry.station_watch", "record",
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
        _logger.exception("station_watch.check_and_spawn failed (ignored)")


def _prune_videos(keep=KEEP_VIDEOS):
    """Delete all but the newest *keep* .h264 bursts; sidecars are kept.

    Runs after every successful recording, so the cap can never be exceeded
    by more than one pass. Never raises -- a failed prune must not cost the
    'pass recorded' post that follows it.
    """
    try:
        vids = sorted((f for f in os.listdir(OUT_DIR) if f.endswith(".h264")),
                      key=lambda f: os.path.getmtime(os.path.join(OUT_DIR, f)))
        for f in vids[:-keep] if keep > 0 else vids:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(OUT_DIR, f))
                _logger.info("pruned old pass video %s (cap %d)", f, keep)
    except Exception:  # noqa: BLE001
        _logger.warning("video prune failed (ignored)", exc_info=True)


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

    # "ISS" default: a marker armed before satellites were tagged has no sat.
    sat = str(p.get("sat", "ISS")).lower()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = rise.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, "%s_%s.h264" % (sat, stamp))
    started = datetime.now().astimezone()
    path, n = sky_camera.capture_burst(seconds=seconds, out_path=out)
    ended = datetime.now().astimezone()
    meta = {"pass": p, "burst": str(path), "frames": n,
            "capture_start": started.isoformat(timespec="seconds"),
            "capture_end": ended.isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(OUT_DIR, "%s_%s.json" % (sat, stamp)), "w"),
              indent=1)

    _prune_videos()

    msg = ("%s pass recorded: peak alt %.0f deg at %s, %s frames -> %s"
           % (p.get("sat", "ISS"), p["peak_alt_deg"], p["peak"][11:19], n, path))
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
        print("no visible-in-frame station passes in the window")
        return 1
    for p in passes:
        tags = [t for t, on in (("camera", p.get("in_camera")),
                                ("horizon", p.get("above_horizon"))) if on]
        print("%-9s rise %s  peak %s (alt %.0f)  set %s  [%s]"
              % (p.get("sat", "ISS"), p["rise"][11:19], p["peak"][11:19],
                 p["peak_alt_deg"], p["set"][11:19], "+".join(tags)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
