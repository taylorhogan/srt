#!/usr/bin/env python3
"""Hourly: record what every weather model predicted, so they can be scored later.

Groundwork, not a decision-maker. Nothing reads this file to gate imaging; it
exists so that in a few weeks there is enough paired data to answer "which model
should we actually believe" with a number instead of an anecdote.

The reason it exists: on 2026-08-14 Open-Meteo's default `best_match` reported 0%
cloud cover for every hour to midday while the sky camera was measuring overcast
(sky 8->19 ADU, limiting magnitude 4.96->3.6, matched stars 123->26, moon 42 deg
below the horizon so not moonrise). Queried individually, the models behind that
blend disagreed enormously for the same hour -- ICON 100%, UKMO 84%, MeteoFrance
64%, against GFS, HRRR, GEM and best_match all at 0%. See
local/evidence/20260814_openmeteo_cloud_miss/.

It would have been easy, that night, to switch to ICON and call it fixed. That
would have been fitting a model to one hour: ICON also read 5% at 04:00 and 100%
at 05:00, and UKMO fell to 0% by 05:00. Both are jumpy. The camera already
records ground truth every 5 minutes in local/sky_log.jsonl -- limiting_mag,
purity, sky_median_adu -- so the honest experiment is to log all the forecasts
and join them against it afterwards.

Everything is stored in **UTC on whole hours**, as parallel arrays sharing one
`valid_from_utc`. Local time is not used anywhere in the file: the join against
sky_log.jsonl has to survive the DST change in November, and an array indexed
off local midnight silently gains or loses an hour that night.

Both the issue time (`fetched`) and the valid time are kept, because "was the
forecast right" is a different question at 1 hour of lead than at 12, and only
the pair can separate them. That is also why the whole horizon is logged every
run rather than just the current hour.

`cloud_cover_low` is recorded beside total cover because low cloud is what
actually closes the roof; high cirrus can read as heavy cover while the sky is
still workable for imaging.

A source that fails is recorded as an error and the run continues -- losing the
NWS grid should not also throw away that hour's Open-Meteo sample. The run only
exits non-zero if *nothing* was captured, so a genuine outage still shows up as a
failed task rather than a silent gap.

Usage:  python scripts/forecast_log.py [--hours N] [--print] [--no-write]
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

# The blend (`best_match`) is logged alongside the raw models deliberately: it is
# what iris_astronomy/weather.py actually consumes today, so its error is the
# error the observatory is currently exposed to, and dropping it would hide the
# baseline the others need to beat.
MODELS = [
    "best_match",
    "ecmwf_ifs025",
    "gfs_seamless",
    "gfs_hrrr",
    "icon_seamless",
    "gem_seamless",
    "meteofrance_seamless",
    "jma_seamless",
    "ukmo_seamless",
    "knmi_seamless",
]

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS = "https://api.weather.gov/points/%.4f,%.4f"
# api.weather.gov rejects requests without a contact in the User-Agent.
NWS_HEADERS = {"User-Agent": "iris-observatory (taylor.hogan@gmail.com)"}

LOG_PATH = "local/forecast_log.jsonl"
DEFAULT_HOURS = 24


def _floor_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def fetch_open_meteo(lat, lon, hours):
    """Per-model cloud cover, on whole UTC hours starting at the current one."""
    r = requests.get(OPEN_METEO, params={
        "latitude": lat, "longitude": lon,
        "hourly": "cloud_cover,cloud_cover_low",
        "models": ",".join(MODELS),
        "timezone": "UTC",
        "forecast_days": 2,
    }, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    times = h["time"]

    start = _floor_hour(datetime.now(timezone.utc))
    stamp = start.strftime("%Y-%m-%dT%H:00")
    try:
        i0 = times.index(stamp)
    except ValueError:
        # Open-Meteo occasionally starts the array at the next hour. Falling back
        # to index 0 would silently shift every value by an hour, which is the
        # one error this file cannot tolerate, so take the first hour at or after
        # now and record what that actually was.
        i0 = next(i for i, t in enumerate(times) if t >= stamp)
        start = datetime.fromisoformat(times[i0]).replace(tzinfo=timezone.utc)

    def series(field, model):
        # With several models requested the keys are suffixed; with one they are
        # not. Ask for the suffixed key first so a single-model run cannot
        # quietly attribute one model's numbers to all of them.
        for key in ("%s_%s" % (field, model), field):
            if key in h:
                return h[key][i0:i0 + hours]
        return None

    out = {}
    for m in MODELS:
        cc = series("cloud_cover", m)
        if cc is None:
            continue
        out[m] = {"cc": cc, "ccl": series("cloud_cover_low", m)}
    if not out:
        raise RuntimeError("no cloud_cover series came back for any model")
    return start, out


def _expand_nws(values, start, hours):
    """NWS gives value + ISO-8601 duration intervals; flatten to hourly.

    A single entry can cover many hours (a settled forecast is published as one
    long interval), so reading one value per entry would both misalign the array
    and drop most of the horizon.
    """
    grid = {}
    for v in values:
        span = v["validTime"]
        head, _, dur = span.partition("/")
        t = datetime.fromisoformat(head).astimezone(timezone.utc)

        # PnDTnH -- days and hours are the only units NWS uses here.
        days = hrs = 0
        num = ""
        in_time = False
        for ch in dur.lstrip("P"):
            if ch.isdigit():
                num += ch
            elif ch == "T":
                in_time = True
                num = ""
            elif ch == "D":
                days = int(num or 0)
                num = ""
            elif ch == "H":
                hrs = int(num or 0)
                num = ""
            else:
                num = ""
        total = days * 24 + hrs or 1
        for k in range(total):
            grid[_floor_hour(t + timedelta(hours=k))] = v["value"]

    return [grid.get(start + timedelta(hours=k)) for k in range(hours)]


def fetch_nws(lat, lon, start, hours):
    p = requests.get(NWS_POINTS % (lat, lon), headers=NWS_HEADERS, timeout=45)
    p.raise_for_status()
    props = p.json()["properties"]
    g = requests.get(props["forecastGridData"], headers=NWS_HEADERS, timeout=45)
    g.raise_for_status()
    sky = g.json()["properties"]["skyCover"]["values"]
    return {
        "office": "%s %d,%d" % (props["gridId"], props["gridX"], props["gridY"]),
        "sky": _expand_nws(sky, start, hours),
    }


def collect(lat, lon, hours):
    errors = {}
    start, om = None, {}
    try:
        start, om = fetch_open_meteo(lat, lon, hours)
    except Exception as e:
        errors["open_meteo"] = "%s: %s" % (type(e).__name__, e)

    # Open-Meteo sets the window. If it failed there is nothing to align NWS to,
    # and an NWS-only row on its own clock would not join cleanly later.
    nws = None
    if start is not None:
        try:
            nws = fetch_nws(lat, lon, start, hours)
        except Exception as e:
            errors["nws"] = "%s: %s" % (type(e).__name__, e)

    row = {
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid_from_utc": start.isoformat(timespec="minutes") if start else None,
        "hours": hours,
        "open_meteo": om,
        "nws": nws,
    }
    if errors:
        row["errors"] = errors
    return row


def show(row):
    if not row.get("valid_from_utc"):
        print("nothing captured:", row.get("errors"))
        return
    start = datetime.fromisoformat(row["valid_from_utc"])
    n = min(6, row["hours"])
    head = [(start + timedelta(hours=k)).strftime("%H:%M") for k in range(n)]
    print("cloud cover %%, valid UTC (issued %s)" % row["fetched"][11:16])
    print("%-22s %s" % ("model", "  ".join("%5s" % x for x in head)))
    for m, d in row["open_meteo"].items():
        cells = ["%5s" % ("--" if v is None else v) for v in d["cc"][:n]]
        print("%-22s %s" % (m, "  ".join(cells)))
    if row.get("nws"):
        cells = ["%5s" % ("--" if v is None else v) for v in row["nws"]["sky"][:n]]
        print("%-22s %s" % ("nws " + row["nws"]["office"], "  ".join(cells)))
    if row.get("errors"):
        print("errors:", json.dumps(row["errors"]))


def main(argv):
    argv = list(argv)
    hours = DEFAULT_HOURS
    if "--hours" in argv:
        hours = int(argv[argv.index("--hours") + 1])

    cfg = config.data()
    loc = cfg["location"]
    row = collect(loc["latitude"], loc["longitude"], hours)

    if "--print" in argv or "--no-write" in argv:
        show(row)

    if "--no-write" not in argv:
        path = Path(LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        got = len(row["open_meteo"]) + (1 if row.get("nws") else 0)
        print("logged %d source(s) x %dh -> %s" % (got, hours, path))

    # A row with no sources at all is a failed run, and Task Scheduler should see
    # it as one -- see sky_monitor.bat on why silent successes are the enemy.
    return 0 if row["open_meteo"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
