#!/usr/bin/env python3
"""Retention policy for the sky-frame archive: keep the rare, prune the rest.

A flat rolling cap is the wrong policy for this archive. Clear frames outnumber
rainy ones by orders of magnitude, so "keep the newest N" throws away a genuine
storm to make room for the four thousandth picture of a clear sky. The frames
worth having are exactly the ones a cap deletes first, and a deleted frame
cannot be recovered.

So: label frames, preserve the interesting ones out of the pruner's reach, and
let the cap fall on the boring majority.

Two labels, deliberately from independent sources:

  * What the weather record says happened -- hourly precipitation in mm from
    Open-Meteo, asked retrospectively with past_days. Note this is the ACTUAL
    figure, not the precipitation_probability forecast that weather.py reads
    for planning; a forecast is the wrong thing to label the past with.

  * What our own detector thought -- frames it marked untrustworthy. These are
    the ambiguous ones, and the disagreements between the two labels are the
    most valuable frames in the archive: weather says dry but the detector
    balked means dew, insects, or something else worth being able to name.

Preserved frames move to <capture_dir>/keep/, which the pruner's non-recursive
glob cannot see. That is the whole enforcement mechanism -- no database to fall
out of sync with the files.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

INDEX = "local/sky_log.jsonl"
KEEP_DIR = "keep"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cam(cfg=None):
    return (cfg or config.data()).get("sky camera", {})


def append_index(status, root=None):
    """One line per capture ATTEMPT. Tiny, and also the night's time series.

    Attempt, not success: a run whose camera did not answer appends a row with
    camera="unavailable" and no `captured`. Without those rows the file cannot
    answer how often the camera works, which is the question it looks like it
    answers.
    """
    p = (Path(root) if root else _root()) / INDEX
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in status.items() if not k.startswith("_")}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return p


def read_index(root=None):
    p = (Path(root) if root else _root()) / INDEX
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue                       # a torn line must not kill the run
    return rows


def precipitation_hours(past_days=5, cfg=None):
    """{UTC hour -> mm} for hours that actually had precipitation.

    Returns None (not an empty dict) if the lookup fails, so callers can tell
    "no rain" apart from "do not know" and decline to prune on ignorance.
    """
    import requests
    loc = (cfg or config.data())["location"]
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=20,
                         params={"latitude": loc["latitude"],
                                 "longitude": loc["longitude"],
                                 "hourly": "precipitation,rain,snowfall",
                                 "past_days": int(past_days),
                                 "forecast_days": 1,
                                 "timezone": "UTC"})
        r.raise_for_status()
        h = r.json()["hourly"]
    except Exception:
        return None
    out = {}
    for t, precip, rain, snow in zip(h["time"], h["precipitation"], h["rain"],
                                     h["snowfall"]):
        if precip is None:
            continue
        if precip > 0 or (snow or 0) > 0:
            out[datetime.fromisoformat(t).replace(tzinfo=timezone.utc)] = {
                "precip_mm": precip, "rain_mm": rain, "snow_cm": snow}
    return out


def _frame_hour(path):
    from sentry import plate_solve
    return plate_solve.frame_time(path).replace(minute=0, second=0,
                                                microsecond=0)


def preserve_events(root=None, past_days=5, verbose=False):
    """Move frames worth keeping into keep/. Returns a summary dict."""
    root = Path(root) if root else _root()
    cfg = config.data()
    cam = _cam(cfg)
    d = root / cam.get("capture_dir", "local/sky_frames")
    if not d.is_dir():
        return {"moved": 0, "reason": "no capture dir"}
    keep = d / KEEP_DIR
    keep.mkdir(parents=True, exist_ok=True)

    wet = precipitation_hours(past_days, cfg)
    # Frames the detector itself distrusted. Available with no network, so this
    # half of the policy still works when the weather lookup is down.
    flagged = {row.get("captured", "")[:13] for row in read_index(root)
               if row.get("trustworthy") is False}

    moved, why = 0, {"weather": 0, "detector": 0}
    for f in sorted(d.glob("sky_*.jpg")):
        hour = _frame_hour(f)
        reason = None
        if wet is not None and hour in wet:
            reason = "weather"
        elif hour.strftime("%Y-%m-%dT%H") in flagged:
            reason = "detector"
        if reason is None:
            continue
        try:
            f.rename(keep / f.name)
            moved += 1
            why[reason] += 1
            if verbose:
                print("  keep (%s): %s" % (reason, f.name))
        except OSError:
            pass

    # keep/ is exempt from the rolling cap, so it needs its own ceiling or a
    # long wet spell would fill the disk unattended.
    cap = int(cam.get("keep_event_frames", 4000))
    files = sorted(keep.glob("sky_*.jpg"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    dropped = 0
    for old in files[cap:]:
        try:
            old.unlink()
            dropped += 1
        except OSError:
            pass
    return {"moved": moved, "by": why, "kept_total": min(len(files), cap),
            "over_cap_deleted": dropped,
            "weather": "unavailable" if wet is None else "%d wet hours" % len(wet)}


def main(argv):
    verbose = "--verbose" in argv or "-v" in argv
    res = preserve_events(verbose=verbose)
    print(json.dumps(res, indent=1))
    rows = read_index()
    print("index: %d captures logged" % len(rows))
    if rows:
        bad = sum(1 for r in rows if r.get("trustworthy") is False)
        print("  %d flagged untrustworthy" % bad)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
