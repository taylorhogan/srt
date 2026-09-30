#!/usr/bin/env python3
"""At dusk, ask the sky camera whether tonight is actually clear.

The go/no-go is decided from a FORECAST, at noon and again 10 minutes before
sunset, and until now nothing compared it with the sky. The sky camera has been
measuring the real sky every 5 minutes since August. This is the missing
comparison: once per evening, as soon as the camera has enough dark frames, it
reads how clear the sky is and says so in the chat. When the camera and the
plan disagree, it sends a Pushover too:

  * the plan says skip, the camera says CLEAR: a night the forecast threw away
    (2026-08-13: 0 good hours forecast, 5.8 clear hours measured)
  * the plan says image, the camera says CLOUDY: a night that will open into cloud

ADVISORY ONLY. It never starts, stops or delays anything. Whether a dusk
reading should gate image!! is a separate decision for the owner. The timing
alone argues against it being an automatic gate as things stand: the camera
counts stars only once the sun is 10 deg down (iris_astronomy/sun.is_night),
about 50 min after sunset, and image!! fires 10 min BEFORE sunset.

THE READING. Transparency = plate-solve matches / catalogue stars (V <= 5) in
the field, the same quantity as scripts/sky_transparency_report.py, as the
median of the first DUSK_FRAMES night frames of the evening, divided by a
clear-sky reference. The reference is fixed, not per solution, because a live
check cannot wait for a moon-free week to measure the current solution's
ceiling: the solution epochs scored by forecast_score.py put a clear sky at
72-84%, so 0.75 by default.

WHY IT CAN BE TRUSTED AT DUSK (replayed over every night 2026-08-08..09-24,
moon-free ceilings, rest-of-night hours clear at >= 0.5):
  dusk < 0.20 of a clear sky    20 nights, 18 of them had no clear hour at all
  dusk >= 0.65                  12 nights, 10 of them mostly clear
  in between                    mixed -> UNSURE, said as such
Twilight lowers the reading a little, so a dusk CLEAR is if anything
conservative.

THE MOON only ever lowers the reading, so under a moon (up and >= 30% lit) a
low reading cannot tell cloud from moonlight: 2026-09-29 read 8% all night under
an 86% moon. Moonlit evenings can therefore be CLEAR or UNSURE, never CLOUDY.

Hooked into scripts/sky_monitor.py, which already runs every 5 minutes and
already knows whether it is night, for the same reason as the dawn report:
changing a scheduled task on this machine needs an elevation prompt.

    python sentry/dusk_check.py                    # tonight, if it is dusk
    python sentry/dusk_check.py --date 2026-09-14  # replay an evening, no post
"""
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

ROOT = Path(__file__).resolve().parents[1]
SKY_LOG = ROOT / "local" / "sky_log.jsonl"
DUSK_LOG = ROOT / "local" / "dusk_check.jsonl"
MARKER = ROOT / "local" / "dusk_check_last.txt"
STATE_FILE = ROOT / "scheduler_state.json"
POST_URL = "http://127.0.0.1:8095/api/post"

DEFAULTS = {
    "dusk_frames": 3,
    # Past this sun altitude with fewer frames than wanted (camera dropouts,
    # ~1 run in 5), read what there is rather than say nothing all night.
    "dusk_give_up_sun_deg": -15.0,
    "dusk_clear_transparency": 0.75,
    "dusk_clear_rel": 0.65,
    "dusk_cloudy_rel": 0.20,
    "dusk_moon_max_illum": 0.30,
}


def _settings(cfg=None):
    w = ((cfg or config.data()).get("weather") or {})
    return {k: w.get(k, v) for k, v in DEFAULTS.items()}


def _tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo(config.data()["location"]["timezone"])


def evening_frames(evening: date, rows=None):
    """[(utc_time, row)] of that evening's night frames with a solve, oldest first.

    Evening = local date `evening`, local time after noon. Dawn frames of the
    same date belong to the previous night and are excluded by the noon cut.
    """
    tz = _tz()
    out = []
    if rows is None:
        rows = []
        if SKY_LOG.exists():
            for line in SKY_LOG.open():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    for r in rows:
        if not r.get("night") or r.get("solve_matches") is None:
            continue
        t = datetime.fromisoformat(r.get("captured") or r["generated"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        local = t.astimezone(tz)
        if local.date() == evening and local.hour >= 12:
            out.append((t.astimezone(timezone.utc), r))
    return sorted(out, key=lambda p: p[0])


def moon_at(when):
    """(alt_deg, illuminated_fraction) at a UTC datetime."""
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    loc = config.data()["location"]
    site = EarthLocation.from_geodetic(float(loc["longitude"]) * u.deg,
                                       float(loc["latitude"]) * u.deg,
                                       float(loc.get("elevation", 0)) * u.m)
    t = Time(when)
    moon = get_body("moon", t, site)
    alt = moon.transform_to(AltAz(obstime=t, location=site)).alt.deg
    elong = moon.separation(get_body("sun", t, site)).rad
    return float(alt), float(0.5 * (1 - np.cos(elong)))


def read_sky(frames, s):
    """Transparency reading from dusk frames -> dict (no verdict yet)."""
    import numpy as np
    from sentry import plate_solve
    from scripts.sky_transparency_report import in_field

    sol = plate_solve.load()
    if sol is None:
        return {"error": "no plate solution stored"}
    per = []
    for t, r in frames:
        n = in_field(sol, t)
        if n > 0:
            per.append(r["solve_matches"] / n)
    if not per:
        return {"error": "no catalogue stars in field"}
    transp = float(np.median(per))
    malt, mill = moon_at(frames[-1][0])
    return {
        "frames": len(per),
        "first_utc": frames[0][0].isoformat(timespec="seconds"),
        "last_utc": frames[-1][0].isoformat(timespec="seconds"),
        "sun_alt_deg": frames[-1][1].get("sun_alt_deg"),
        "transparency": round(transp, 3),
        "rel": round(transp / float(s["dusk_clear_transparency"]), 3),
        "moon_alt_deg": round(malt, 1),
        "moon_illum": round(mill, 2),
        "moonlit": bool(malt > 0 and mill >= float(s["dusk_moon_max_illum"])),
    }


def verdict(reading, s):
    """CLEAR / CLOUDY / UNSURE from a reading. Moonlight never yields CLOUDY."""
    rel = reading["rel"]
    if rel >= float(s["dusk_clear_rel"]):
        return "CLEAR"
    if rel < float(s["dusk_cloudy_rel"]) and not reading["moonlit"]:
        return "CLOUDY"
    return "UNSURE"


def tonight_plan(evening: date):
    """What the scheduler decided for this evening, or None if it did not say.

    scheduler_state.json is rewritten at the noon and pre-sunset checks; one
    written before today's noon is yesterday's decision and must not be read
    as tonight's.
    """
    try:
        st = json.loads(STATE_FILE.read_text())
        written = datetime.fromtimestamp(STATE_FILE.stat().st_mtime, _tz())
    except (OSError, ValueError):
        return None
    if written.date() != evening or written.hour < 12:
        return None
    if "will image tonight" not in st:
        return None
    mode = "manual"
    try:
        if (ROOT / "mode.txt").read_text().splitlines()[0].strip() == "MODE AUTO":
            mode = "auto"
    except (OSError, IndexError):
        pass
    return {"image": bool(st["will image tonight"]), "dso": st.get("dso"), "mode": mode}


def compose(reading, v, plan):
    """(message, disagree) for the chat."""
    moon = (" Moon up, %d%% lit, so a low reading can't be told from moonlight."
            % round(100 * reading["moon_illum"])) if reading["moonlit"] and v != "CLEAR" else ""
    head = ("Dusk sky check: %s. The sky cam sees %d%% of the catalogue stars it "
            "should (%.2f of a clear sky, median of %d frames, sun %.0f deg).%s"
            % (v, round(100 * reading["transparency"]), reading["rel"],
               reading["frames"], reading["sun_alt_deg"] or 0, moon))
    if plan is None:
        return head + " No scheduler decision recorded for tonight.", False
    if plan["image"]:
        planned = "Tonight is planned: %s (mode %s)." % (plan["dso"], plan["mode"])
        disagree = v == "CLOUDY"
        tail = " The forecast said go, but the sky looks cloudy." if disagree else ""
    else:
        planned = "Tonight was skipped on the forecast."
        disagree = v == "CLEAR"
        tail = (" But the sky looks clear. Worth a manual start?" if disagree else "")
    return "%s\n%s%s" % (head, planned, tail), disagree


def _post(message, image=None):
    import requests
    data = {"message": message}
    if image:
        data["image_path"] = str(image)
    requests.post(POST_URL, data=data, timeout=20).raise_for_status()


def run(evening=None, post=True, force=False):
    """One evaluation. Returns the logged row, or None if it is not time yet."""
    s = _settings()
    live = evening is None
    evening = evening or datetime.now(_tz()).date()
    if live and not force:
        try:
            if MARKER.exists() and MARKER.read_text().strip() == evening.isoformat():
                return None
        except OSError:
            pass
    frames = evening_frames(evening)
    want = int(s["dusk_frames"])
    if not frames:
        return None
    last_sun = frames[-1][1].get("sun_alt_deg")
    gave_up = last_sun is not None and last_sun <= float(s["dusk_give_up_sun_deg"])
    if len(frames) < want and not gave_up:
        return None
    frames = frames[:want]

    reading = read_sky(frames, s)
    row = {"evening": evening.isoformat(),
           "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if "error" in reading:
        row.update(reading, verdict=None)
        message, disagree = "Dusk sky check: no reading (%s)." % reading["error"], False
    else:
        v = verdict(reading, s)
        plan = tonight_plan(evening) if live else None
        row.update(reading, verdict=v, plan=plan)
        message, disagree = compose(reading, v, plan)
        row["disagree"] = disagree

    print(message)
    if live:
        # Marker BEFORE posting, as in sky_monitor's dawn report: a post that
        # crashes must not be retried every five minutes all night.
        try:
            MARKER.parent.mkdir(parents=True, exist_ok=True)
            MARKER.write_text(evening.isoformat())
        except OSError:
            pass
        try:
            with DUSK_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError as exc:
            print("dusk_check: could not log (%s)" % type(exc).__name__)
    if post:
        frame = None
        name = frames[-1][1].get("captured") or frames[-1][1].get("generated")
        try:
            t = datetime.fromisoformat(name).astimezone(timezone.utc)
            cand = ROOT / "local" / "sky_frames" / ("sky_%s.jpg" % t.strftime("%Y%m%dT%H%M%SZ"))
            frame = cand if cand.exists() else None
        except (TypeError, ValueError):
            pass
        try:
            _post(message, frame)
        except Exception as exc:
            print("dusk_check: chat post failed (%s)" % type(exc).__name__)
        if disagree:
            try:
                from utils import pushover
                pushover.push_message(message)
            except Exception as exc:
                print("dusk_check: push failed (%s)" % type(exc).__name__)
    return row


def step():
    """Called by sky_monitor on every night frame. Never raises."""
    try:
        run()
    except Exception as exc:
        print("dusk_check: failed (ignored): %s: %s" % (type(exc).__name__, exc))


def main(argv):
    if "--date" in argv:
        d = date.fromisoformat(argv[argv.index("--date") + 1])
        row = run(evening=d, post=False)
        if row is None:
            print("no dusk frames for %s" % d)
            return 1
        return 0
    row = run(post="--no-post" not in argv, force="--force" in argv)
    if row is None:
        print("not dusk yet, or tonight already checked (use --force)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
