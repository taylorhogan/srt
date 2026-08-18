"""
sky_transparency_report.py
Plot what fraction of the catalogue the sky camera could actually see, over a
night, and post it to the observatory feed.

Run at sunrise. The sky monitor already records, every five minutes, how many
catalogue stars land on real detections (``solve_matches``). On its own that is
an absolute count and hard to read: it falls when cloud rolls in, but it also
falls simply because the sky rotates and a different, sparser patch drifts into
the camera's field. Dividing by the number of catalogue stars actually IN the
field at that moment removes the second effect and leaves transparency.

    python scripts/sky_transparency_report.py            # last night
    python scripts/sky_transparency_report.py --date 2026-08-16
    python scripts/sky_transparency_report.py --no-post  # plot only

The denominator counts catalogue stars inside the camera's field brighter than
MAG_LIMIT. That cut is not cosmetic: without it the denominator includes stars
this camera physically cannot reach, and a perfect night reads 30% instead of
94%. Calibrated on the best frame in the log -- 187 matches of 271 detections
on 2026-08-16 04:19 -- where the match fraction reaches 94% at mag 5.0 and
falls away below it, which is the signature of a real detection limit rather
than of weather.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
from astropy.utils import iers

# The astropy cache on this box is not writable, so a bare import tries to
# fetch IERS tables, fails, and warns four times per run before falling back.
# Earth-orientation drift is milliarcseconds and cannot move a star out of a
# 2560 px field.
iers.conf.auto_download = False
iers.conf.auto_max_age = None

from configs import config
from iris_astronomy import bright_stars
from sentry import plate_solve as ps

LOG_PATH = "local/sky_log.jsonl"
OUT_PATH = "local/sky_transparency.png"
MAG_LIMIT = 5.0
FRAME_W, FRAME_H = 2560, 1440
POST_URL = "http://127.0.0.1:8095/api/post"


def in_field(sol, when, mag_limit=MAG_LIMIT):
    """Catalogue stars inside the camera's field, brighter than the limit."""
    alt, az, vmag, _ra, _dec = bright_stars.above_horizon(when)
    x, y, th = ps.cam_to_pix(
        ps.unit_from_altaz(az, alt) @ np.asarray(sol["R"]).T,
        sol["cx"], sol["cy"], sol["F"], sol["model"], sol["flip"])
    inside = ((x > 20) & (x < FRAME_W - 20) & (y > 20) & (y < FRAME_H - 20)
              & (th < 1.45) & (vmag <= mag_limit))
    return int(inside.sum())


def night_rows(rows, date=None):
    """Rows from one night, keyed on the evening it began.

    A night spans midnight, so it cannot be selected by calendar date. Rows are
    grouped by the date twelve hours earlier, so a dusk-to-dawn run is filed under
    the evening it started -- "the night of the 15th" in the ordinary sense.
    """
    out = {}
    for r in rows:
        t = r.get("generated")
        if not t or r.get("solve_matches") is None:
            continue
        try:
            when = datetime.fromisoformat(t).astimezone()
        except ValueError:
            continue
        if not r.get("night"):
            continue
        # Naive LOCAL time, not tz-aware. matplotlib renders tz-aware
        # datetimes in UTC regardless of their offset, which put the axis four
        # hours away from the local times printed in this script's own summary
        # -- a chart captioned 20:54-05:04 with an axis reading 01:00-09:00.
        local = when.replace(tzinfo=None)
        out.setdefault((local - timedelta(hours=12)).date(), []).append((local, r))
    if not out:
        return None, []
    key = date or max(out)
    return key, sorted(out.get(key, []), key=lambda p: p[0])


def build(date=None, mag_limit=MAG_LIMIT):
    sol = ps.load()
    if sol is None:
        return None, "no plate solution stored; run sentry/plate_solve.py <frame> --save"
    rows = [json.loads(l) for l in open(LOG_PATH) if l.strip()]
    key, night = night_rows(rows, date)
    if not night:
        return None, "no night-time rows with a solve in %s" % LOG_PATH

    times, pcts, matches, totals = [], [], [], []
    for when, r in night:
        # in_field must be given an ABSOLUTE time: bright_stars.above_horizon
        # would read a naive datetime as UTC and rotate the sky by the offset,
        # moving every catalogue star out of the field.
        n = in_field(sol, when.astimezone(), mag_limit)
        if n <= 0:
            continue
        times.append(when)
        matches.append(r["solve_matches"])
        totals.append(n)
        pcts.append(100.0 * r["solve_matches"] / n)
    if not pcts:
        return None, "no usable samples for %s" % key

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(times, pcts, color="#4cc9f0", lw=1.4, zorder=3)
    ax.scatter(times, pcts, s=14, color="#4cc9f0", zorder=4)
    med = float(np.median(pcts))
    ax.axhline(med, color="white", ls="--", lw=1, alpha=0.7)
    ax.text(times[0], med, " median %.0f%%" % med, va="bottom",
            color="white", fontsize=9)
    ax.set_ylim(0, max(100.0, max(pcts) * 1.1))
    ax.set_ylabel("catalogue stars seen (%%)\nof those in frame brighter than mag %.1f"
                  % mag_limit)
    ax.set_xlabel("Time (local)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_title("Sky transparency, night of %s   %d samples" % (key, len(pcts)),
                 color="white")

    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    plt.tight_layout()
    out = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

    best, worst = max(pcts), min(pcts)
    summary = (
        "Sky transparency, night of %s\n"
        "  median %.0f%% of catalogue stars seen (best %.0f%%, worst %.0f%%)\n"
        "  %d samples, %d-%d catalogue stars in frame brighter than mag %.1f\n"
        "  %s - %s local"
        % (key, med, best, worst, len(pcts), min(totals), max(totals), mag_limit,
           times[0].strftime("%H:%M"), times[-1].strftime("%H:%M")))
    return out, summary


def post(image_path, message):
    import requests
    with open(image_path, "rb") as fh:
        r = requests.post(POST_URL, data={"message": message},
                          files={"image": (os.path.basename(image_path), fh, "image/png")},
                          timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="night ending on this YYYY-MM-DD (default: latest)")
    ap.add_argument("--mag-limit", type=float, default=MAG_LIMIT)
    ap.add_argument("--no-post", action="store_true")
    args = ap.parse_args()

    date = None
    if args.date:
        date = datetime.strptime(args.date, "%Y-%m-%d").date()

    path, summary = build(date, args.mag_limit)
    if path is None:
        print(summary)
        return 1
    print(summary)
    print("wrote %s" % path)
    if args.no_post:
        return 0
    try:
        post(path, summary)
        print("posted to the observatory feed")
    except Exception as exc:
        # The plot is on disk either way; a webchat that is down must not lose
        # the night's measurement.
        print("could not post (%s: %s) -- plot still written"
              % (type(exc).__name__, str(exc)[:120]))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
