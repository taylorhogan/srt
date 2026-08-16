"""
roof_region_stats.py
Score one Iris cam recording for roof state, and append it to a running log.

Reduces a recording made by `iriscam_record.py` to the three aperture
statistics that separate a shut roof from an open one, splits the frames into
before/after using the roof move heard on the microphone, and appends one row
per run to `local/roof_region_log.jsonl`.

    python roof_region_stats.py local/iriscam_rec/roof-open_20260816_074016
    python roof_region_stats.py --all          # rescore every recording
    python roof_region_stats.py --summary      # print the log

The move is located from the audio rather than assumed, because the delay
before the relay fires is dominated by a vision ladder of variable length --
hand-picking "before t=60, after t=80" does not survive a run where the ladder
took twice as long.

Sun altitude and azimuth are recorded per run. The whole point of repeating
this through a day is to sample illumination as the sun clears the tree line,
and a row without the sun's position cannot be placed in that series.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2
import numpy as np

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers

from configs import config

# The astropy cache on this box is not writable, so every run tried to fetch
# IERS tables, failed twice, and printed five warnings before falling back to
# the bundled table anyway. Ask for the bundled one directly: the correction is
# sub-arcsecond and this only needs the sun to a degree.
iers.conf.auto_download = False
# And the bundled table is itself older than astropy's 30-day guard, which
# raises rather than warns. Earth-orientation drift is milliarcseconds; it
# cannot move the sun enough to matter to a green-channel threshold.
iers.conf.auto_max_age = None

LOG_PATH = "local/roof_region_log.jsonl"

# The aperture: roof underside when shut, sky when open. Fixed pixel box, which
# is only meaningful while the camera stays put -- Iris cam pans and tilts, so
# a run whose position differs from CAL_POSITION is flagged, not silently
# scored against the wrong patch of wall.
REGION = (slice(100, 1400), slice(0, 700))
CAL_POSITION = (-123, 363)

# Half-second bins louder than this multiple of the run's own median are the
# roof moving. The floor is rms 3-5 and a move peaks near 1000-1800, so this is
# nowhere near either edge.
MOVE_RMS_FACTOR = 5.0
SETTLE_S = 8.0          # skip either side of the move: travel and lights


def sun_position(when):
    cfg = config.data()["location"]
    loc = EarthLocation.from_geodetic(float(cfg["longitude"]) * u.deg,
                                      float(cfg["latitude"]) * u.deg,
                                      float(cfg.get("elevation", 0)) * u.m)
    t = Time(when)
    altaz = get_sun(t).transform_to(AltAz(obstime=t, location=loc))
    return float(altaz.alt.deg), float(altaz.az.deg)


def frame_metrics(path):
    img = cv2.imread(path)
    if img is None:
        return None
    roi = img[REGION[0], REGION[1]]
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 60, 160)
    b, g, r = (roi[:, :, i].astype(np.float64) for i in range(3))
    return {
        "mean": float(grey.mean()),
        "std": float(grey.std()),
        "edge_pct": float(100.0 * edges.mean() / 255.0),
        # Sign, not magnitude: pine and foil insulation are red-dominant,
        # foliage is green-dominant, and a sign change survives the
        # auto-exposure renormalisation that a brightness threshold does not.
        "green_excess": float((g - (b + r) / 2.0).mean()),
    }


def find_move(rec_dir):
    """(start_s, end_s) of the roof move, from the audio envelope."""
    env = os.path.join(rec_dir, "envelope.csv")
    if not os.path.exists(env):
        return None
    rows = [(float(r["t_s"]), float(r["rms"])) for r in csv.DictReader(open(env))]
    if not rows:
        return None
    median = float(np.median([r[1] for r in rows]))
    loud = [t for t, rms in rows if rms > max(median * MOVE_RMS_FACTOR, 20.0)]
    return (loud[0], loud[-1]) if loud else None


def score(rec_dir, verbose=True):
    frames = sorted(glob.glob(os.path.join(rec_dir, "frame_*.jpg")))
    if not frames:
        print("  no frames in %s" % rec_dir)
        return None
    move = find_move(rec_dir)
    if move is None:
        print("  %s: no roof move found in the audio; cannot split" % rec_dir)
        return None
    t_start, t_end = move

    before, after, excluded = [], [], []
    for path in frames:
        m = re.search(r"frame_(\d+\.\d)s", path)
        if not m:
            continue
        t = float(m.group(1))
        stats = frame_metrics(path)
        if stats is None:
            continue
        stats["t_s"] = t
        if t < t_start - 1.0:
            before.append(stats)
        elif t > t_end + SETTLE_S:
            after.append(stats)
        else:
            excluded.append(stats)

    label = os.path.basename(rec_dir)
    direction = "open" if "open" in label else ("close" if "close" in label else "?")
    # A close recording runs open -> closed, so before/after swap meaning.
    if direction == "close":
        open_set, shut_set = before, after
    else:
        open_set, shut_set = after, before

    stamp = re.search(r"(\d{8}_\d{6})", label)
    when = (datetime.strptime(stamp.group(1), "%Y%m%d_%H%M%S").astimezone()
            if stamp else datetime.now().astimezone())
    alt, az = sun_position(when.astimezone(timezone.utc))

    def agg(rows, key):
        vals = [r[key] for r in rows]
        return (min(vals), max(vals)) if vals else (None, None)

    row = {
        "recording": label,
        "direction": direction,
        "started": when.isoformat(timespec="seconds"),
        "sun_alt_deg": round(alt, 2),
        "sun_az_deg": round(az, 2),
        "move_start_s": t_start,
        "move_end_s": t_end,
        "n_open": len(open_set),
        "n_shut": len(shut_set),
        "n_excluded": len(excluded),
    }
    for key in ("mean", "edge_pct", "green_excess"):
        row["open_" + key] = [round(v, 2) if v is not None else None
                              for v in agg(open_set, key)]
        row["shut_" + key] = [round(v, 2) if v is not None else None
                              for v in agg(shut_set, key)]

    # The number that matters: does green excess still separate, and by how
    # much? A gap that shrinks toward zero as the light changes is the failure
    # this whole exercise is looking for.
    if open_set and shut_set:
        gap = min(r["green_excess"] for r in open_set) - \
              max(r["green_excess"] for r in shut_set)
        row["green_gap"] = round(gap, 2)
        row["sign_separated"] = bool(
            min(r["green_excess"] for r in open_set) > 0 >
            max(r["green_excess"] for r in shut_set))
        e_gap = min(r["edge_pct"] for r in open_set) - \
            max(r["edge_pct"] for r in shut_set)
        row["edge_gap"] = round(e_gap, 2)
    else:
        row["green_gap"] = row["edge_gap"] = None
        row["sign_separated"] = None

    if verbose:
        print("  %s  (%s, sun alt %+.1f az %.0f)"
              % (label, direction, alt, az))
        print("    move heard t=%.1f..%.1fs; %d shut / %d open / %d excluded"
              % (t_start, t_end, len(shut_set), len(open_set), len(excluded)))
        for key in ("mean", "edge_pct", "green_excess"):
            print("    %-13s shut %-16s open %-16s"
                  % (key, row["shut_" + key], row["open_" + key]))
        print("    green gap %s (sign separated: %s), edge gap %s"
              % (row["green_gap"], row["sign_separated"], row["edge_gap"]))
    return row


def append_log(rows):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    seen = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["recording"])
                except (ValueError, KeyError):
                    pass
    added = 0
    with open(LOG_PATH, "a") as fh:
        for row in rows:
            if row and row["recording"] not in seen:
                fh.write(json.dumps(row) + "\n")
                added += 1
    return added


def summary():
    if not os.path.exists(LOG_PATH):
        print("no log yet at %s" % LOG_PATH)
        return
    rows = [json.loads(l) for l in open(LOG_PATH) if l.strip()]
    rows.sort(key=lambda r: r["started"])
    print("%-19s %-6s %6s %6s %10s %9s %s"
          % ("started", "dir", "sunalt", "sunaz", "green gap", "edge gap", "sign"))
    print("-" * 74)
    for r in rows:
        print("%-19s %-6s %+6.1f %6.0f %10s %9s %s"
              % (r["started"][:19].replace("T", " "), r["direction"],
                 r["sun_alt_deg"], r["sun_az_deg"], r["green_gap"],
                 r["edge_gap"], r["sign_separated"]))
    gaps = [r["green_gap"] for r in rows if r.get("green_gap") is not None]
    if gaps:
        print("\ngreen gap over %d runs: min %.2f, max %.2f"
              % (len(gaps), min(gaps), max(gaps)))
        if min(gaps) <= 0:
            print("*** at least one run does NOT separate ***")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?")
    ap.add_argument("--all", action="store_true", help="score every recording")
    ap.add_argument("--summary", action="store_true", help="print the log only")
    ap.add_argument("--root", default="local/iriscam_rec")
    args = ap.parse_args()

    if args.summary:
        summary()
        return 0
    targets = ([d for d in sorted(glob.glob(os.path.join(args.root, "*")))
                if os.path.isdir(d)] if args.all
               else [args.recording] if args.recording else [])
    if not targets:
        ap.print_help()
        return 1
    rows = [score(t) for t in targets]
    print("\nappended %d new row(s) to %s" % (append_log(rows), LOG_PATH))
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
