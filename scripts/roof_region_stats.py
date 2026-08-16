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

# The aperture: roof underside when shut, sky when open.
REGION_Y = (100, 1400)
REGION_X = (0, 700)
CAL_POSITION = (-123, 363)

# The pixel box above is only meaningful if the camera is pointing where it was
# when the box was chosen, and on this camera the encoder does NOT establish
# that. Measured 2026-08-16: frames at the identical reading (-123, 363) sit up
# to 121 px apart horizontally and 55-76 px vertically. Against a 700 px-wide
# aperture that is 17% of the region, i.e. enough to score the wrong wall.
#
# So every frame is registered against a stored reference before it is scored,
# by template-matching a patch of ceiling that does not change with roof state,
# and the aperture box is shifted by the measured offset. A frame that will not
# register is UNKNOWN, never a state.
REFERENCE_PATH = "local/roof_region_reference.jpg"
REFERENCE_META = "local/roof_region_reference.json"
# Band the patches are drawn from: fixed ceiling planking on the far side of
# the frame from the aperture, below the flat panel. The panel is excluded on
# purpose -- it is bright, near the aperture, and its appearance changes
# completely when the roof opens, so a patch on its edge matches badly exactly
# when the answer matters.
FIDUCIAL_BAND = ((250, 1000), (1750, 2450))
PATCH_PX = 160
SEARCH_PX = 260
N_PATCHES = 6

# One patch is not enough. A single auto-picked patch reported shifts up to
# 176 px WITHIN one recording, during which the camera never moved, and still
# scored 0.71 -- so the match score alone does not catch a false match. Several
# patches are matched independently and the median is taken; they are trusted
# only when they agree, because independent false matches do not agree.
MIN_MATCH_SCORE = 0.50
MAX_PATCH_SPREAD_PX = 40    # disagreement between patches => unknown
MIN_PATCHES_AGREE = 4
MAX_SHIFT_PX = 250          # beyond this the aperture box leaves its subject

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


def make_reference(frame_path, position=None):
    """Adopt *frame_path* as the framing every later run is measured against."""
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("cannot read %s" % frame_path)
    (y0, y1), (x0, x1) = FIDUCIAL_BAND
    band = img[y0:y1, x0:x1]
    # Pick the most distinctive spot automatically rather than hardcoding a
    # corner: a hand-picked patch on plain planking matches everywhere.
    corners = cv2.goodFeaturesToTrack(band, maxCorners=N_PATCHES * 3,
                                      qualityLevel=0.01, minDistance=140,
                                      blockSize=9)
    if corners is None:
        raise SystemExit("no distinctive feature in the fiducial band")
    half = PATCH_PX // 2
    anchors = []
    for c in corners:
        cx, cy = (int(v) for v in c[0])
        ax, ay = x0 + cx - half, y0 + cy - half
        # Keep the whole patch inside the band, so no patch can straddle the
        # panel or the aperture.
        if ax >= x0 and ay >= y0 and ax + PATCH_PX <= x1 and ay + PATCH_PX <= y1:
            anchors.append([ax, ay])
        if len(anchors) >= N_PATCHES:
            break
    if len(anchors) < MIN_PATCHES_AGREE:
        raise SystemExit("only %d usable patches in the fiducial band"
                         % len(anchors))
    cv2.imwrite(REFERENCE_PATH, cv2.imread(frame_path))
    meta = {"source": frame_path, "anchors": anchors,
            "patch_px": PATCH_PX, "position": list(position or CAL_POSITION),
            "created": datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(REFERENCE_META, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("reference set from %s; %d patches at %s"
          % (frame_path, len(anchors), anchors))
    return meta


def _reference():
    if not (os.path.exists(REFERENCE_PATH) and os.path.exists(REFERENCE_META)):
        return None, None, None
    meta = json.load(open(REFERENCE_META))
    ref = cv2.imread(REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)
    p = meta["patch_px"]
    templates = [ref[ay:ay + p, ax:ax + p] for ax, ay in meta["anchors"]]
    return ref, meta, templates


def register(img_grey, meta, templates):
    """(dx, dy, worst_score) by median over several patches, or None.

    Returns None when the patches disagree. That is the whole point of using
    more than one: a false match is confident but arbitrary, so independent
    false matches scatter, while true matches all report the same offset.
    """
    p = meta["patch_px"]
    votes = []
    for (ax, ay), tpl in zip(meta["anchors"], templates):
        sx, sy = max(0, ax - SEARCH_PX), max(0, ay - SEARCH_PX)
        window = img_grey[sy:ay + p + SEARCH_PX, sx:ax + p + SEARCH_PX]
        if window.shape[0] < p or window.shape[1] < p:
            continue
        res = cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= MIN_MATCH_SCORE:
            votes.append((sx + loc[0] - ax, sy + loc[1] - ay, float(score)))
    if len(votes) < MIN_PATCHES_AGREE:
        return None
    mx = float(np.median([v[0] for v in votes]))
    my = float(np.median([v[1] for v in votes]))
    agree = [v for v in votes
             if abs(v[0] - mx) <= MAX_PATCH_SPREAD_PX
             and abs(v[1] - my) <= MAX_PATCH_SPREAD_PX]
    if len(agree) < MIN_PATCHES_AGREE:
        return None
    return (int(round(np.median([v[0] for v in agree]))),
            int(round(np.median([v[1] for v in agree]))),
            min(v[2] for v in agree))


def frame_metrics(path, meta=None, templates=None):
    img = cv2.imread(path)
    if img is None:
        return None
    y0, y1 = REGION_Y
    x0, x1 = REGION_X
    shift = None
    if meta is not None and templates:
        shift = register(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), meta, templates)
        if shift is None or shift[2] < MIN_MATCH_SCORE:
            return {"unregistered": True, "match_score":
                    None if shift is None else round(shift[2], 3)}
        dx, dy, _ = shift
        if abs(dx) > MAX_SHIFT_PX or abs(dy) > MAX_SHIFT_PX:
            return {"unregistered": True, "shift": [dx, dy],
                    "match_score": round(shift[2], 3)}
        # Follow the camera: the aperture is wherever the fiducial says it is.
        y0, y1, x0, x1 = y0 + dy, y1 + dy, x0 + dx, x1 + dx
        h, w = img.shape[:2]
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return {"unregistered": True, "match_score": None}
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 60, 160)
    b, g, r = (roi[:, :, i].astype(np.float64) for i in range(3))
    return {
        "unregistered": False,
        "shift": None if shift is None else [shift[0], shift[1]],
        "match_score": None if shift is None else round(shift[2], 3),
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

    ref, meta, templates = _reference()
    if meta is None:
        print("  NO REFERENCE -- run --set-reference first; the aperture box is")
        print("  only valid for one camera pointing and this camera moves.")
        return None

    before, after, excluded, unregistered = [], [], [], []
    for path in frames:
        m = re.search(r"frame_(\d+\.\d)s", path)
        if not m:
            continue
        t = float(m.group(1))
        stats = frame_metrics(path, meta, templates)
        if stats is None:
            continue
        stats["t_s"] = t
        if stats.get("unregistered"):
            unregistered.append(stats)
            continue
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
        "n_unregistered": len(unregistered),
    }
    registered = [s for s in before + after + excluded if s.get("shift")]
    if registered:
        row["shift_x"] = [min(s["shift"][0] for s in registered),
                          max(s["shift"][0] for s in registered)]
        row["shift_y"] = [min(s["shift"][1] for s in registered),
                          max(s["shift"][1] for s in registered)]
        row["match_score_min"] = min(s["match_score"] for s in registered)
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
        # Track the OPEN side's own margin above zero, not just the gap. On
        # 2026-08-16 the gap held at ~5.6 while this fell from +2.18 to +0.39
        # as the sun rose, because the shut side moved further negative to
        # compensate. Sign separation depends on this number alone.
        row["open_green_min"] = round(min(r["green_excess"] for r in open_set), 2)
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
        if registered:
            print("    registration: shift x %s y %s, worst score %.3f"
                  % (row["shift_x"], row["shift_y"], row["match_score_min"]))
        if unregistered:
            print("    *** %d frame(s) would NOT register -- excluded as unknown"
                  % len(unregistered))
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
    print("%-19s %-6s %6s %6s %10s %10s %9s %s"
          % ("started", "dir", "sunalt", "sunaz", "green gap", "open min",
             "edge gap", "sign"))
    print("-" * 85)
    for r in rows:
        print("%-19s %-6s %+6.1f %6.0f %10s %10s %9s %s"
              % (r["started"][:19].replace("T", " "), r["direction"],
                 r["sun_alt_deg"], r["sun_az_deg"], r["green_gap"],
                 r.get("open_green_min", "-"),
                 r["edge_gap"], r["sign_separated"]))
    gaps = [r["green_gap"] for r in rows if r.get("green_gap") is not None]
    if gaps:
        print("\ngreen gap over %d runs: min %.2f, max %.2f"
              % (len(gaps), min(gaps), max(gaps)))
        if min(gaps) <= 0:
            print("*** at least one run does NOT separate ***")
    mins = [r["open_green_min"] for r in rows if r.get("open_green_min") is not None]
    if mins:
        print("open-side margin above zero: min %.2f, max %.2f%s"
              % (min(mins), max(mins),
                 "   *** approaching zero ***" if min(mins) < 0.5 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?")
    ap.add_argument("--all", action="store_true", help="score every recording")
    ap.add_argument("--summary", action="store_true", help="print the log only")
    ap.add_argument("--set-reference", metavar="FRAME",
                    help="adopt FRAME as the framing all runs are measured "
                         "against (do this once, with the roof SHUT)")
    ap.add_argument("--root", default="local/iriscam_rec")
    args = ap.parse_args()

    if args.set_reference:
        make_reference(args.set_reference)
        return 0
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
