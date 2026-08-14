#!/usr/bin/env python3
"""Does the all-sky camera predict the seeing the telescope actually gets?

Joins the sky camera's per-frame measurements (local/sky_log.jsonl) to the
telescope's measured FWHM (<dso_dir>/frame_stats.json, the `stats` command's
cache) and correlates them. Nothing is re-measured; both sides are already on
disk.

    python scripts/seeing_vs_skycam.py [--dso NAME] [--filter Ha] [--block-min 30]

Run it again as nights accumulate. As of 2026-08-14 exactly ONE night overlaps
(bubble, 50 frames, 2026-08-09), because sky_log.jsonl only starts 2026-08-08 --
so the between-night answer, which is the one that matters for scheduling, has
n=1 and cannot be computed at all. This script exists so that stops being true
without anyone having to redo the analysis.

WHY THIS IS NOT A SIMPLE CORRELATION
------------------------------------
Seeing drifts slowly. On 2026-08-09 the lag-1 autocorrelation of FWHM between
frames 5 minutes apart was +0.86, so 50 frames carry nothing like 50 independent
samples. Correlating raw frames inflates every coefficient and makes noise look
like signal: sky background scored rho=+0.41 that way, and fell to r=+0.09,
p=0.82 once the same data was aggregated into 30-minute blocks. Percent-of-stars
went the other way and held up (r=-0.56, p=0.10). So the raw-frame number is not
reported at all here except as the autocorrelation diagnostic that discredits it.

Two genuinely different questions, reported separately:

  * WITHIN-NIGHT -- does the camera track seeing as it changes during a night?
    Block medians, each night centred on its own mean so a night that was simply
    good overall cannot masquerade as a within-night trend.
  * BETWEEN-NIGHT -- does the camera predict which nights will be good? One
    point per night. This is the one worth having, because it is the one that
    could inform scheduling, and it is also the slowest to accumulate.

A KNOWN BIAS, stated because it flatters the result
---------------------------------------------------
`stars_matched` / `stars_expected` is absent from rows where the plate solve was
too poor to verify -- which is exactly what happens under heavy cloud. So the
worst skies silently drop out of the percent-of-stars column while remaining in
the sky-background column. That censoring removes the samples where the
relationship should be strongest, so a weak percent-of-stars correlation here is
a lower bound, and the two predictors are not being scored on identical data.
Per-predictor n is printed for that reason.

`median_fwhm_px` from the camera is included as a CONTROL that should stay near
zero. At 1617 px/rad the plate scale is ~128 arcsec/pixel, so 2-arcsecond seeing
is under 1/60th of a pixel and the camera cannot see it even in principle. If
that column ever comes back strongly correlated, the join is wrong, not the
physics.

With ~7 predictors tested, treat a lone p<0.05 as a lead rather than a result --
the same caution seeing_vs_weather.py carries, and for the same reason.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import permutations
from zoneinfo import ZoneInfo

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

SKY_LOG = "local/sky_log.jsonl"
MATCH_WINDOW_S = 300          # the sky camera's own cadence

# label -> (sky_log key, "higher means better seeing?")
PREDICTORS = [
    ("pct stars",    "pct_stars",       "more stars = sharper"),
    ("star count",   "stars",           "more stars = sharper"),
    ("limiting mag", "limiting_mag",    "deeper = sharper"),
    ("sky ADU",      "sky_median_adu",  "brighter = softer"),
    ("frame level",  "frame_level_adu", "brighter = softer"),
    ("purity",       "purity",          "purer = sharper"),
    ("cam fwhm px",  "median_fwhm_px",  "CONTROL, expect ~0"),
]


def load_frames(image_dir, dso=None, filt=None):
    """Telescope frames with a measured FWHM, from every target's stats cache."""
    out = []
    pattern = os.path.join(image_dir, dso or "*", "frame_stats.json")
    for path in glob.glob(pattern):
        target = os.path.basename(os.path.dirname(path))
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError) as e:
            print("skipping %s: %s" % (path, e), file=sys.stderr)
            continue
        for r in rows:
            if not (r.get("time") and r.get("fwhm_arcsec")):
                continue
            if filt and r.get("filter") != filt:
                continue
            t = datetime.fromisoformat(r["time"])
            if t.tzinfo is None:
                # frame_stats stores UTC without an offset.
                t = t.replace(tzinfo=timezone.utc)
            out.append({"t": t, "target": target, "fwhm": r["fwhm_arcsec"],
                        "filter": r.get("filter")})
    out.sort(key=lambda r: r["t"])
    return out


def load_sky():
    rows = []
    for line in open(SKY_LOG, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        t = datetime.fromisoformat(s["generated"])
        exp, mat = s.get("stars_expected"), s.get("stars_matched")
        s["pct_stars"] = (100.0 * mat / exp) if (exp and mat is not None) else None
        rows.append((t, s))
    rows.sort(key=lambda x: x[0])
    return rows


def join(frames, sky):
    """Nearest sky row within MATCH_WINDOW_S. Unmatched frames are dropped."""
    if not sky:
        return []
    times = [t for t, _ in sky]
    out = []
    for fr in frames:
        i = np.searchsorted(times, fr["t"])
        best, gap = None, None
        for j in (i - 1, i):
            if 0 <= j < len(sky):
                d = abs((times[j] - fr["t"]).total_seconds())
                if gap is None or d < gap:
                    best, gap = sky[j][1], d
        if best is not None and gap <= MATCH_WINDOW_S:
            rec = {"t": fr["t"], "fwhm": fr["fwhm"], "target": fr["target"],
                   "filter": fr["filter"], "gap_s": gap}
            for _, key, _ in PREDICTORS:
                rec[key] = best.get(key)
            out.append(rec)
    return out


def night_key(t, tz):
    """Local date of the evening a timestamp belongs to.

    Shifted by 12 hours so a run spanning local midnight stays one night, which
    is the whole point -- keying on the calendar date would split every session
    in two and halve the between-night sample.
    """
    return (t.astimezone(tz) - timedelta(hours=12)).date()


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def perm_p(x, y, iters=20000, seed=0):
    """Exact over all orderings when small, sampled when not."""
    r0 = pearson(x, y)
    if r0 is None:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n <= 8:
        perms = list(permutations(range(n)))
        hits = sum(1 for p in perms if abs(np.corrcoef(x[list(p)], y)[0, 1]) >= abs(r0))
        return hits / len(perms)
    rng = np.random.default_rng(seed)
    hits = sum(1 for _ in range(iters)
               if abs(np.corrcoef(rng.permutation(x), y)[0, 1]) >= abs(r0))
    return hits / iters


def autocorr_report(recs):
    by_night = defaultdict(list)
    for r in recs:
        by_night[r["night"]].append(r["fwhm"])
    print("\nFWHM autocorrelation (why raw frames must not be counted as independent)")
    for night in sorted(by_night):
        f = np.asarray(by_night[night], float)
        if len(f) < 12:
            print("  %s  n=%-3d  (too short)" % (night, len(f)))
            continue
        lags = "  ".join("lag-%d %+.2f" % (k, np.corrcoef(f[:-k], f[k:])[0, 1])
                         for k in (1, 2, 5))
        print("  %s  n=%-3d  %s" % (night, len(f), lags))


def table(title, note, rows, series_of):
    print("\n%s" % title)
    print("  %s" % note)
    print("  %-13s %8s %8s %6s   %s" % ("predictor", "r", "p", "n", "expected sign"))
    for label, key, expect in PREDICTORS:
        xs, ys = series_of(key)
        if xs is None or len(xs) < 3:
            print("  %-13s %8s %8s %6s   %s"
                  % (label, "--", "--", len(xs) if xs is not None else 0, expect))
            continue
        r = pearson(xs, ys)
        if r is None:
            print("  %-13s %8s %8s %6d   %s" % (label, "flat", "--", len(xs), expect))
            continue
        p = perm_p(xs, ys)
        print("  %-13s %+8.3f %8.3f %6d   %s" % (label, r, p, len(xs), expect))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dso", help="restrict to one target directory")
    ap.add_argument("--filter", dest="filt",
                    help="restrict to one FITS FILTER; FWHM differs between "
                         "filters, so mixing them adds variance the join "
                         "cannot remove")
    ap.add_argument("--block-min", type=int, default=30,
                    help="within-night block size in minutes (default 30)")
    args = ap.parse_args(argv)

    cfg = config.data()
    tz = ZoneInfo(cfg["location"]["timezone"])
    image_dir = cfg["nina"]["image_dir"]

    frames = load_frames(image_dir, args.dso, args.filt)
    sky = load_sky()
    if not frames or not sky:
        print("nothing to join: %d frames, %d sky rows" % (len(frames), len(sky)))
        return 1

    recs = join(frames, sky)
    for r in recs:
        r["night"] = night_key(r["t"], tz)

    print("telescope frames with FWHM : %d  (%s .. %s)"
          % (len(frames), frames[0]["t"].date(), frames[-1]["t"].date()))
    print("sky camera rows            : %d  (%s .. %s)"
          % (len(sky), sky[0][0].date(), sky[-1][0].date()))
    print("matched within %ds         : %d" % (MATCH_WINDOW_S, len(recs)))
    if not recs:
        print("\nNo overlap yet. The sky camera log has to cover an imaging "
              "night before there is anything to correlate.")
        return 0

    by_night = defaultdict(list)
    for r in recs:
        by_night[r["night"]].append(r)
    print("nights with overlap        : %d" % len(by_night))
    print("\n  %-12s %5s  %-10s %-8s %s"
          % ("night", "n", "target", "filters", "median FWHM"))
    for night in sorted(by_night):
        v = by_night[night]
        print("  %-12s %5d  %-10s %-8s %.2f\""
              % (night, len(v), v[0]["target"],
                 ",".join(sorted({x["filter"] or "?" for x in v})),
                 float(np.median([x["fwhm"] for x in v]))))

    autocorr_report(recs)

    # ---- within-night: block medians, each night centred on its own mean
    blocks = defaultdict(list)
    for r in recs:
        base = min(x["t"] for x in by_night[r["night"]])
        k = int((r["t"] - base).total_seconds() // (args.block_min * 60))
        blocks[(r["night"], k)].append(r)

    def within(key):
        per_night = defaultdict(list)
        for (night, _), v in blocks.items():
            xs = [x[key] for x in v if x.get(key) is not None]
            if not xs:
                continue
            per_night[night].append((float(np.median(xs)),
                                     float(np.median([x["fwhm"] for x in v]))))
        X, Y = [], []
        for night, pairs in per_night.items():
            if len(pairs) < 3:      # a night with one block says nothing here
                continue
            xm = np.mean([p[0] for p in pairs])
            ym = np.mean([p[1] for p in pairs])
            X += [p[0] - xm for p in pairs]
            Y += [p[1] - ym for p in pairs]
        return (X, Y) if X else (None, None)

    table("WITHIN-NIGHT (does it track seeing changing during a night?)",
          "%d-minute block medians, each night centred on its own mean"
          % args.block_min, recs, within)

    # ---- between-night: one point per night
    def between(key):
        X, Y = [], []
        for night, v in sorted(by_night.items()):
            xs = [x[key] for x in v if x.get(key) is not None]
            if not xs:
                continue
            X.append(float(np.median(xs)))
            Y.append(float(np.median([x["fwhm"] for x in v])))
        return (X, Y) if X else (None, None)

    table("BETWEEN-NIGHT (does it predict which nights are good?)",
          "one point per night -- the answer that would inform scheduling",
          recs, between)

    if len(by_night) < 5:
        print("\n  Only %d night(s) overlap. The between-night table above is "
              "not an answer yet;\n  it needs roughly 8-10 nights before a "
              "permutation test can say anything." % len(by_night))
    return 0


if __name__ == "__main__":
    sys.exit(main())
