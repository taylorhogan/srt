#!/usr/bin/env python3
"""Score every logged cloud forecast against what the sky camera actually saw.

forecast_log.py has logged every Open-Meteo model (plus the NWS grid) hourly
since 2026-08-14, precisely so that "which model should we believe" could be
answered with a number instead of an anecdote. This is the other half: join
those forecasts to the sky camera's measurements and score them.

GROUND TRUTH is transparency, the same quantity sky_transparency_report.py
plots: plate-solve matches divided by the catalogue stars (V <= 5) actually in
the field at that moment. Per UTC hour, the median over that hour's 5-minute
frames. Three things corrupt it, and each is handled rather than ignored:

  * The MOON. A bright moon lowers the star fraction on a perfectly clear sky
    (2026-09-29: 8% all night under an 86% moon, with no way to tell how much
    was cloud). Hours are scored only when the moon is below the horizon or
    less than MOON_MAX_ILLUM lit; --with-moon includes them anyway.
  * TWILIGHT. Only frames with the sun below SUN_MAX_ALT count.
  * The PLATE SOLUTION. matches collapse when the camera drifts off its stored
    solution (0.44 deg over 2026-08-11..24 flattened the grade), which reads
    exactly like cloud, and a solution that has drifted mid-epoch still
    does. Each solution was in force from its solve until the
    archive file named sky_solution_<T>.json was written at T, so every frame
    is scored against the solution it was actually measured with, and
    transparency is expressed relative to the CEILING of its own solution epoch
    (the CEILING_PCTL percentile of that epoch's moon-free hours). A good hour
    in any epoch then reads ~1.0 whatever that epoch's absolute match rate.

A forecast is joined to a truth hour by valid time. Each (model, hour) has up to
24 forecasts at different leads; within each lead bucket only the most recent
issue counts, so every hour is counted once per bucket, not once per fetch.

SCORES, per model and lead bucket:
  rho    Spearman correlation of cloud cover with transparency, sign-flipped so
         higher is better. Threshold-free, so it cannot be tuned into looking
         good, and it is the headline number.
  POD    of the hours the camera called clear, the fraction the model called
         clear (cover < --clear-cc). Missed clear hours are lost imaging.
  FAR    of the hours the model called clear, the fraction the camera called
         cloudy. False clears open the roof into cloud.
  acc    fraction of hours classified right.
  HSS    Heidke skill score: accuracy above chance; 0 = no better than guessing
         with the model's own clear/cloudy rates, 1 = perfect.
The camera calls an hour clear when relative transparency >= --clear-rel.

This scores; it does not decide. Switching the model the scheduler reads is a
separate call to make once the numbers have held up over enough nights.

Usage:
  python scripts/forecast_score.py                 # all nights, all leads
  python scripts/forecast_score.py --since 2026-09-01
  python scripts/forecast_score.py --hours-csv local/forecast_score_hours.csv
  python scripts/forecast_score.py --with-moon     # include moonlit hours
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
from astropy.utils import iers

# Same reason as sky_transparency_report.py: the cache is not writable here and
# milliarcsecond Earth-orientation drift cannot move a star out of the field.
iers.conf.auto_download = False
iers.conf.auto_max_age = None

from scripts.sky_transparency_report import in_field

SKY_LOG = "local/sky_log.jsonl"
FORECAST_LOG = "local/forecast_log.jsonl"
SOLUTION = "local/sky_solution.json"
SOLUTION_ARCHIVE = "local/sky_solution_*.json"

SUN_MAX_ALT = -12.0
MOON_MAX_ILLUM = 0.30
MIN_FRAMES_PER_HOUR = 6
CEILING_PCTL = 90
LEAD_BUCKETS = [(0, 3), (3, 9), (9, 18), (18, 25)]
BOOTSTRAP = 400


def _utc(s):
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def _floor_hour(t):
    return t.replace(minute=0, second=0, microsecond=0)


def night_of(hour_utc):
    """The evening date a UTC hour's night began on (Eastern dusk is ~22-00 UTC)."""
    return (hour_utc - timedelta(hours=16)).date()


def load_solutions():
    """[(valid_until_utc or None, solution)], oldest first.

    An archive named sky_solution_<T>.json is the solution that was REPLACED at
    T (its own solved_at is earlier), so it covers frames up to T. The live file
    covers everything after the newest archive.
    """
    epochs = []
    for p in glob.glob(SOLUTION_ARCHIVE):
        m = re.search(r"sky_solution_(\d{8}T\d{6})Z\.json$", p)
        if not m:
            continue
        until = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        epochs.append((until, json.load(open(p))))
    epochs.sort(key=lambda e: e[0])
    epochs.append((None, json.load(open(SOLUTION))))
    return epochs


def epoch_index(epochs, t):
    untils = [u for u, _ in epochs[:-1]]
    return bisect_right(untils, t)


def truth_hours(since, until):
    """{utc_hour: {"frames": [...], ...}} from night frames with a solve."""
    hours = {}
    for line in open(SKY_LOG):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("night") or r.get("solve_matches") is None:
            continue
        sun = r.get("sun_alt_deg")
        if sun is None or sun > SUN_MAX_ALT:
            continue
        t = _utc(r.get("captured") or r["generated"])
        if (since and t < since) or (until and t >= until):
            continue
        h = hours.setdefault(_floor_hour(t), {"matches": [], "rain": False})
        h["matches"].append(r["solve_matches"])
        if r.get("rain_alert"):
            h["rain"] = True
    return {k: v for k, v in hours.items() if len(v["matches"]) >= MIN_FRAMES_PER_HOUR}


def moon_state(times):
    """(alt_deg, illuminated_fraction) arrays for UTC datetimes."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time
    from configs import config

    loc = config.data()["location"]
    site = EarthLocation.from_geodetic(float(loc["longitude"]) * u.deg,
                                       float(loc["latitude"]) * u.deg,
                                       float(loc.get("elevation", 0)) * u.m)
    t = Time(times)
    moon = get_body("moon", t, site)
    alt = moon.transform_to(AltAz(obstime=t, location=site)).alt.deg
    elong = moon.separation(get_body("sun", t, site)).rad
    return alt, 0.5 * (1 - np.cos(elong))


def build_truth(since, until, with_moon):
    epochs = load_solutions()
    hours = truth_hours(since, until)
    keys = sorted(hours)
    if not keys:
        return []
    mids = [k + timedelta(minutes=30) for k in keys]
    malt, millum = moon_state(mids)

    rows = []
    for k, mid, a, f in zip(keys, mids, malt, millum):
        ei = epoch_index(epochs, mid)
        # One denominator per hour: the sky turns 15 deg in an hour, which
        # changes the in-field count by a few percent -- small beside the
        # spread of the hour's own frames, and 12x cheaper than per frame.
        n = in_field(epochs[ei][1], mid)
        if n <= 0:
            continue
        moonlit = bool(a > 0 and f >= MOON_MAX_ILLUM)
        rows.append({
            "hour": k, "epoch": ei, "in_field": n,
            "frames": len(hours[k]["matches"]),
            "transp": float(np.median(hours[k]["matches"])) / n,
            "moon_alt": float(a), "moon_illum": float(f), "moonlit": moonlit,
            "rain": hours[k]["rain"],
        })

    # Ceiling per epoch from moon-free hours ONLY, even under --with-moon. The
    # ceiling is "what a clear sky reads with this solution"; taken from moonlit
    # hours it drops to the moon's level and every moonlit hour reads clear
    # (the 2026-09-25 epoch, entirely under a gibbous moon, came out at 10%).
    # An epoch without enough moon-free hours cannot be normalised and is
    # left out rather than guessed.
    scored = [r for r in rows if with_moon or not r["moonlit"]]
    for ei in {r["epoch"] for r in rows}:
        vals = [r["transp"] for r in rows if r["epoch"] == ei and not r["moonlit"]]
        ceil = float(np.percentile(vals, CEILING_PCTL)) if len(vals) >= 10 else None
        for r in rows:
            if r["epoch"] == ei:
                r["ceiling"] = ceil
                r["rel"] = r["transp"] / ceil if ceil else None
    return [r for r in scored if r.get("rel") is not None]


def forecasts():
    """{(source, utc_hour): [(lead_h, cc, ccl)]} over every logged fetch."""
    out = {}
    for line in open(FORECAST_LOG):
        try:
            f = json.loads(line)
        except ValueError:
            continue
        v0 = f.get("valid_from_utc")
        if not isinstance(v0, str) or not f.get("fetched"):
            continue
        fetched, v0 = _utc(f["fetched"]), _utc(v0)
        series = {m: (d.get("cc"), d.get("ccl"))
                  for m, d in (f.get("open_meteo") or {}).items()}
        nws = (f.get("nws") or {}).get("sky")
        if nws:
            series["nws"] = (nws, None)
        for src, (cc, ccl) in series.items():
            if not cc:
                continue
            for i, c in enumerate(cc):
                if c is None:
                    continue
                h = v0 + timedelta(hours=i)
                lead = (h + timedelta(minutes=30) - fetched).total_seconds() / 3600
                low = ccl[i] if ccl and i < len(ccl) else None
                out.setdefault((src, h), []).append((lead, c, low))
    return out


def spearman(x, y):
    if len(x) < 5:
        return None
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    # Average ranks for ties, which cloud cover is full of (0 and 100).
    for arr, r in ((np.asarray(x), rx), (np.asarray(y), ry)):
        for v in np.unique(arr):
            idx = arr == v
            if idx.sum() > 1:
                r[idx] = r[idx].mean()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def night_bootstrap(c, r, night, n_boot=BOOTSTRAP, seed=0):
    """5-95% interval of -rho, resampling whole NIGHTS.

    Hours within a night are not independent -- one cloud deck covers eight of
    them -- so resampling hours would claim ~8x the confidence the data has.
    The seed is fixed so every model sees the same resampled nights and the
    intervals can be compared row against row.
    """
    keys = np.unique(night)
    groups = [np.flatnonzero(night == k) for k in keys]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        s = spearman(c[idx], r[idx])
        if s is not None:
            out.append(-s)
    if len(out) < n_boot // 2:
        return None, None
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def score(pairs, clear_cc, clear_rel, field):
    """pairs: [(cc, ccl, rel, night)] -> dict of scores."""
    vals = [(p[0] if field == "cc" else p[1], p[2], p[3]) for p in pairs]
    vals = [v for v in vals if v[0] is not None]
    if not vals:
        return None
    c = np.array([v[0] for v in vals], float)
    r = np.array([v[1] for v in vals], float)
    night = np.array([v[2] for v in vals])
    rho = spearman(c, r)
    lo, hi = night_bootstrap(c, r, night)
    fc, oc = c < clear_cc, r >= clear_rel
    a, b = int((fc & oc).sum()), int((fc & ~oc).sum())      # hit, false clear
    m, d = int((~fc & oc).sum()), int((~fc & ~oc).sum())    # miss, correct cloudy
    n = a + b + m + d
    expected = ((a + b) * (a + m) + (m + d) * (b + d)) / n
    return {
        "n": n, "nights": len(np.unique(night)),
        "rho": None if rho is None else -rho, "rho_lo": lo, "rho_hi": hi,
        "pod": a / (a + m) if a + m else None,
        "far": b / (a + b) if a + b else None,
        "acc": (a + d) / n,
        "hss": (a + d - expected) / (n - expected) if n != expected else None,
    }


def run(args):
    since = _utc(args.since) if args.since else None
    until = _utc(args.until) if args.until else None
    truth = build_truth(since, until, args.with_moon)
    if not truth:
        print("no scorable sky-camera hours")
        return 1
    fc = forecasts()
    sources = sorted({s for s, _ in fc}, key=lambda s: (s == "nws", s != "best_match", s))

    by_hour = {t["hour"]: t for t in truth}
    n_clear = sum(1 for t in truth if t["rel"] >= args.clear_rel)
    nights = {night_of(t["hour"]) for t in truth}
    print("Forecast scoring against the sky camera")
    print("  %d scored hours over %d nights (%s .. %s UTC), %s"
          % (len(truth), len(nights), truth[0]["hour"].strftime("%Y-%m-%d"),
             truth[-1]["hour"].strftime("%Y-%m-%d"),
             "moonlit hours INCLUDED" if args.with_moon else
             "moon down or <%d%% lit only" % (MOON_MAX_ILLUM * 100)))
    print("  camera called %d clear (rel transparency >= %.2f), %d cloudy; "
          "model clear = cover < %d%%"
          % (n_clear, args.clear_rel, len(truth) - n_clear, args.clear_cc))
    ceilings = sorted({(t["epoch"], t["ceiling"]) for t in truth})
    print("  solution-epoch ceilings (p%d transparency): %s"
          % (CEILING_PCTL, ", ".join("%d:%.0f%%" % (e, c * 100) for e, c in ceilings)))

    table = []
    for lo, hi in LEAD_BUCKETS:
        print("\nLead %d-%d h   (%s cover)" % (lo, hi, "LOW" if args.low else "total"))
        print("  %-22s %5s %6s %-11s %5s %5s %5s %6s"
              % ("model", "hours", "rho", " (5-95%)", "POD", "FAR", "acc", "HSS"))
        rows = []
        for src in sources:
            pairs = []
            for hour, t in by_hour.items():
                cands = [x for x in fc.get((src, hour), []) if lo <= x[0] < hi]
                if not cands:
                    continue
                lead, cc, ccl = min(cands, key=lambda x: x[0])   # latest issue
                pairs.append((cc, ccl, t["rel"], night_of(hour)))
            s = score(pairs, args.clear_cc, args.clear_rel, "ccl" if args.low else "cc")
            if s is None:
                continue
            s.update(model=src, lead=f"{lo}-{hi}")
            rows.append(s)
            table.append(s)
        rows.sort(key=lambda s: -(s["rho"] if s["rho"] is not None else -9))
        f = lambda v, w=5, p=2: ("%*.*f" % (w, p, v)) if v is not None else " " * (w - 1) + "-"
        for s in rows:
            ci = ("(%.2f-%.2f)" % (s["rho_lo"], s["rho_hi"])) if s["rho_lo"] is not None else ""
            print("  %-22s %5d %s %-11s %s %s %s %s" % (
                s["model"], s["n"], f(s["rho"], 6), ci, f(s["pod"]), f(s["far"]),
                f(s["acc"]), f(s["hss"], 6)))

    # Is the blend just one model here? If so the scheduler is reading that
    # model under another name, which matters when choosing what to switch to.
    same = {}
    for (src, hour), lst in fc.items():
        if src in ("best_match", "nws"):
            continue
        bm = {round(l): c for l, c, _ in fc.get(("best_match", hour), [])}
        for l, c, _ in lst:
            if round(l) in bm:
                d = same.setdefault(src, [0, 0])
                d[0] += bm[round(l)] == c
                d[1] += 1
    if same:
        print("\nbest_match identical to: " + ", ".join(
            "%s %.0f%%" % (k, 100 * a / n) for k, (a, n) in
            sorted(same.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:4]))

    if args.hours_csv:
        with open(args.hours_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["hour_utc", "epoch", "frames", "in_field", "transp", "rel",
                        "moon_alt", "moon_illum", "rain"])
            for t in truth:
                w.writerow([t["hour"].isoformat(), t["epoch"], t["frames"], t["in_field"],
                            "%.4f" % t["transp"], "%.3f" % t["rel"],
                            "%.1f" % t["moon_alt"], "%.2f" % t["moon_illum"], int(t["rain"])])
        print("\nwrote %s" % args.hours_csv)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(table, fh, indent=1)
        print("wrote %s" % args.json)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", help="UTC date/time, e.g. 2026-09-01")
    ap.add_argument("--until", help="UTC date/time, exclusive")
    ap.add_argument("--with-moon", action="store_true", help="also score moonlit hours")
    ap.add_argument("--low", action="store_true", help="score low cloud instead of total")
    ap.add_argument("--clear-cc", type=float, default=30.0,
                    help="model calls an hour clear below this cover %% (default 30)")
    ap.add_argument("--clear-rel", type=float, default=0.5,
                    help="camera calls an hour clear at this fraction of its "
                         "epoch's ceiling transparency (default 0.5)")
    ap.add_argument("--hours-csv", help="write the per-hour ground truth here")
    ap.add_argument("--json", help="write the score table here")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
