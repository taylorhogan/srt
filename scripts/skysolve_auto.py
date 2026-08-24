"""
skysolve_auto.py
Re-solve the sky camera automatically, but only when the stored plate solution
has demonstrably stopped fitting -- and only from the best frame of the night.

WHY THIS EXISTS, and why it is not simply "re-solve every clear night":

A blind solve is ABSOLUTE -- it matches the catalogue from scratch -- so there
is no accumulating drift that solving more often would head off. Verification,
by contrast, is already free: allsky_monitor/sky_monitor compute solve_matches
on every 5-minute sample. So the right shape is check constantly, fix rarely.
Every install is another chance to replace a good solution with a worse one,
and a nightly solve would spend minutes of CPU on a fault that appears roughly
every two weeks (measured: 0.44 deg of camera drift over the 13 days from
2026-08-11 to 2026-08-24, with nothing touching the camera).

WHY IT RUNS IN THE MORNING, off the archive. A solve needs a DARK FRAME, not
darkness. Running at ~10:00 from the night's best archived frame means it never
races dawn, never competes with an imaging run for CPU, and gets to pick the
cleanest frame of the whole night instead of whatever happens to be current.
The 2026-08-24 repair was done exactly this way: purity 1.000, 0 false
positives, sun -26.8 deg, and it produced 122 matches at 3.53 px.

THE FAULT THIS CATCHES is a grade collapse, not a detection failure. Star
detection stays perfectly healthy while the camera drifts -- 157-189 stars per
frame at purity ~1.0 -- and what breaks is the match of detections to the
catalogue. Its signature is a FLAT completeness curve (a real detection limit
falls off steeply with brightness) and limiting_mag going None.

RISK. The solution is a measurement calibration, not a hardware interlock. The
one gate that reads it (allsky_monitor._add_roof) treats "many catalogue stars
match" as proof the roof is OPEN, so a stale or wrong solution drives matches
DOWN and the gate falls back to vision. That direction is safe, and a closed
roof cannot produce catalogue matches at any orientation. Nothing here moves
hardware.

Usage:
    python scripts/skysolve_auto.py              # decide and act
    python scripts/skysolve_auto.py --dry-run    # decide, report, change nothing
    python scripts/skysolve_auto.py --night 2026-08-23
    python scripts/skysolve_auto.py --force      # skip the persistence gate
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from pathlib import Path

LOG_PATH = "local/sky_log.jsonl"
FRAME_DIR = "local/sky_frames"
STATE_PATH = "local/skysolve_auto_state.json"

# ---------------------------------------------------------------- thresholds

# How many trustworthy samples must independently report the solution unfit
# before it is believed. A single frame is not evidence: haze drops the match
# count too, which is why the monitor already refuses to judge the solution on
# an untrustworthy frame. Real drift fails EVERY frame of the night -- measured
# 2026-08-23, all 47 trustworthy frames between 3% and 9% -- so a demanding
# gate costs a genuine fault nothing and rejects a bad patch of sky.
MIN_FAILING_SAMPLES = 10

# ...and they must be the clear majority of the night's verdicts, so a night
# that mostly fitted cannot be re-solved on the strength of its worst hour.
MIN_FAILING_FRACTION = 0.80

# What counts as "failing", as a share of detections that matched.
#
# NOT the monitor's own note_solve flag, which fires at share < 0.08 -- that
# threshold sits INSIDE the broken population (measured 2026-08-23: broken
# frames ran 3-9% and straddled it), so counting notes made a fully broken
# night read 71% failing and the persistence gate rejected it as weather. The
# two populations are far apart when measured directly -- working nights sit at
# 44-70% share, broken ones at 3-10% -- so the discriminator belongs in the gap
# where nothing lives, not at the edge of one of them.
FAILING_SHARE = 0.25

# The frame the solve is attempted on must be genuinely dark and genuinely
# clean. -18 deg is astronomical twilight; purity is the negative-image control
# from star_count, where ~1.0 means essentially every detection is real.
SOLVE_MIN_SUN_DEG = -18.0
SOLVE_MIN_PURITY = 0.95
SOLVE_MIN_STARS = 60

# ------- guards on the RESULT. All must pass or nothing is written.
# Beating the old solution on the same frame is necessary but not sufficient:
# when the old one is broken, "better than broken" is a low bar. So the new
# solve must also stand on its own.
GUARD_MIN_MATCHES = 25          # absolute floor, not just "beats the old"
GUARD_MAX_RESIDUAL_PX = 5.0     # good solves measure 3.5; broken ones 5.5-6.8
GUARD_MIN_NAMED = 3             # named catalogue stars -- the anti-coincidence
                                # check. A wrong solve lands on no real names.


def _night_of(ts):
    """The observing night a UTC timestamp belongs to (local-ish, -12h rule)."""
    t = datetime.fromisoformat(ts).replace(tzinfo=None)
    return (t - timedelta(hours=12)).date()


def _frame_for(row):
    """Archived frame path for a log row, or None if it is not on disk."""
    cap = row.get("captured") or row.get("generated")
    if not cap:
        return None
    stamp = cap.replace("+00:00", "").replace("-", "").replace(":", "")
    p = Path(FRAME_DIR) / ("sky_%sZ.jpg" % stamp)
    return p if p.exists() else None


def load_rows():
    if not os.path.exists(LOG_PATH):
        return []
    return [json.loads(l) for l in open(LOG_PATH) if l.strip()]


def assess(rows, night=None):
    """Should we re-solve? Returns (decision: bool, detail: dict)."""
    nights = {}
    for r in rows:
        cap = r.get("captured") or r.get("generated")
        if not cap or not r.get("night"):
            continue
        nights.setdefault(str(_night_of(cap)), []).append(r)
    if not nights:
        return False, {"why": "no night rows in %s" % LOG_PATH}
    key = night or max(nights)
    got = nights.get(key, [])
    # Only frames the detector itself certifies. The monitor already declines to
    # judge the solution on a poor frame, so these are the only rows carrying a
    # meaningful verdict.
    judged = [r for r in got if r.get("trustworthy")
              and r.get("solve_matches") is not None
              and not (r.get("note_solve") or "").startswith("sky too poor")]
    def _share(r):
        return r["solve_matches"] / max(r.get("stars") or 1, 1)

    failing = [r for r in judged if _share(r) < FAILING_SHARE]
    detail = {"night": key, "samples": len(got), "judged": len(judged),
              "failing": len(failing),
              "fraction": round(len(failing) / len(judged), 3) if judged else None}
    if not judged:
        detail["why"] = "no trustworthy frames carried a solve verdict"
        return False, detail
    if len(failing) < MIN_FAILING_SAMPLES:
        detail["why"] = ("only %d failing samples, need %d"
                         % (len(failing), MIN_FAILING_SAMPLES))
        return False, detail
    if len(failing) / len(judged) < MIN_FAILING_FRACTION:
        detail["why"] = ("failing fraction %.0f%% below %.0f%% -- looks like "
                         "weather, not drift"
                         % (100 * len(failing) / len(judged),
                            100 * MIN_FAILING_FRACTION))
        return False, detail
    detail["why"] = ("%d of %d trustworthy frames report the solution unfit"
                     % (len(failing), len(judged)))
    return True, detail


def pick_frame(rows, night):
    """The night's cleanest deep-dark frame, as (path, row). Purity first.

    Purity before star count deliberately: the highest-count frames of a hazy
    night carry the most false positives (measured 2026-08-24: 336 stars at
    purity 0.857 against 189 stars at purity 1.000), and a solve wants real
    point sources, not a long list.
    """
    cands = []
    for r in rows:
        cap = r.get("captured") or r.get("generated")
        if not cap or not r.get("night") or str(_night_of(cap)) != night:
            continue
        if (r.get("sun_alt_deg") is None
                or r["sun_alt_deg"] > SOLVE_MIN_SUN_DEG
                or (r.get("purity") or 0) < SOLVE_MIN_PURITY
                or (r.get("stars") or 0) < SOLVE_MIN_STARS):
            continue
        f = _frame_for(r)
        if f is not None:
            cands.append((round(r["purity"], 3), r["stars"], str(f), r))
    if not cands:
        return None, None
    cands.sort(key=lambda c: (-c[0], -c[1]))
    return cands[0][2], cands[0][3]


def _state():
    try:
        return json.load(open(STATE_PATH))
    except Exception:
        return {}


def _write_state(d):
    try:
        Path(STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        json.dump(d, open(STATE_PATH, "w"), indent=2)
    except Exception:
        pass


def notify(msg, push=True):
    print(msg, flush=True)
    try:
        from cmd_processing import social_server
        social_server.post_social_message(msg)
    except Exception:
        pass
    if push:
        try:
            from utils import pushover
            pushover.push_message(msg)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and report, but never write a solution")
    ap.add_argument("--night", default=None, help="YYYY-MM-DD (default: latest)")
    ap.add_argument("--force", action="store_true",
                    help="skip the persistence gate (still applies result guards)")
    ap.add_argument("--quiet", action="store_true",
                    help="no pushover/webchat, print only")
    args = ap.parse_args()

    rows = load_rows()
    go, detail = assess(rows, args.night)
    night = detail.get("night")
    print("night %s: %s" % (night, detail.get("why")))
    if not go and not args.force:
        return 0

    st = _state()
    if st.get("last_night") == night and not args.force:
        print("already re-solved for night %s -- not repeating" % night)
        return 0

    frame, row = pick_frame(rows, night)
    if frame is None:
        print("no frame clean enough to solve from (need sun<=%.0f, purity>=%.2f, "
              "stars>=%d)" % (SOLVE_MIN_SUN_DEG, SOLVE_MIN_PURITY, SOLVE_MIN_STARS))
        return 0
    print("solving from %s (stars %d, purity %.3f, sun %.1f)"
          % (os.path.basename(frame), row["stars"], row["purity"],
             row["sun_alt_deg"]))

    from sentry import plate_solve as ps, star_count
    res = star_count.count_stars(frame)
    when = ps.frame_time(frame)
    old = ps.load()
    oldv = ps.verify(old, frame, when, res["_stars"]) if old else {"matches": 0}
    print("stored solution on this frame: %d matches" % oldv["matches"])

    sol = ps.solve(frame, when, res["_stars"])
    if sol is None:
        print("no solution found -- nothing changed")
        return 0
    named = ps.named_identifications(sol, when, res["_stars"])
    print("new solve: %d matches, residual %s px, %d named stars"
          % (sol["matches"], sol["residual_px"], len(named)))

    # ---- guards. Every one must pass; the first failure writes nothing.
    fails = []
    if sol["matches"] < oldv["matches"]:
        fails.append("matches fewer stars than the stored solution (%d < %d)"
                     % (sol["matches"], oldv["matches"]))
    if sol["matches"] < GUARD_MIN_MATCHES:
        fails.append("only %d matches, floor is %d"
                     % (sol["matches"], GUARD_MIN_MATCHES))
    if sol["residual_px"] is None or sol["residual_px"] > GUARD_MAX_RESIDUAL_PX:
        fails.append("residual %s px above the %.1f px ceiling"
                     % (sol["residual_px"], GUARD_MAX_RESIDUAL_PX))
    if len(named) < GUARD_MIN_NAMED:
        fails.append("named only %d catalogue stars, need %d -- looks like "
                     "coincidence" % (len(named), GUARD_MIN_NAMED))
    if fails:
        notify("Sky camera auto-resolve REJECTED for %s: %s. Stored solution "
               "kept unchanged." % (night, "; ".join(fails)), push=not args.quiet)
        return 0

    if args.dry_run:
        print("DRY RUN -- would install: %d matches, residual %s px, "
              "axis alt %.2f az %.2f" % (sol["matches"], sol["residual_px"],
                                         sol["axis_alt_deg"], sol["axis_az_deg"]))
        return 0

    # True angular motion, not the raw azimuth difference. Azimuth converges
    # near the zenith and this axis sits at ~84 deg, where 3.67 deg of azimuth
    # is 0.44 deg of actual sky -- quoting the raw number overstates a move by
    # an order of magnitude.
    import math
    a0, a1 = math.radians(old["axis_alt_deg"]), math.radians(sol["axis_alt_deg"])
    dz = math.radians((sol["axis_az_deg"] - old["axis_az_deg"] + 180) % 360 - 180)
    sep = math.degrees(math.acos(min(1.0, math.sin(a0) * math.sin(a1)
                                     + math.cos(a0) * math.cos(a1) * math.cos(dz))))

    root = Path(ps._root())
    src = root / ps.solution_file(ps.DEFAULT_PROFILE)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = src.with_name(src.stem + "_" + stamp + ".json")
    try:
        shutil.copy2(src, bak)
    except Exception as exc:
        notify("Sky camera auto-resolve ABORTED: could not back up the old "
               "solution (%s). Nothing changed." % type(exc).__name__,
               push=not args.quiet)
        return 1
    ps.save(sol)
    _write_state({"last_night": night, "when": stamp,
                  "matches": sol["matches"], "residual_px": sol["residual_px"],
                  "moved_deg": round(sep, 3), "backup": bak.name})
    notify("Sky camera re-solved automatically (%s): %d catalogue matches, "
           "residual %s px, camera had drifted %.2f deg. Was %d matches. "
           "Old solution kept as %s."
           % (night, sol["matches"], sol["residual_px"], sep,
              oldv["matches"], bak.name), push=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
