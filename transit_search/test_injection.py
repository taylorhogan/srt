#!/usr/bin/env python3
"""Synthetic-injection acceptance test for the transit search.

The companion to ``transient_search/test_injection.py``, and written for the
same reason. A transit search that reports nothing looks identical whether it
is working perfectly or is structurally blind, so cuts tuned until the false
positives went away cannot be distinguished from cuts tuned until the *true*
positives went away. On the transient side that distinction turned out to
matter: the shape window there was rejecting 85% of real point sources and the
bright-source mask was deleting the brightest transients outright, and neither
was visible in any normal run.

This asks the same question of transits. It injects box dips of known depth
into the raw flux of real stars in a real field, runs the whole search, and
measures what fraction come back — a completeness curve in transit depth.

    .venv/Scripts/python transit_search/test_injection.py hatp32 Clear

Two numbers come out:

  * the shallowest depth recovered at least half the time — the sensitivity
    floor, which is the thing the site currently cannot state. HAT-P-32b at
    2.15% is one success with no known floor beneath it;
  * how many candidates the search reports with nothing injected. Unlike the
    transient case these are NOT automatically false positives: real variable
    stars live in every field and the search is supposed to find them. Treat it
    as context, not as an error rate.

Requires enough frames on one night for a transit to be sampled at all.

NOT YET CALIBRATED — do not quote a depth floor from this. Targets are drawn
from the middle of the *brightness* distribution, and brightness is not
photometric quality. Measured on hatp32 Clear, the same 4% injection landed at
SNR 12.4 on one star and SNR 1.8 on another, and a 0.2% injection outranked
every 4% one. At that spread the curve measures the noise distribution of
arbitrarily chosen stars rather than the sensitivity of the search. Fix by
selecting targets on lowest RMS (matching how _differential_normalize picks its
comparison ensemble), or by expressing depth in units of each star's own
scatter, so every injection is a comparable test.

The monotonicity assertion below still applies and is still worth running: it
is what exposed injections landing on unsearched stars, and then the
shared-epoch flaw in this test itself.
"""

import math
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config
from transit_search import transit as tr
from utils.injection_checks import check_monotonic_recovery, report

# Fractional depths, faint to obvious. For scale: HAT-P-32b is 0.0215, a hot
# Jupiter across a solar disc is ~0.01, and a Neptune is ~0.001 — below
# anything a 17" is expected to reach, and included so the floor is bracketed
# from underneath rather than guessed at.
DEPTH_LEVELS = [0.002, 0.005, 0.01, 0.02, 0.04]
PER_DEPTH = 3           # injections per depth (recovery fraction per level)
DURATION_H = 2.0        # transit duration, hours
MATCH_PX = 2.0          # a candidate this close to an injected star counts


def _recovered(truth: dict, candidates: list) -> bool:
    return any(
        math.hypot(c.get("x", 1e9) - truth["x"], c.get("y", 1e9) - truth["y"]) <= MATCH_PX
        for c in candidates)


def main() -> None:
    dso = sys.argv[1] if len(sys.argv) > 1 else "hatp32"
    filt = sys.argv[2] if len(sys.argv) > 2 else "Clear"
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    scratch = Path(os.path.join(
        os.path.dirname(__file__), "..", cfg["scratch"]["directory"])).resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str) -> None:
        print(f"  [{dso}/{filt}] {msg}")

    n_inject = len(DEPTH_LEVELS) * PER_DEPTH
    # Report well beyond the number injected, so a real detection is never
    # counted as a miss just for ranking below the display cutoff.
    top_n = n_inject + 15

    print("== baseline run (no injection) ==")
    base = tr.run_transit_search(
        dso, filt, image_dir, scratch / f"inj_baseline_{dso}_{filt}.jpg",
        progress_cb=_progress, top_n_override=top_n)
    base_cands = base.get("candidates", [])
    print(f"  baseline candidates (context, not an error rate): {len(base_cands)}")
    for c in base_cands[:5]:
        print(f"    score {c.get('score', 0):.1f}  depth "
              f"{c.get('transit_depth', 0)*100:.2f}%  ({c.get('x', 0):.0f},{c.get('y', 0):.0f})")

    print(f"\n== injected run ({n_inject} transits) ==")
    out = tr.run_transit_search(
        dso, filt, image_dir, scratch / f"inj_test_{dso}_{filt}.jpg",
        progress_cb=_progress, top_n_override=top_n,
        inject={"depths": DEPTH_LEVELS, "per_depth": PER_DEPTH,
                "duration_h": DURATION_H, "seed": 0})
    cands = out.get("candidates", [])
    truth = out.get("injected", [])
    if not truth:
        raise SystemExit("no injections were recorded — pipeline did not inject")

    print("\n   depth   recovered")
    by_depth: dict = {}
    for t in truth:
        by_depth.setdefault(t["depth"], []).append(_recovered(t, cands))
    floor = None
    counts: list[int] = []
    for d in DEPTH_LEVELS:
        hits = by_depth.get(d, [])
        frac = sum(hits) / len(hits) if hits else 0.0
        counts.append(sum(hits))
        print(f"  {d*100:5.2f}%   {sum(hits)}/{len(hits)}  ({frac*100:.0f}%)")
        if frac >= 0.5 and floor is None:
            floor = d

    print("\n== summary ==")
    print(f"  frames / baseline      : {out.get('frame_count')} / "
          f"{out.get('baseline_days')} d")
    print(f"  stars searched         : {out.get('n_stars_searched')}")
    print(f"  in-transit samples     : {truth[0]['n_in_transit']} of {out.get('frame_count')}")
    print(f"  baseline candidates    : {len(base_cands)}")
    print(f"  shallowest 50%-recovered depth : "
          f"{f'{floor*100:.2f}%' if floor is not None else 'none recovered'}")

    # A deeper transit must not be harder to find. This is the check that
    # caught injections landing on unsearched stars, and then caught the
    # shared-epoch flaw in this very test.
    ok, problems = check_monotonic_recovery(
        DEPTH_LEVELS, counts, PER_DEPTH, label="depth", fmt="{:.2%}")
    sys.exit(report(ok, problems))


if __name__ == "__main__":
    main()
