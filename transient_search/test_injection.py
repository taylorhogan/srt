"""Synthetic-injection acceptance test for the transient / image-differencing
pipeline.

A real supernova can't be summoned on demand, so this validates the pipeline by
injecting fake point sources of known position and brightness into the science
(latest-night) frames, running the full search, and checking (a) how faint a
source is still recovered — a completeness curve — and (b) how many false
positives appear on the un-injected difference.

Run against a real multi-night galaxy field:

    .venv/Scripts/python transient_search/test_injection.py m101 r

Requires ≥2 nights of the target on the given filter (same as the command).
"""

import math
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from astropy.io import fits

from configs import config
from transient_search import difference as tr
from transit_search.transit import _find_dso_dir
from utils.injection_checks import check_monotonic_recovery, report

# Flux levels (total ADU) injected, from faint to bright.
FLUX_LEVELS = [100, 200, 400, 800, 1600, 3200, 6400]
PER_LEVEL = 3          # injections per flux level (recovery fraction per level)
MATCH_PX = 3.0         # a returned candidate within this radius counts as recovered


def _frame_size(image_dir: Path, dso: str, filt: str) -> tuple:
    """(width, height) from the first LIGHT frame of the target/filter."""
    dso_dir = _find_dso_dir(image_dir, dso)
    if dso_dir is None:
        raise SystemExit(f"No image directory found for '{dso}'")
    for f in sorted(dso_dir.rglob("*.fits")):
        if f.parent.name.upper() == "LIGHT":
            hdr = fits.getheader(f)
            return int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    raise SystemExit(f"No LIGHT frames under {dso_dir}")


def _injection_grid(w: int, h: int, n: int) -> list:
    """`n` well-separated positions in the central half of the frame."""
    cols = max(1, int(math.ceil(math.sqrt(n * w / h))))
    rows = int(math.ceil(n / cols))
    xs = [w * (0.3 + 0.4 * (c + 0.5) / cols) for c in range(cols)]
    ys = [h * (0.3 + 0.4 * (r + 0.5) / rows) for r in range(rows)]
    pts = [(x, y) for y in ys for x in xs]
    return pts[:n]


def main() -> None:
    dso = sys.argv[1] if len(sys.argv) > 1 else "m101"
    filt = sys.argv[2] if len(sys.argv) > 2 else "r"
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    scratch = Path(os.path.join(
        os.path.dirname(__file__), "..", cfg["scratch"]["directory"])).resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str) -> None:
        print(f"  [{dso}/{filt}] {msg}")

    w, h = _frame_size(image_dir, dso, filt)
    positions = _injection_grid(w, h, len(FLUX_LEVELS) * PER_LEVEL)
    inject = []
    truth = []                    # (x, y, flux) we expect to recover
    for i, (x, y) in enumerate(positions):
        flux = FLUX_LEVELS[i // PER_LEVEL]
        inject.append({"x": x, "y": y, "flux_adu": flux})
        truth.append((x, y, flux))

    # 1) Baseline: no injection → any survivors are false positives.
    print("== baseline run (no injection) ==")
    base = tr.run_transient_search(
        dso, filt, image_dir, scratch / f"inj_baseline_{dso}_{filt}.jpg",
        progress_cb=_progress)
    false_positives = [c for c in base.get("candidates", []) if not c.get("gaia_rejected")]
    print(f"  false positives (survivors on un-injected diff): {len(false_positives)}")

    # 2) Injected run → measure recovery per flux level.
    print(f"== injected run ({len(inject)} sources) ==")
    out = tr.run_transient_search(
        dso, filt, image_dir, scratch / f"inj_test_{dso}_{filt}.jpg",
        progress_cb=_progress, inject=inject, top_n=len(inject) + 10)
    cands = out.get("candidates", [])

    def _recovered(tx: float, ty: float) -> bool:
        return any(math.hypot(c["x"] - tx, c["y"] - ty) <= MATCH_PX for c in cands)

    print("\n  flux(ADU)  recovered")
    by_level = {}
    for (tx, ty, flux) in truth:
        by_level.setdefault(flux, []).append(_recovered(tx, ty))
    faintest = None
    counts: list[int] = []
    for flux in FLUX_LEVELS:
        hits = by_level.get(flux, [])
        frac = sum(hits) / len(hits) if hits else 0.0
        counts.append(sum(hits))
        print(f"  {flux:>8}   {sum(hits)}/{len(hits)}  ({frac*100:.0f}%)")
        if frac >= 0.5 and faintest is None:
            faintest = flux

    print("\n== summary ==")
    print(f"  false positives : {len(false_positives)}")
    print(f"  faintest 50%-recovered flux : "
          f"{faintest if faintest is not None else 'none recovered'} ADU")

    # A brighter source must not be harder to find. Asserted rather than
    # eyeballed, because this is the check that exposed both the shape window
    # and the self-masking bright-source mask.
    ok, problems = check_monotonic_recovery(
        FLUX_LEVELS, counts, PER_LEVEL, label="flux", fmt="{:.0f} ADU")
    sys.exit(report(ok, problems))


if __name__ == "__main__":
    main()
