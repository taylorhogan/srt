"""Validate the fix: guarded FWHM reference + higher max_control_points.

For the R group, pick the guarded reference, then count how many frames align
at max_control_points = 50 (current default), 100, 200, 300 — with timing.

Read-only.
"""
import os, sys, time
from pathlib import Path

if __package__ in (None, ""):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

import numpy as np
from configs import config
from stacking import stacker

cfg = config.data()
dso_dir = Path(cfg["nina"]["image_dir"]) / "ngc5907"
FILT = (sys.argv[1] if len(sys.argv) > 1 else "R").upper()


def is_light(f): return f.parent.name.upper() == "LIGHT"


paths = sorted([f for f in dso_dir.rglob("*.fits") if is_light(f)
                and stacker.group_by_filter([f]).get(FILT)],
               key=lambda f: f.stat().st_mtime)
frames = [stacker._load_fits_2d(p) for p in paths]
fwhm = [stacker._measure_fwhm(p) for p in paths]
ref_idx = stacker._reference_index_by_fwhm(fwhm)
print(f"{FILT}: {len(frames)} frames, guarded reference = frame {ref_idx} (FWHM {fwhm[ref_idx]:.2f})")

import astroalign as aa
ref = frames[ref_idx]
for mcp in (50, 100, 200, 300):
    ok = 0
    t0 = time.time()
    for j in range(len(frames)):
        if j == ref_idx:
            continue
        try:
            aa.find_transform(frames[j], ref, max_control_points=mcp)
            ok += 1
        except Exception:
            pass
    dt = time.time() - t0
    print(f"  max_control_points={mcp:>4}: {ok:>2}/{len(frames)-1}  "
          f"({dt:.0f}s total, {dt/(len(frames)-1):.1f}s/frame)")
