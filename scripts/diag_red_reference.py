"""Validate the guarded FWHM reference picker on the R group that cratered
(1/34). Prints per-frame FWHM, what the OLD (unguarded min) would pick vs the
NEW guarded picker, then tests astroalign alignment against the NEW reference.

Read-only.
"""
import os, sys
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


lights = [f for f in dso_dir.rglob("*.fits") if is_light(f)]
frames_paths = sorted(stacker.group_by_filter(lights).get(FILT, []),
                      key=lambda f: f.stat().st_mtime)
print(f"{FILT} group: {len(frames_paths)} frames")

fwhm = [stacker._measure_fwhm(p) for p in frames_paths]
for i, (p, f) in enumerate(zip(frames_paths, fwhm)):
    print(f"  {i:>3} {p.name[:30]:<30} FWHM {f:>6.2f}")

positive = [(f, i) for i, f in enumerate(fwhm) if f and f > 0]
old_pick = min(positive)[1] if positive else None
new_pick = stacker._reference_index_by_fwhm(fwhm)
print(f"\nOLD unguarded min-FWHM pick: frame {old_pick} (FWHM {fwhm[old_pick]:.2f})")
print(f"NEW guarded pick:            frame {new_pick} (FWHM {fwhm[new_pick]:.2f})")

# Confirm the new reference actually aligns the others
import astroalign as aa
ref = stacker._load_fits_2d(frames_paths[new_pick])
ok = 0
for j, p in enumerate(frames_paths):
    if j == new_pick:
        continue
    try:
        aa.find_transform(stacker._load_fits_2d(p), ref)
        ok += 1
    except Exception:
        pass
print(f"\nAlignment vs NEW reference: {ok}/{len(frames_paths)-1}")
