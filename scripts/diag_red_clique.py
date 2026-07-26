"""Characterise the two R cliques: within vs across alignment, sep source
counts, background level/gradient, and whether more control points help.

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


def is_light(f): return f.parent.name.upper() == "LIGHT"


paths = sorted([f for f in dso_dir.rglob("*.fits") if is_light(f)
                and stacker.group_by_filter([f]).get("R")],
               key=lambda f: f.stat().st_mtime)

# Representative indices: early (06-02/03), mid (06-04), late (06-05/09)
idxs = [0, 1, 4, 8, 12, 17, 21, 25, 29, 33]
frames = {i: stacker._load_fits_2d(paths[i]) for i in idxs}

import sep
print(f"{'idx':>3} {'name':<22} {'bkg':>8} {'rms':>7} {'src@5s':>8} {'src@10s':>8} {'med_flux':>9} {'med_a_px':>8}")
for i in idxs:
    d = frames[i].astype(float)
    bkg = sep.Background(d)
    sub = d - bkg.back()
    src = sep.extract(sub, thresh=10.0, err=bkg.rms())
    src5 = sep.extract(sub, thresh=5.0, err=bkg.rms())
    flux = np.median(src["flux"]) if len(src) else 0
    a = np.median(src["a"]) if len(src) else 0
    print(f"{i:>3} {paths[i].name[:22]:<22} {np.median(bkg.back()):>8.1f} "
          f"{np.median(bkg.rms()):>7.1f} {len(src5):>8} {len(src):>8} {flux:>9.0f} {a:>8.2f}")

import astroalign as aa


def trial(i, j, **kw):
    try:
        t, (s, d) = aa.find_transform(frames[i], frames[j], **kw)
        return f"OK({len(s)})"
    except Exception as e:
        return f"FAIL"


print("\nPairwise alignment (default params):")
pairs = [(0, 1), (0, 4), (0, 8), (4, 8), (8, 17), (0, 17), (17, 21), (17, 25), (25, 29)]
for i, j in pairs:
    print(f"  {i:>2} -> {j:>2}: {trial(i, j)}")

print("\n0 -> 17 with more/looser control points:")
for kw in (dict(max_control_points=100), dict(max_control_points=200),
           dict(detection_sigma=3), dict(detection_sigma=10),
           dict(max_control_points=300, detection_sigma=3)):
    print(f"  {kw}: {trial(0, 17, **kw)}")
