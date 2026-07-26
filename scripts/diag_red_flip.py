"""Test the meridian-flip hypothesis for the two R-group cliques.

Takes an early frame (clique A) and a late frame (clique B) and tries
astroalign alignment: as-is, and with the early frame flipped L-R, U-D, and
rotated 180. If a flip/rotation makes it align, the cliques are a
mirror/flip artifact (meridian flip) the pipeline must normalise before
registration. Also prints image shape + a few orientation headers.

Read-only.
"""
import os, sys
from pathlib import Path

if __package__ in (None, ""):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

import numpy as np
from astropy.io import fits
from configs import config
from stacking import stacker

cfg = config.data()
dso_dir = Path(cfg["nina"]["image_dir"]) / "ngc5907"


def is_light(f): return f.parent.name.upper() == "LIGHT"


paths = sorted([f for f in dso_dir.rglob("*.fits") if is_light(f)
                and stacker.group_by_filter([f]).get("R")],
               key=lambda f: f.stat().st_mtime)

early = paths[0]    # 06-02 clique A
late = paths[17]    # 06-05 clique B
print(f"early (clique A): {early.name}")
print(f"late  (clique B): {late.name}")

for label, p in (("early", early), ("late", late)):
    with fits.open(p) as h:
        hdr = h[0].header
        keys = {k: hdr.get(k) for k in
                ("NAXIS1", "NAXIS2", "XBINNING", "YBINNING", "PIERSIDE",
                 "FLIPPED", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2")}
    print(f"  {label}: {keys}")

a = stacker._load_fits_2d(early)
b = stacker._load_fits_2d(late)

import astroalign as aa


def trial(name, arr):
    try:
        aa.find_transform(arr, b)
        print(f"  {name:<18} OK")
    except Exception as e:
        print(f"  {name:<18} FAIL ({str(e)[:45]})")


print("\nAligning early -> late under various orientations:")
trial("as-is", a)
trial("fliplr", np.fliplr(a))
trial("flipud", np.flipud(a))
trial("rot180", np.rot90(a, 2))
trial("transpose", a.T)
