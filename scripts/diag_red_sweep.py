"""Is the R-group failure a reference problem or a field/data problem?

Sweeps several plausible-FWHM candidate references and, for each, counts how
many other frames align. Also groups frames by approximate pointing (WCS CRVAL
or OBJCTRA/DEC header) to see if the field shifted across nights.

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
FILT = (sys.argv[1] if len(sys.argv) > 1 else "R").upper()


def is_light(f): return f.parent.name.upper() == "LIGHT"


def pointing(p):
    with fits.open(p) as h:
        hdr = h[0].header
        for k in ("CRVAL1", "OBJCTRA", "RA"):
            if k in hdr:
                ra = hdr[k]
        dec = None
        for k in ("CRVAL2", "OBJCTDEC", "DEC"):
            if k in hdr:
                dec = hdr[k]
        ra = hdr.get("CRVAL1", hdr.get("RA", hdr.get("OBJCTRA")))
        return ra, dec, hdr.get("CROTA2", hdr.get("ROTATOR", None))


paths = sorted([f for f in dso_dir.rglob("*.fits") if is_light(f)
                and stacker.group_by_filter([f]).get(FILT)],
               key=lambda f: f.stat().st_mtime)
print(f"{FILT}: {len(paths)} frames")

# Pointing spread
print("\nPointing / rotation per frame:")
for i, p in enumerate(paths):
    try:
        ra, dec, rot = pointing(p)
    except Exception as e:
        ra, dec, rot = f"err{e}", None, None
    print(f"  {i:>3} {p.name[:26]:<26} RA={ra} DEC={dec} ROT={rot}")

# Load once
frames = [stacker._load_fits_2d(p) for p in paths]
fwhm = [stacker._measure_fwhm(p) for p in paths]
med = np.median([f for f in fwhm if f > 0])
cand = [i for i, f in enumerate(fwhm) if f >= 0.5 * med]
# sample up to 8 plausible candidates spread across the list
step = max(1, len(cand) // 8)
cand = cand[::step][:8]

import astroalign as aa
print(f"\nAlignment success per candidate reference (median FWHM {med:.2f}):")
for ci in cand:
    ref = frames[ci]
    ok = 0
    for j in range(len(frames)):
        if j == ci:
            continue
        try:
            aa.find_transform(frames[j], ref)
            ok += 1
        except Exception:
            pass
    print(f"  ref frame {ci:>3} (FWHM {fwhm[ci]:5.2f}): {ok}/{len(frames)-1}")
