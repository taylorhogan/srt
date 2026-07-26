"""Step 2: is the blue-registration failure an exposure-mismatch artifact?

Reads EXPTIME from each blue sub, then tests astroalign alignment:
  - 30s  -> 30s  (shallow to shallow)
  - 30s  -> 300s (shallow to deep, what the pipeline currently does)
  - 300s -> 300s (deep to deep)

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
image_dir = Path(cfg["nina"]["image_dir"])
dso_dir = image_dir / "ngc5907"


def is_light(f): return f.parent.name.upper() == "LIGHT"


def exptime(p):
    with fits.open(p) as h:
        return float(h[0].header.get("EXPTIME", h[0].header.get("EXPOSURE", 0)))


def align(frame, ref, sigma=5):
    import astroalign as aa
    try:
        t, (src, dst) = aa.find_transform(frame, ref, detection_sigma=sigma)
        return f"OK ({len(src)})"
    except Exception as exc:
        return f"FAIL ({str(exc)[:40]})"


lights = [f for f in dso_dir.rglob("*.fits") if is_light(f)]
blue = sorted([f for f in lights if stacker._read_filter(f).upper() == "B"]
              if hasattr(stacker, "_read_filter") else
              stacker.group_by_filter(lights).get("B", []),
              key=lambda f: f.stat().st_mtime)

print(f"{len(blue)} blue subs")
expmap = {}
for p in blue:
    e = exptime(p)
    expmap.setdefault(round(e), []).append(p)
print("EXPTIME split:", {k: len(v) for k, v in expmap.items()})

# Pick reference frames per exposure class
classes = sorted(expmap.keys())
short_e = classes[0]
long_e = classes[-1]
shorts = expmap[short_e]
longs = expmap[long_e]

short_ref = stacker._load_fits_2d(shorts[0])
long_ref = stacker._load_fits_2d(longs[0])

print(f"\n--- {short_e}s -> {short_e}s (shallow->shallow, ref={shorts[0].name[:24]}) ---")
for p in shorts[1:6]:
    print(f"  {p.name[:28]:<28} {align(stacker._load_fits_2d(p), short_ref)}")

print(f"\n--- {short_e}s -> {long_e}s (shallow->deep = current pipeline) ---")
for p in shorts[:5]:
    print(f"  {p.name[:28]:<28} {align(stacker._load_fits_2d(p), long_ref)}")

print(f"\n--- {long_e}s -> {long_e}s (deep->deep, ref={longs[0].name[:24]}) ---")
for p in longs[1:]:
    print(f"  {p.name[:28]:<28} {align(stacker._load_fits_2d(p), long_ref)}")
