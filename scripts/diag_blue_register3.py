"""Step 3: why is frame 6 a toxic reference, and does FWHM-based or
align-success-based reference selection fix it?

For each blue sub: FWHM (pipeline's own measure) + median star elongation.
Then: for each of the 10 sampled candidate references, how many of the other
17 frames successfully align to it.

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
image_dir = Path(cfg["nina"]["image_dir"])
dso_dir = image_dir / "ngc5907"


def is_light(f): return f.parent.name.upper() == "LIGHT"


def elongation(data):
    """Median a/b axis ratio of detected sources (>1 = trailed)."""
    import sep
    d = data.astype(float)
    bkg = sep.Background(d)
    src = sep.extract(d - bkg.back(), thresh=8.0, err=bkg.rms())
    if len(src) == 0:
        return float("nan")
    a, b = src["a"], src["b"]
    good = b > 0
    return float(np.median(a[good] / b[good])) if good.any() else float("nan")


def align_ok(frame, ref):
    import astroalign as aa
    try:
        aa.find_transform(frame, ref)
        return True
    except Exception:
        return False


lights = [f for f in dso_dir.rglob("*.fits") if is_light(f)]
blue = sorted(stacker.group_by_filter(lights).get("B", []),
              key=lambda f: f.stat().st_mtime)
frames = [stacker._load_fits_2d(p) for p in blue]

print(f"{'idx':>3} {'frame':<28} {'FWHM_px':>8} {'elong':>6} {'src@8s':>7}")
import sep
for i, (p, fr) in enumerate(zip(blue, frames)):
    fw = stacker._measure_fwhm(p)
    el = elongation(fr)
    d = fr.astype(float); bkg = sep.Background(d)
    n = len(sep.extract(d - bkg.back(), thresh=8.0, err=bkg.rms()))
    mark = "  <-- pipeline ref" if i == 6 else ""
    print(f"{i:>3} {p.name[:28]:<28} {fw:>8.2f} {el:>6.2f} {n:>7}{mark}")

# Mutual alignment success per candidate reference (the 10 the pipeline samples)
step = max(1, len(frames) // 10)
cands = list(range(0, len(frames), step))[:10]
print(f"\nAlignment success rate per candidate reference:")
print(f"{'ref idx':>7} {'frame':<28} {'aligns':>10}")
for ci in cands:
    ref = frames[ci]
    ok = sum(align_ok(frames[j], ref) for j in range(len(frames)) if j != ci)
    print(f"{ci:>7} {blue[ci].name[:28]:<28} {ok:>4}/{len(frames)-1}")
