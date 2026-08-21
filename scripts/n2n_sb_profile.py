#!/usr/bin/env python3
"""Where does a field sit on the surface-brightness axis?

    python scripts/n2n_sb_profile.py --dir "/media/taylor/cdk17/FromCDK17/NGC 6888" \
        --filters Ha,O2 --limit 22 --label "NGC 6888"

Lab manual step 26 found that what predicts how well a training target
generalises is not how much data it holds, nor how much faint structure it
contains, but how closely its **surface-brightness distribution** matches the
field you intend to denoise. Bubble tracks ic1396 to within 0.2 percentage
points on the two bins holding 97% of the frame, and beats sh2-92 — which has
more faint structure and twice the depth — on every measure of extended
retention.

That makes "where does this field sit on the axis" a question worth answering
before spending nights on a target or picking a training set. This measures it.

The axis is the fraction of frame area in each smoothed surface-brightness bin,
in units of that stack's own sky sigma. It runs from fields largely filled with
emission above the noise (sh2-92: 30% of area below background) to fields that
are mostly blank sky around compact objects (ngc5033: 51%).

**Uncalibrated by default, and deliberately.** This is a shape measurement, not
a photometric one: the profile is taken after `subtract_background`, which
removes the vignetting a flat would have corrected, and the 64 px block mean
washes out the hot pixels a dark would have removed. Forcing calibration off
also avoids a worse failure — `nn.stacks.stack_paths` resolves masters from
config by filter name, so archive lights from another epoch would silently get
the current epoch's flats and darks applied at the wrong temperature.

Do not train on stacks from this script. Use it to decide what to stack properly.

Known caveat: the `>1σ` fraction counts point sources as well as diffuse
emission, so a dense star field (m13, m92) scores high without being "filled" in
the sense that matters. Compare fields of similar type.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Reference points measured 2026-08-20, for placing a new field in context.
REFERENCE = [
    ("sh2-92|Ha", 29.8, 65.4, 4.82),
    ("sh2-92|O-III", 36.6, 58.5, 4.91),
    ("ic1396|Ha  (target)", 41.5, 55.4, 3.04),
    ("bubble|Ha", 41.7, 55.3, 3.05),
    ("bubble|O-III", 45.0, 51.3, 3.70),
    ("ic1396|O-III (target)", 45.5, 51.5, 2.97),
    ("ngc5907|L", 48.2, 48.9, 2.92),
    ("ngc5033|G", 51.5, 47.4, 1.14),
]


def log(m: str = "") -> None:
    print(m, flush=True)


def light_frames(root: Path, filt: str, exptime: int, limit: int) -> list:
    """LIGHT frames matching a literal FITS FILTER string.

    Matched literally rather than through `color_process._ALIASES` because the
    archive uses spellings the alias table does not carry — 2024 data labels
    O-III as `O2`, which every existing path would silently skip.
    """
    from astropy.io import fits

    out = []
    for fp in sorted(root.rglob("*.fits")):
        if "LIGHT" not in [p.upper() for p in fp.parts]:
            continue
        if "RECYCLE" in str(fp):
            continue
        try:
            h = fits.getheader(fp)
            if str(h.get("FILTER", "")).strip() != filt:
                continue
            if exptime and round(float(h.get("EXPTIME", 0))) != exptime:
                continue
        except Exception:
            continue
        out.append(fp)
        if limit and len(out) >= limit:
            break
    return out


def stack_uncalibrated(paths: list, filt: str) -> "np.ndarray":
    from stacking import stacker

    from nn import stacks as nstacks

    ref = nstacks.shared_reference_for(paths, filt, progress_cb=lambda m: log(f"    {m}"))
    img, meta = stacker.stack(
        list(paths),
        method=stacker.StackMethod.SIGMA_CLIP_FWHM,
        shared_reference=ref,
        # Explicitly none: see the module docstring. Config would otherwise
        # supply the current epoch's masters for an archive light frame.
        bias_paths=None, dark_paths=None, flat_paths=None,
        register=True,
        progress_cb=lambda m: None,
    )
    log(f"    kept {meta.get('n_frames')} of {len(paths)}")
    return img


def profile(a: "np.ndarray") -> tuple:
    import importlib.util

    from nn import denoiser

    spec = importlib.util.spec_from_file_location(
        "xc", os.path.join(_root, "scripts", "n2n_extended_check.py"))
    xc = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(xc)
    sys.argv = argv

    a = a.astype(np.float32)
    r, _ = denoiser.subtract_background(a)
    good = ~(a == np.median(a))
    sig = 1.4826 * np.median(np.abs(r[good] - np.median(r[good])))
    sb = xc.smoothed_sb(r, 64) / sig
    tot = good.sum()
    below = 100 * ((sb < 0) & good).sum() / tot
    zero1 = 100 * ((sb >= 0) & (sb < 1) & good).sum() / tot
    above = 100 * ((sb >= 1) & good).sum() / tot
    return float(sig), float(below), float(zero1), float(above)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory to search for LIGHT frames")
    ap.add_argument("--filters", required=True, help="literal FITS FILTER strings")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--limit", type=int, default=22, help="frames per filter")
    ap.add_argument("--label", default="")
    ap.add_argument("--save", default="", help="directory to write stacks into")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)

    root = Path(args.dir)
    label = args.label or root.name
    rows = []
    for filt in [f.strip() for f in args.filters.split(",") if f.strip()]:
        paths = light_frames(root, filt, args.exptime, args.limit)
        if len(paths) < 6:
            log(f"[{label}|{filt}] only {len(paths)} frames — skipped")
            continue
        log(f"[{label}|{filt}] stacking {len(paths)} frames UNCALIBRATED")
        img = stack_uncalibrated(paths, filt)
        if img is None:
            log(f"[{label}|{filt}] stacking failed")
            continue
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            safe = filt.replace(" ", "_").replace("/", "_")
            p = Path(args.save) / f"{label.replace(' ', '')}_{safe}_uncal.npy"
            np.save(p, img.astype(np.float32))
            log(f"    saved {p}")
        rows.append((f"{label}|{filt}", len(paths), *profile(img)))
        del img

    log(f"\n{'field':26s} {'n':>4s} {'skyσ':>7s} {'<0σ':>7s} {'0-1σ':>7s} {'>1σ':>7s}")
    log("-" * 62)
    for name, n, sig, b, z, a in rows:
        log(f"{name:26s} {n:4d} {sig:7.3f} {b:7.1f} {z:7.1f} {a:7.2f}   <== measured")
    log("")
    for name, b, z, a in REFERENCE:
        log(f"{name:26s} {'':4s} {'':7s} {b:7.1f} {z:7.1f} {a:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
