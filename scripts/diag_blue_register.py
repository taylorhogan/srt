"""Diagnose why blue-channel subs fail astroalign registration.

For the most-recent DSO session (or one named on the CLI), groups LIGHT frames
by FILTER and, for the chosen filter, reports:
  - sep source counts at thresh 3 / 5 / 10 sigma (is 3-sigma counting noise?)
  - the peak-flux distribution of detected sources
  - whether astroalign.find_transform succeeds against a reference at the
    default detection_sigma vs. higher values.

Read-only. No files are modified.
"""
import os
import sys
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


def is_light(f: Path) -> bool:
    return f.parent.name.upper() == "LIGHT"


def latest_session(paths: list[Path], gap_hours: float = 8.0) -> list[Path]:
    """Keep only frames from the most recent contiguous session."""
    paths = sorted(paths, key=lambda f: f.stat().st_mtime)
    if not paths:
        return paths
    cut = paths[-1].stat().st_mtime - gap_hours * 3600
    return [p for p in paths if p.stat().st_mtime >= cut]


def load2d(path: Path) -> np.ndarray:
    return stacker._load_fits_2d(path)


def sep_counts(data: np.ndarray):
    import sep
    d = data.astype(float)
    bkg = sep.Background(d)
    sub = d - bkg.back()
    rms = bkg.rms()
    out = {}
    for thr in (3.0, 5.0, 10.0):
        try:
            src = sep.extract(sub, thresh=thr, err=rms)
            peaks = np.sort(src["peak"])[::-1] if len(src) else np.array([])
            out[thr] = (len(src), peaks)
        except Exception as exc:
            out[thr] = (f"err: {exc}", np.array([]))
    return out, float(np.median(rms))


def try_align(frame, reference, det_sigma):
    import astroalign as aa
    try:
        t, (src, dst) = aa.find_transform(
            frame, reference, detection_sigma=det_sigma, max_control_points=50
        )
        return f"OK  ({len(src)} matched stars)"
    except Exception as exc:
        return f"FAIL ({exc})"


def main():
    arg = " ".join(sys.argv[1:]).strip().lower()
    filt_arg = None
    if arg:
        # allow "ngc5907 B" form
        parts = arg.split()
        if len(parts) >= 2 and len(parts[-1]) <= 3:
            filt_arg = parts[-1].upper()
            arg = " ".join(parts[:-1])

    # Pick DSO dir
    if arg:
        cands = [d for d in image_dir.iterdir()
                 if d.is_dir() and arg.replace(" ", "") in d.name.lower().replace(" ", "")]
        dso_dir = cands[0] if cands else None
    else:
        all_fits = sorted((f for f in image_dir.rglob("*.fits") if is_light(f)),
                          key=lambda f: f.stat().st_mtime)
        dso_dir = all_fits[-1].parent
        while dso_dir.parent != image_dir and dso_dir.parent != dso_dir:
            dso_dir = dso_dir.parent
    if not dso_dir:
        print("No DSO dir found"); return
    print(f"DSO dir: {dso_dir}")

    # Match the pipeline: ALL LIGHT frames in the dir, grouped by filter.
    lights = [f for f in dso_dir.rglob("*.fits") if is_light(f)]
    groups = stacker.group_by_filter(lights)
    print("Filter groups (all frames):",
          {k: len(v) for k, v in groups.items()})

    # Choose the filter to diagnose
    if filt_arg and filt_arg in groups:
        fname = filt_arg
    else:
        # default: the blue-ish filter, else the smallest group
        blueish = [k for k in groups if k.upper() in ("B", "BLUE", "SII", "S2")]
        fname = blueish[0] if blueish else min(groups, key=lambda k: len(groups[k]))
    paths = sorted(groups[fname], key=lambda f: f.stat().st_mtime)
    print(f"\n=== Diagnosing filter '{fname}': {len(paths)} frames ===")

    # Per-frame source counts
    print(f"\n{'frame':<28} {'src@3s':>8} {'src@5s':>8} {'src@10s':>8} {'bkgRMS':>8} "
          f"{'p50peak':>9} {'p95peak':>9}")
    frames = []
    for p in paths:
        data = load2d(p)
        frames.append(data)
        counts, rms = sep_counts(data)
        c3, pk3 = counts[3.0]
        c5, _ = counts[5.0]
        c10, _ = counts[10.0]
        p50 = f"{np.percentile(pk3, 50):.0f}" if len(pk3) else "-"
        p95 = f"{np.percentile(pk3, 95):.0f}" if len(pk3) else "-"
        print(f"{p.name[:28]:<28} {str(c3):>8} {str(c5):>8} {str(c10):>8} "
              f"{rms:>8.1f} {p50:>9} {p95:>9}")

    # Registration test vs the same reference logic the pipeline uses
    ref_idx = stacker._best_reference_idx(frames)
    reference = frames[ref_idx]
    print(f"\nReference = frame {ref_idx} ({paths[ref_idx].name})")
    print(f"\n{'frame':<28} {'det_sigma=5(default)':>22} {'det_sigma=8':>14} {'det_sigma=12':>14}")
    for i, (p, fr) in enumerate(zip(paths, frames)):
        if i == ref_idx:
            continue
        r5 = try_align(fr, reference, 5)
        r8 = try_align(fr, reference, 8)
        r12 = try_align(fr, reference, 12)
        print(f"{p.name[:28]:<28} {r5:>22} {r8:>14} {r12:>14}")


if __name__ == "__main__":
    main()
