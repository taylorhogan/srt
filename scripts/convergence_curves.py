#!/usr/bin/env python3
"""Stack-convergence curves for several filters of one target, on one axes.

    python scripts/convergence_curves.py --dso "NGC 6888" \
        --root "/media/taylor/cdk17/FromCDK17" --filters S-II,Ha,O-III \
        --cal-root "/media/taylor/cdk17/FromCDK17" --cal-epoch 2024-09 \
        --out ~/Desktop/ngc6888_convergence.png

Same measurement the web chat's `snr` command makes — `stacker.convergence_curve`
— with two differences that matter for archive data.

`snr` scans `cfg["nina"]["image_dir"]` and posts one plot per filter to the chat.
That cannot reach a target on the archive drive, and one plot per filter is the
wrong shape for comparing channels: the question "does S-II converge as well as
Ha" is answered by putting them on shared axes, not by three separate images.

Calibration is taken from an explicit tree rather than config, for the same
reason the render path does it: config resolves the current epoch, and applying
2026 masters at -10C to 2024 lights at -20C is worse than not calibrating.

The y-axis is RMSE against the all-frames stack, normalised by that stack's
sigma-clipped median — dimensionless, 0 means identical to the golden stack. A
channel whose curve sits higher at the same frame count is converging more
slowly, which is the quantitative form of "this channel is noisier than that
one at equal depth".
"""

import argparse
import os
import sys
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dso", required=True, help="directory name under --root")
    ap.add_argument("--root", default="", help="default: this machine's subs_dir")
    ap.add_argument("--filters", required=True)
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--cal-root", default="")
    ap.add_argument("--cal-epoch", default="")
    ap.add_argument("--trials", type=int, default=10,
                    help="subsets per frame count; 20 is the snr command's default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import socket

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits

    from configs import config
    from stacking import stacker
    from stacking.color_process import canonical_filter

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rnd", os.path.join(_root, "scripts", "n2n_lrgb_render.py"))
    rnd = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(rnd)
    sys.argv = argv

    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    root = Path(args.root) if args.root else Path(machine["subs_dir"])
    target = root / args.dso
    if not target.is_dir():
        log(f"no such directory: {target}")
        return 1

    wanted = [f.strip() for f in args.filters.split(",") if f.strip()]
    by_filter = {w: [] for w in wanted}
    for fp in sorted(target.rglob("*.fits")):
        if fp.parent.name.upper() != "LIGHT" or "RECYCLE" in str(fp):
            continue
        try:
            h = fits.getheader(fp)
            if round(float(h.get("EXPTIME", 0))) != args.exptime:
                continue
            c = canonical_filter(str(h.get("FILTER", "")).strip())
        except Exception:
            continue
        if c in by_filter:
            by_filter[c].append(fp)

    results = {}
    for filt in wanted:
        paths = by_filter.get(filt, [])
        if len(paths) < 8:
            log(f"[{filt}] {len(paths)} frames — too few, skipped")
            continue
        cal = None
        if args.cal_root:
            bias, dark, flat = rnd.archive_calibration(
                Path(args.cal_root), filt, args.exptime, args.cal_epoch)
            cal = stacker.load_calibration_set(bias, dark)
            if cal is not None and flat:
                npy = stacker.build_master_flat_npy(
                    flat, cal, filt.replace("-", ""))
                if npy is not None:
                    cal = cal._replace(flat_npy=npy)
            log(f"[{filt}] {len(paths)} frames, calibration "
                f"{len(bias)} bias / {len(dark)} dark / {len(flat)} flat")
        else:
            log(f"[{filt}] {len(paths)} frames, uncalibrated")
        counts, rmse, slope_pct, final_pct = stacker.convergence_curve(
            paths, filter_name=filt, n_trials=args.trials, calibration=cal,
            progress_cb=lambda m: log(f"    {m}"))
        results[filt] = (counts, rmse)
        log(f"[{filt}] done: RMSE {rmse[0]:.4f} at n={counts[0]} -> "
            f"{rmse[-1]:.4f} at n={counts[-1]}, slope {slope_pct:.1f}%, "
            f"final {final_pct:.2f}%")

    if not results:
        log("nothing measured")
        return 1

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    colours = {"Ha": "#3fa34d", "S-II": "#c1443c", "O-III": "#3d7ea6",
               "L": "#888888", "R": "#c1443c", "G": "#3fa34d", "B": "#3d7ea6"}
    for filt, (counts, rmse) in results.items():
        n_all = np.asarray(counts, dtype=float)
        r_all = np.asarray(rmse, dtype=float)
        # The final count IS the golden stack, measured against itself. Its
        # RMSE is not 0 but ~1e-6 — floating-point residue, not a measurement —
        # so a `> 0` filter does not catch it and matplotlib happily plots it,
        # collapsing five decades of y-range into one vertical line. Drop it by
        # identity: any point at the full frame count is the golden.
        keep = (r_all > 0) & (n_all < n_all.max())
        n_p, r_p = n_all[keep], r_all[keep]
        if not len(n_p):
            continue
        ax.plot(n_p, r_p, "o-", ms=4, lw=1.6,
                label=f"{filt}  ({int(n_all[-1])} frames)", color=colours.get(filt))
        # 1/sqrt(N) through the first point: independent noise averaging down.
        ax.plot(n_p, r_p[0] * np.sqrt(n_p[0] / n_p), ":", lw=1.0,
                color=colours.get(filt), alpha=0.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frames stacked")
    ax.set_ylabel("normalised RMSE vs all-frames stack")
    ax.set_title(f"{args.dso} — stack convergence by filter"
                 f"\ndotted: 1/√N through each curve's first point")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    # Persist the measurements beside the plot. Each curve costs ~10 min of FWHM
    # measurement and registration, so re-plotting — a different axis, an added
    # curve, a fixed bug — must not require re-measuring.
    import json
    data = out.with_suffix(".json")
    data.write_text(json.dumps(
        {"dso": args.dso, "exptime": args.exptime, "trials": args.trials,
         "curves": {k: {"counts": list(map(int, c)), "rmse": list(map(float, r))}
                    for k, (c, r) in results.items()}}, indent=2))
    log(f"\nwrote {out}")
    log(f"wrote {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
