#!/usr/bin/env python3
"""Quantify and visualise plateau jitter on roof-close traces (bent-wheel check).

A bent wheel adds a periodic load ripple while the motor runs, so it shows up as
elevated *variance* on the running plateau — not as a shift in the median power
the scalar envelope uses. This compares the latest close against the good-library
closes on a roughness metric and plots the plateaus.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "sentry", "roof_signatures")


def load(path):
    with open(path) as f:
        return json.load(f)


def plateau(sig):
    """Return (t, p) restricted to the running plateau, using the file's features."""
    t = np.asarray(sig["samples"]["t"], float)
    p = np.asarray(sig["samples"]["power"], float)
    f = sig["features"]
    base, peak = f["baseline_w"], f["peak_w"]
    thr = base + max(5.0, 0.25 * (peak - base))
    mask = p > thr
    if not mask.any():
        return t[:0], p[:0]
    lo, hi = t[mask].min(), t[mask].max()
    # Trim 0.5s off each end to drop inrush ledge / shutdown ramp
    keep = (t >= lo + 0.5) & (t <= hi - 0.5)
    return t[keep], p[keep]


def roughness(t, p):
    """Jitter metrics on a plateau: ripple std about a smoothed trend, and ptp."""
    if p.size < 10:
        return None
    # Smooth with a ~0.5s moving average to model the intended (flat) load,
    # then measure the residual ripple the wheel rides on top of it.
    win = max(3, int(0.5 * (p.size / (t[-1] - t[0]))))  # ~0.5s in samples
    kern = np.ones(win) / win
    trend = np.convolve(p, kern, mode="same")
    resid = p[win:-win] - trend[win:-win]
    return {
        "ripple_std_w": float(np.std(resid)),
        "ripple_ptp_w": float(np.ptp(resid)),
        "raw_std_w": float(np.std(p)),
    }


def main():
    latest = max(glob.glob(os.path.join(ROOT, "*", "close", "*_close.json")),
                 key=os.path.basename)
    good = sorted(glob.glob(os.path.join(ROOT, "good", "close", "*_close.json")))

    print(f"LATEST close: {os.path.basename(latest)}")
    lt, lp = plateau(load(latest))
    lr = roughness(lt, lp)
    print(f"  ripple_std={lr['ripple_std_w']:.2f} W  "
          f"ripple_ptp={lr['ripple_ptp_w']:.1f} W  raw_std={lr['raw_std_w']:.2f} W")

    print("GOOD-library closes:")
    good_ripples = []
    for g in good:
        gt, gp = plateau(load(g))
        gr = roughness(gt, gp)
        good_ripples.append(gr["ripple_std_w"])
        print(f"  {os.path.basename(g):40} ripple_std={gr['ripple_std_w']:.2f} W")

    if good_ripples:
        gm, gs = np.mean(good_ripples), np.std(good_ripples)
        z = (lr["ripple_std_w"] - gm) / gs if gs > 1e-9 else float("inf")
        print(f"\nGood ripple_std: {gm:.2f} ± {gs:.2f} W   "
              f"-> latest z = {z:.1f}  "
              f"({lr['ripple_std_w'] / gm:.1f}x the good mean)")

    # ---- plot: detrended ripple of latest close vs the good closes ----------
    fig, ax = plt.subplots(figsize=(11, 6))
    for g in good:
        gt, gp = plateau(load(g))
        ax.plot(gt - gt[0], gp - np.median(gp), color="0.7", lw=0.9,
                label="good close" if g == good[0] else None)
    ax.plot(lt - lt[0], lp - np.median(lp), color="tab:blue", lw=1.3,
            label="latest close (bent wheel)")
    ax.set_xlabel("time on running plateau (s)")
    ax.set_ylabel("power deviation from plateau median (W)")
    ax.set_title("Roof close plateau ripple — latest vs good library")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    out = os.path.abspath(os.path.join(ROOT, "..", "..", "roof_close_jitter.png"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
