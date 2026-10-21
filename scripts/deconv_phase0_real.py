#!/usr/bin/env python3
"""Phase 0d of docs/LEARNED_DEBLUR_PLAN.md: the bar the learned model must clear.

    python scripts/deconv_phase0_real.py run      # starless + rl40 on real half-stacks
    python scripts/deconv_phase0_real.py report

The starless default and the older whole-luminance RL run on the two
independent half-stacks of the ladder's held-out targets (ngc5907 and
ngc7320, L, raw and denoised), cropped to 3000 px round the galaxy, and are
scored the way the plan scores the learned model where no truth exists:

  stars, per amplitude bin    FWHM (profile), aperture flux ratio out/in
                              (r <= 12 px, sky from 24-36), deepest ring in
                              local sigma; isolated stars only
  split-half correlation      D_i = bandpass(out_i - in_i, 1-8 px) on the two
                              halves; corr(D_0, D_1) on pixels with signal.
                              Real recovered detail agrees between halves
                              that share nothing but the sky; invented detail
                              does not. corr(bandpass(in_0), bandpass(in_1))
                              is the same number for the input, the ceiling
                              set by the halves' own noise.

Outputs under local/deconv_phase0/real/. The starless pipeline has so far
been judged by eye only; this is its first table.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

BASE = Path(_root) / "local" / "deconv_phase0" / "real"
STACKS = Path(_root) / "local" / "n2n_ladder" / "stacks"
MODEL = Path(_root) / "local" / "models" / "n2n_ladder_groups_galaxies_300s.pt"
CROP = 3000
CENTRES = {"ngc5907": (3194, 4788), "ngc7320": (3080, 4620), "ngc6946": (3194, 4788)}   # crop centres (frame centre; the approved crop)
STAR_BINS = [(30, 100), (100, 1000), (1000, 10000), (10000, 1e9)]


def _crop(a, centre):
    cy, cx = centre
    h = CROP // 2
    return np.ascontiguousarray(a[cy - h:cy + h, cx - h:cx + h], dtype=np.float32)


def inputs():
    """(target, half, full raw half-stack). The methods run on the FULL frame
    and are cropped after: run on a 3000 px crop, starless classified disk
    clumps of ngc5907 as deep star masks and filled them at sky (measured
    2026-10-21, 64 masks on the crop vs none on the full frame), which the
    first baseline tables carried as the starless row."""
    for t, c in CENTRES.items():
        for i in (0, 1):
            a = np.array(np.load(STACKS / f"{t}__L_train{i}.npy", mmap_mode="r"), dtype=np.float32)
            a[~np.isfinite(a)] = np.nanmedian(a)
            yield t, i, a


def cmd_run(a):
    import torch
    from nn import denoiser
    from stacking import deblur as db
    from stacking import starless
    BASE.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = denoiser.load_model(MODEL)
    for t, i, raw in inputs():
        c = CENTRES[t]
        t0 = time.time()
        den = denoiser.denoise_frame(raw, model, device=dev)[:raw.shape[0], :raw.shape[1]]
        print(f"{t} half{i}: denoised in {time.time() - t0:.0f}s")
        for dom, X in (("raw", raw), ("den", den)):
            np.save(BASE / f"{t}_{dom}{i}_input.npy", _crop(X, c))
            t0 = time.time()
            out, info = db.deblur(X, iters=40, clamp_to="sky")
            np.save(BASE / f"{t}_{dom}{i}_rl40.npy", _crop(out, c))
            print(f"{t} {dom}{i}: rl40 {time.time() - t0:.0f}s {info.get('why', '')}")
            del out
            t0 = time.time()
            res, infos = starless.deblur_set({"L": X}, iters=40, log=lambda s: None)
            np.save(BASE / f"{t}_{dom}{i}_starless.npy", _crop(res["L"], c))
            print(f"{t} {dom}{i}: starless {time.time() - t0:.0f}s {infos['L'].get('why', '')} "
                  f"stars={infos['L'].get('stars')}")
            del res
        del raw, den


# -------------------------------------------------------------- metrics ----
def find_stars(X, sig):
    """Isolated local maxima above 30 sigma, off the galaxy: (y, x, amp)."""
    from scipy.ndimage import maximum_filter, median_filter
    bg = median_filter(X[::8, ::8], 25)
    bg = np.kron(bg, np.ones((8, 8)))[:X.shape[0], :X.shape[1]]
    d = X - bg
    mx = maximum_filter(d, 7)
    pk = np.argwhere((d == mx) & (d > 30 * sig))
    pk = pk[(pk[:, 0] > 150) & (pk[:, 0] < X.shape[0] - 150) & (pk[:, 1] > 150) & (pk[:, 1] < X.shape[1] - 150)]
    amps = d[pk[:, 0], pk[:, 1]]
    keep = []
    for j, (y, x) in enumerate(pk):
        dd = np.hypot(pk[:, 0] - y, pk[:, 1] - x)
        near = (dd > 0) & (dd < 40) & (amps > 0.1 * amps[j])
        if near.any():
            continue
        # not on the galaxy: the large-scale background must be flat here
        if abs(bg[y, x] - np.median(bg)) > 2 * sig:
            continue
        keep.append((y, x, float(amps[j])))
    return keep


def star_metrics(O, X, stars, sig):
    yy, xx = np.mgrid[-40:41, -40:41]
    rr = np.hypot(yy, xx)
    ann = (rr > 24) & (rr < 36)
    rb = np.minimum(rr.astype(int), 40)
    rows = {}
    for y, x, A in stars:
        so = O[y - 40:y + 41, x - 40:x + 41].astype(np.float64)
        si = X[y - 40:y + 41, x - 40:x + 41].astype(np.float64)
        sky_o, sky_i = np.median(so[ann]), np.median(si[ann])
        lsig = 1.4826 * np.median(np.abs(so[ann] - sky_o))
        so, si = so - sky_o, si - sky_i
        prof = np.bincount(rb.ravel(), weights=so.ravel(), minlength=41) / np.bincount(rb.ravel(), minlength=41)
        peak = float(so[rr < 1.5].max())
        fwhm = np.nan
        if peak > 0:
            below = np.nonzero(prof[:24] < 0.5 * peak)[0]
            if len(below) and below[0] > 0:
                j = below[0]; f = (prof[j - 1] - 0.5 * peak) / (prof[j - 1] - prof[j])
                fwhm = 2.0 * (j - 1 + f)
        flux = float(so[rr <= 12].sum() / max(si[rr <= 12].sum(), 1e-6))
        ring = float(prof[6:21].min() / max(lsig, 1e-6))
        for lo, hi in STAR_BINS:
            if lo * sig <= A < hi * sig:
                key = f"{lo:g}-{hi:g}s" if hi < 1e8 else f">={lo:g}s"
                rows.setdefault(key, []).append((fwhm, flux, ring))
    out = {}
    for key, v in rows.items():
        v = np.array(v, dtype=float)
        out[key] = {"n": int(len(v)), "fwhm": float(np.nanmedian(v[:, 0])),
                    "flux": float(np.median(v[:, 1])), "ring_min": float(np.min(v[:, 2])),
                    "ring_med": float(np.median(v[:, 2]))}
    return out


def bandpass(x):
    from scipy.ndimage import gaussian_filter
    x = x.astype(np.float64)
    return gaussian_filter(x, 1) - gaussian_filter(x, 8)


def split_half(out0, out1, in0, in1, sig):
    """Spearman correlation of the added band-passed detail between halves, on
    galaxy pixels with every star disc cut out. The first version kept the
    stars and read 0.999 for everything: a star's own change dwarfs the galaxy
    and is identical in both halves by construction."""
    from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation
    from scipy.stats import spearmanr
    sm = gaussian_filter(in0.astype(np.float64), 4)
    mask = (sm - np.median(sm)) > 3 * sig / 4          # pixels with signal
    d = in0 - gaussian_filter(in0.astype(np.float64), 16)
    peaks = (d == maximum_filter(d, 7)) & (d > 15 * sig)
    mask &= ~binary_dilation(peaks, iterations=30)      # 30 px round every star
    d0, d1 = bandpass(out0 - in0)[mask], bandpass(out1 - in1)[mask]
    b0, b1 = bandpass(in0)[mask], bandpass(in1)[mask]
    rho = float(spearmanr(d0, d1).correlation) if d0.std() > 0 and d1.std() > 0 else float("nan")
    return {"rho_added": rho, "rho_input": float(spearmanr(b0, b1).correlation),
            "added_rms_sigma": float(np.sqrt(0.5 * (d0.var() + d1.var())) / sig), "n_px": int(mask.sum())}


def cmd_report(a):
    from stacking import deblur as db
    res = {}
    for t in CENTRES:
        sig_raw = None
        for dom in ("raw", "den"):
            X = [np.load(BASE / f"{t}_{dom}{i}_input.npy") for i in (0, 1)]
            # The denoised stack's own sigma is ~50x smaller (0.07 ADU on
            # ngc7320) and made a 1.5 ADU dip read as a 22-sigma ring. Both
            # domains are measured in the RAW half's sigma, the photon noise.
            sig = db.sky_sigma(X[0]) if dom == "raw" else sig_raw
            sig_raw = sig_raw or sig
            stars = find_stars(X[0], sig)
            extra = sorted({q.stem.split("_", 2)[2] for q in BASE.glob(f"{t}_{dom}0_*.npy")}
                           - {"input", "rl40", "starless"})      # learned models, by their --name
            for m in ("input", "rl40", "starless", *extra):
                O = X if m == "input" else [np.load(BASE / f"{t}_{dom}{i}_{m}.npy") for i in (0, 1)]
                key = f"{t}/{dom}/{m}"
                res[key] = {"stars": star_metrics(O[0], X[0], stars, sig),
                            "split": split_half(O[0], O[1], X[0], X[1], sig), "sky_sigma": sig,
                            "n_stars": len(stars)}
    with open(BASE / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    bins = [f"{lo:g}-{hi:g}s" if hi < 1e8 else f">={lo:g}s" for lo, hi in STAR_BINS]
    print("STARS (median FWHM px / flux ratio out:in r<=12 / deepest ring in sigma) per amplitude bin, in sky sigma")
    print(f"{'target/domain/method':26s}" + "".join(f"{b:>22s}" for b in bins) + f"{'rho_added':>11s}{'rho_input':>11s}{'added rms':>11s}")
    for key, r in res.items():
        row = f"{key:26s}"
        for b in bins:
            v = r["stars"].get(b)
            row += f"{'-':>22s}" if not v else f"{v['fwhm']:7.2f} {v['flux']:6.3f} {v['ring_min']:7.2f}"
        s = r["split"]
        row += f"{s['rho_added']:11.3f}{s['rho_input']:11.3f}{s['added_rms_sigma']:11.3f}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
