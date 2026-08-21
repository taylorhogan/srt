#!/usr/bin/env python3
"""Compare denoisers on extended structure with error bars that mean something.

    python scripts/n2n_compare_paired.py --raw local/n2n_lrgb_render/ic1396_O-III_raw.npy \
        --models pooledNB=local/models/n2n_pooledNB_300s.pt \
                 deep=local/models/n2n_depth_full_300s.pt

`n2n_extended_check.py` reports a retention number per surface-brightness bin.
It does not report an uncertainty, and on 2026-08-21 that turned out to matter
more than any result taken with it: six hypotheses were argued from differences
of 0.01-0.07 with no idea what a repeat measurement would produce.

Two things this script fixes.

**Effective sample size is structures, not pixels.** A retention bin can hold a
million pixels and still be poorly determined, because those pixels sit in a few
spatially correlated patches. On ic1396 O-III only 54 tiles of 256 px carry
enough 4-8 sigma emission to measure, so the honest n is 54. Bootstrapping over
tiles rather than pixels reflects that; bootstrapping over pixels would fake
precision by a factor of ~100.

**Comparisons must be paired.** Both models see the same field, so the
tile-to-tile variation in how much structure a tile holds is common to both and
should cancel. Unpaired 95% intervals on ic1396 were +/-0.05 and called every
comparison indistinguishable; the paired test on the same data resolves
differences of 0.02 cleanly. Same numbers, different question: "is A better than
B *here*" rather than "what is A's retention".

**What this still does not control.** Training stochasticity. Two models trained
from identical data with different seeds also differ tile by tile, and this test
will call that difference significant too. A paired result is evidence that two
*checkpoints* differ, not that two *configurations* do. To claim the latter,
first measure the seed-to-seed spread for one configuration (`n2n_depth_ab.py
--train-seed`) and require the effect to exceed it.
"""

import argparse
import os
import sys

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

BINS = [(0, 1, "0-1σ"), (1, 2, "1-2σ"), (2, 4, "2-4σ"), (4, 8, "4-8σ"),
        (8, 16, "8-16σ"), (16, 32, "16-32σ")]


def log(m: str = "") -> None:
    print(m, flush=True)


def per_tile_retention(raw_path: str, model_path: str, tile: int, min_px: int):
    """Retention per tile per bin, so comparisons can be paired on tile identity."""
    import importlib.util

    import torch
    from scipy.ndimage import binary_dilation

    from nn import denoiser
    from nn.noise2noise_model import UNet

    spec = importlib.util.spec_from_file_location(
        "xc", os.path.join(_root, "scripts", "n2n_extended_check.py"))
    xc = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(xc)
    sys.argv = argv

    raw = np.load(raw_path).astype(np.float32)
    r, _ = denoiser.subtract_background(raw)
    good = ~binary_dilation(raw == np.median(raw), iterations=8)
    sig = 1.4826 * np.median(np.abs(r[good] - np.median(r[good])))
    sb = xc.smoothed_sb(r, 64) / sig

    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = UNet(residual="linear")
    model.load_state_dict(ck["model_state"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    den = denoiser.denoise_frame(raw, model, device=dev)
    d, _ = denoiser.subtract_background(den)

    h, w = r.shape
    tiles = [(y, x) for y in range(0, h - tile, tile) for x in range(0, w - tile, tile)]
    out = {}
    for lo, hi, lbl in BINS:
        sel = (sb >= lo) & (sb < hi) & good
        vals = np.full(len(tiles), np.nan)
        for i, (y, x) in enumerate(tiles):
            m = sel[y:y + tile, x:x + tile]
            if m.sum() < min_px:
                continue
            dv = np.median(d[y:y + tile, x:x + tile][m])
            rv = np.median(r[y:y + tile, x:x + tile][m])
            if abs(rv) > 1e-4:
                vals[i] = dv / rv
        out[lbl] = vals
    del raw, r, den, d
    return out, {"epoch": ck.get("epoch"), "loss": ck.get("loss"),
                 "source_bias": ck.get("source_bias"), "seed": ck.get("seed"),
                 "train_seed": ck.get("train_seed")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--min-px", type=int, default=500,
                    help="pixels of a bin a tile must hold to count")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--bins", default="1-2σ,4-8σ", help="which bins to compare")
    ap.add_argument("--baseline", default="", help="compare all against this name")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)

    models = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path:
            log(f"bad --models entry {spec!r}, expected name=path")
            return 1
        models[name] = path

    want = [b.strip() for b in args.bins.split(",") if b.strip()]
    per, meta = {}, {}
    for name, path in models.items():
        if not os.path.exists(path):
            log(f"{name}: {path} missing — skipped")
            continue
        per[name], meta[name] = per_tile_retention(args.raw, path, args.tile, args.min_px)
        log(f"loaded {name:22s} epoch={meta[name]['epoch']} "
            f"loss={meta[name]['loss']} source_bias={meta[name]['source_bias']} "
            f"train_seed={meta[name]['train_seed']}")

    log(f"\npoint estimates (median over tiles)\n")
    hdr = f"  {'model':24s}" + "".join(f"{b:>10s}" for b in want)
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    for name in per:
        cells = []
        for b in want:
            v = per[name][b]
            cells.append(f"{np.nanmedian(v):10.3f}")
        log(f"  {name:24s}" + "".join(cells))
    for b in want:
        n = int(np.isfinite(next(iter(per.values()))[b]).sum())
        log(f"    {b}: {n} usable tiles")

    rng = np.random.default_rng(0)
    names = list(per)
    pairs = ([(a, args.baseline) for a in names if a != args.baseline]
             if args.baseline else
             [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))])

    log(f"\npaired per-tile differences, {args.reps} bootstrap replicates")
    log(f"  {'comparison':40s} {'bin':>7s} {'median Δ':>9s} {'95% CI':>19s}  verdict")
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        for lbl in want:
            va, vb = per[a][lbl], per[b][lbl]
            ok = np.isfinite(va) & np.isfinite(vb)
            d = va[ok] - vb[ok]
            n = len(d)
            if n < 8:
                log(f"  {a+' - '+b:40s} {lbl:>7s}   only {n} tiles — skipped")
                continue
            idx = rng.integers(0, n, (args.reps, n))
            boot = np.median(d[idx], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            verdict = "SIGNIFICANT" if lo * hi > 0 else "n.s."
            log(f"  {a+' - '+b:40s} {lbl:>7s} {np.median(d):+9.3f} "
                f"{f'[{lo:+.3f}, {hi:+.3f}]':>19s}  {verdict}")
    log("\nSignificant here means the two checkpoints differ on this field. It does "
        "\nNOT mean their configurations differ — compare against the seed-to-seed "
        "\nspread of a single configuration before concluding that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
