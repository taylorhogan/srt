#!/usr/bin/env python3
"""Does the denoiser keep extended structure, and keep it equally per channel?

    python scripts/n2n_extended_check.py --model M --raw a.npy b.npy --labels Ha O-III

The gate this project already has is blind to the defect this measures. Source
survival, aperture flux quintiles and corr-against-ceiling are all either
point-source photometry or whole-frame averages; low-surface-brightness extended
emission falls between them. Lab manual step 21 found the denoiser keeping
76/36/51/41% of faint extended flux in L/R/G/B on ngc5907 — a per-channel
difference, and therefore a colour cast (the green halo) that every existing
number reported as healthy.

Two things make this measurement work where a naive one fails:

- **Bin by *smoothed* surface brightness, not by pixel value.** At 1-2 sigma a
  pixel is mostly noise, so binning on raw pixel value puts signal and noise in
  the same bin and the ratio collapses toward zero for any working denoiser —
  which says nothing. Smoothing first (a wide median) estimates the local
  surface brightness the extended emission actually has, then the retention is
  measured on the unsmoothed values within that bin.

- **Report the cross-channel spread, not just the mean.** A denoiser that eats
  30% of faint flux from every channel equally dims the image; one that eats 36%
  from R and 51% from G changes its colour. The second is far worse for a
  display product and only the spread reveals it.

Ratios are reported against the raw, per surface-brightness bin, in units of
each channel's own sky sigma so channels are compared at matched SNR rather
than matched ADU.
"""

import argparse
import os
import sys

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def log(m: str = "") -> None:
    print(m, flush=True)


def smoothed_sb(a: np.ndarray, box: int = 64) -> np.ndarray:
    """Local surface brightness, noise averaged out, at full resolution.

    Block-mean then nearest-neighbour expand: cheap, and the block edges do not
    matter because the result is only ever used to *assign bins*.
    """
    h, w = a.shape
    hh, ww = h // box * box, w // box * box
    blocks = a[:hh, :ww].reshape(hh // box, box, ww // box, box).mean(axis=(1, 3))
    out = np.repeat(np.repeat(blocks, box, axis=0), box, axis=1)
    full = np.empty_like(a)
    full[:hh, :ww] = out
    if hh < h:
        full[hh:, :ww] = out[-1:, :]
    if ww < w:
        full[:, ww:] = full[:, ww - 1:ww]
    return full


def retention(raw: np.ndarray, den: np.ndarray, edges, box: int = 64) -> dict:
    from scipy.ndimage import binary_dilation

    from nn import denoiser

    raw = np.ascontiguousarray(raw.astype(np.float32))
    den = np.ascontiguousarray(den.astype(np.float32))
    bad = binary_dilation((raw == np.median(raw)) | (den == np.median(den)),
                          iterations=8)
    r, _ = denoiser.subtract_background(raw)
    d, _ = denoiser.subtract_background(den)
    good = ~bad
    sig = 1.4826 * np.median(np.abs(r[good] - np.median(r[good])))
    sb = smoothed_sb(r, box) / sig

    out = {"sky_sigma": float(sig), "bins": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = good & (sb >= lo) & (sb < hi)
        n = int(m.sum())
        if n < 5000:
            out["bins"].append(None)
            continue
        a, b = float(np.median(r[m])), float(np.median(d[m]))
        out["bins"].append({"lo": lo, "hi": hi, "n": n, "raw": a, "den": b,
                            "kept": b / a if abs(a) > 1e-4 else float("nan")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", nargs="+", required=True, help="raw stack .npy per channel")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if len(args.raw) != len(args.labels):
        log("--raw and --labels must be the same length")
        return 1

    import torch

    from nn import denoiser
    from nn.noise2noise_model import UNet

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = UNet(residual="linear")
    model.load_state_dict(ck["model_state"])
    model.eval()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"model {os.path.basename(args.model)} "
        f"(epoch {ck.get('epoch')}, loss {ck.get('loss')})  device {dev}")

    EDGES = [-2, 0, 1, 2, 4, 8, 16, 32, 1e9]
    res = {}
    for path, lab in zip(args.raw, args.labels):
        a = np.load(path).astype(np.float32)
        d = denoiser.denoise_frame(a, model, device=dev)
        res[lab] = retention(a, d, EDGES, args.box)
        log(f"  {lab}: sky sigma {res[lab]['sky_sigma']:.3f} ADU")
        del a, d

    log("\nfraction of extended flux kept, by smoothed surface brightness "
        "(units of that channel's sky sigma)")
    hdr = "  " + "SB bin".ljust(14) + "".join(f"{l:>12s}" for l in args.labels)
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        cells = []
        for l in args.labels:
            b = res[l]["bins"][i]
            cells.append(f"{b['kept']:12.3f}" if b else f"{'-':>12s}")
        log("  " + lbl.ljust(14) + "".join(cells))

    log("\ncross-channel spread per bin — this is the colour-safety number")
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        vals = [res[l]["bins"][i]["kept"] for l in args.labels
                if res[l]["bins"][i]]
        if len(vals) < 2:
            continue
        lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        spread = max(vals) - min(vals)
        flag = "   <-- colour shift" if spread > 0.05 else ""
        log(f"  {lbl:14s} min {min(vals):.3f}  max {max(vals):.3f}  "
            f"spread {spread:.3f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
