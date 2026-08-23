#!/usr/bin/env python3
"""Measure the tiling artefact: how much does the output depend on tile placement?

    python scripts/n2n_tile_artifact.py --raw local/n2n_lrgb_render/ic1396_O-III_raw.npy \
        --model local/models/n2n_pooledNB_300s.pt

A perfect tiled inference is translation-consistent: shift the tile grid, get
the same answer. The lab manual (step 20) found it is not — a diagonal weave in
blank sky with residual power peaking at exactly the overlap (64 px) and tile
(512 px) scales, pointing at the Tukey blend averaging two tiles wherever they
disagree.

The test: denoise the frame twice, once normally and once with the tile grid
shifted by an offset that is a multiple of neither stitching period (the frame
is reflection-padded top-left by the offset, denoised, and cropped back). Any
difference between the two results, away from the frame border, is pure
tile-placement dependence — the artefact, isolated from everything else.
Measured for both stitch modes, plus a fractal-injection recovery check so a
mode cannot win by changing what the model computes rather than how it is
stitched.
"""

import argparse
import os
import sys

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

OFFSET = 192          # multiple of neither 448 (blend step) nor 384 (crop core)
TRIM = 512            # border to discard: padding changes edge context


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    import importlib.util

    import torch

    from nn import denoiser
    from nn.noise2noise_model import UNet

    spec = importlib.util.spec_from_file_location(
        "inj", os.path.join(_root, "scripts", "n2n_fractal_injection.py"))
    inj = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(inj)
    sys.argv = argv

    frame = np.load(args.raw).astype(np.float32)
    h, w = frame.shape
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    m = UNet(residual="linear")
    m.load_state_dict(ck["model_state"])
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sub, _ = denoiser.subtract_background(frame)
    good = frame != np.median(frame)
    sig = float(1.4826 * np.median(np.abs(sub[good] - np.median(sub[good]))))
    truth = inj.fractal_field(frame.shape, 3.0, 1.0, sig, seed=0)
    log(f"{os.path.basename(args.raw)}  sky sigma {sig:.3f} ADU  offset {OFFSET} px\n")

    interior = (slice(TRIM, h - TRIM), slice(TRIM, w - TRIM))
    results = {}
    for mode in ("blend", "crop"):
        d0 = denoiser.denoise_frame(frame, m, device=dev, stitch=mode)
        shifted = np.pad(frame, ((OFFSET, 0), (OFFSET, 0)), mode="reflect")
        d1 = denoiser.denoise_frame(shifted, m, device=dev,
                                    stitch=mode)[OFFSET:, OFFSET:]
        diff = (d0 - d1)[interior].astype(np.float64)
        rms = diff.std() / sig
        p999 = np.percentile(np.abs(diff), 99.9) / sig

        # Where does the placement-dependence live in scale?
        F = np.abs(np.fft.rfft2(diff - diff.mean())) ** 2
        hh, ww = diff.shape
        k = np.hypot(np.fft.fftfreq(hh)[:, None], np.fft.rfftfreq(ww)[None, :])
        k[0, 0] = 1e-9
        scale = 1.0 / k
        tot = F.sum()
        bands = {}
        for lo, hi in ((32, 64), (48, 96), (256, 512), (384, 768)):
            sel = (scale >= lo) & (scale < hi)
            bands[(lo, hi)] = 100 * F[sel].sum() / tot

        # Did stitching change what the model recovers?
        r0 = denoiser.denoise_frame(frame + truth, m, device=dev, stitch=mode)
        g = float((((r0 - d0) * truth)[interior]).sum()
                  / ((truth[interior] ** 2)).sum())
        results[mode] = (rms, p999, bands, g)
        log(f"[{mode}] placement-dependence rms {rms:.4f} sigma, "
            f"p99.9 {p999:.3f} sigma; recovery {g:.3f}")
        for (lo, hi), v in bands.items():
            log(f"        {lo}-{hi} px band: {v:.1f}% of artefact power")
        del d0, d1, r0

    b, c = results["blend"], results["crop"]
    log(f"\ncrop vs blend: artefact rms {c[0]/b[0]:.2f}x, "
        f"p99.9 {c[1]/b[1]:.2f}x, recovery {b[3]:.3f} -> {c[3]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
