#!/usr/bin/env python3
"""Inject nebula-like structure of known shape and measure what survives, by scale.

    python scripts/n2n_fractal_injection.py --raw local/n2n_lrgb_render/ic1396_O-III_raw.npy \
        --models pooledNB=local/models/n2n_pooledNB_300s.pt

`n2n_extended_injection.py` injects Gaussian blobs and finds them recovered at
87-95%. The retention metric says real ic1396 emission at the same surface
brightness survives at 38%, and the rendered image agrees with the metric. Both
cannot be right about the same model.

The obvious difference is morphology. A Gaussian blob is smooth and
single-scaled; real nebulosity is filamentary, with structure across a wide
range of scales and sharp edges. This injects a field with a **power-law power
spectrum**, P(k) ~ k^-beta, which is the standard first-order description of
turbulent interstellar emission (beta ~ 3 to 11/3), instead of a blob.

Knowing the injected field exactly buys the measurement that failed when
attempted on real data: a clean **transfer function by spatial scale**.

    recovered = denoise(base + truth) - denoise(base)
    T(k) = sqrt( <|FFT(recovered)|^2> / <|FFT(truth)|^2> )   in radial k bins

Differencing cancels the underlying field and both background models, so there
is nothing to mask, window or fit. T(k) ~ 1 at every scale means the model
preserves structure and the retention metric is misreading real morphology;
T(k) falling at small scales means the model low-passes and the metric is right.

Three estimators are reported on the same injected field, so any disagreement
between them is a property of the estimators rather than of two different runs:
global amplitude, transfer by scale, and the retention-style ratio binned by
surface brightness that `n2n_extended_check` uses.
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


def fractal_field(shape, beta, rms_sigma, sky_sigma, seed=0,
                  outer_px=1500, inner_px=3):
    """Gaussian random field with P(k) ~ k^-beta, normalised to a target RMS.

    Band-limited deliberately: power above `outer_px` would be an overall
    gradient the background subtraction removes anyway, and below `inner_px`
    would be indistinguishable from pixel noise, which is not what this is
    testing.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    F = np.fft.rfft2(noise)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = 1e-9
    amp = k ** (-beta / 2.0)
    amp[k < 1.0 / outer_px] = 0.0
    amp[k > 1.0 / inner_px] = 0.0
    F *= amp
    field = np.fft.irfft2(F, s=(h, w)).astype(np.float32)
    field -= field.mean()
    field *= (rms_sigma * sky_sigma) / field.std()
    return field


def transfer(rec, truth, nbins=(4096, 1024, 512, 256, 128, 64, 32, 16, 8, 4)):
    """|FFT(recovered)| / |FFT(truth)| in radial bands of spatial scale."""
    h, w = truth.shape
    R = np.fft.rfft2(rec.astype(np.float32))
    T = np.fft.rfft2(truth.astype(np.float32))
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = 1e-9
    scale = 1.0 / k
    out = []
    for lo, hi in zip(nbins[1:], nbins[:-1]):
        sel = (scale >= lo) & (scale < hi)
        if sel.sum() < 50:
            continue
        num = float((np.abs(R[sel]) ** 2).sum())
        den = float((np.abs(T[sel]) ** 2).sum())
        if den > 0:
            out.append((lo, hi, np.sqrt(num / den)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rms", type=float, default=1.0, help="field RMS in sky sigma")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from nn import denoiser
    from nn.noise2noise_model import UNet

    base = np.load(args.raw).astype(np.float32)
    sub, _ = denoiser.subtract_background(base)
    good = base != np.median(base)
    sig = float(1.4826 * np.median(np.abs(sub[good] - np.median(sub[good]))))
    log(f"base {os.path.basename(args.raw)}  sky sigma {sig:.3f} ADU  {base.shape}")

    truth = fractal_field(base.shape, args.beta, args.rms, sig, seed=args.seed)
    log(f"injected field: P(k) ~ k^-{args.beta:g}, RMS {args.rms:g} sigma "
        f"({truth.std():.3f} ADU), range {truth.min():.2f}..{truth.max():.2f} ADU")
    injected = base + truth

    # smoothed truth, for the surface-brightness binning the retention metric uses
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "xc", os.path.join(_root, "scripts", "n2n_extended_check.py"))
    xc = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(xc)
    sys.argv = argv
    sb = xc.smoothed_sb(truth, 64) / sig

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for spec_s in args.models:
        name, _, path = spec_s.partition("=")
        if not os.path.exists(path):
            log(f"{name}: missing {path}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = UNet(residual="linear",
                 features=tuple(ck.get("features") or (32, 64, 128, 256)))
        m.load_state_dict(ck["model_state"])
        m.eval()
        d_base = denoiser.denoise_frame(base, m, device=dev)
        d_inj = denoiser.denoise_frame(injected, m, device=dev)
        rec = (d_inj - d_base).astype(np.float32)

        g = float((rec * truth).sum() / (truth * truth).sum())
        log(f"\n=== {name} (epoch {ck.get('epoch')}) ===")
        log(f"  global amplitude recovered: {g:.3f}")

        log(f"\n  transfer by scale")
        log(f"  {'scale (px/cycle)':>18s} {'T(k)':>7s}")
        for lo, hi, t in transfer(rec, truth):
            log(f"  {f'{lo}-{hi}':>18s} {t:7.3f}   {'#' * int(round(t * 40))}")

        log(f"\n  retention-style ratio, binned by injected surface brightness")
        log(f"  {'bin':>8s} {'lin fit':>9s} {'median ratio':>13s} {'n px':>12s}")
        for lo, hi, lbl in ((0.5, 1, "0.5-1σ"), (1, 2, "1-2σ"), (2, 4, "2-4σ"),
                            (4, 8, "4-8σ"), (8, 32, ">8σ")):
            sel = (sb >= lo) & (sb < hi) & good
            n = int(sel.sum())
            if n < 20000:
                continue
            lin = float((rec[sel] * truth[sel]).sum() / (truth[sel] ** 2).sum())
            med = float(np.median(rec[sel]) / np.median(truth[sel]))
            log(f"  {lbl:>8s} {lin:9.3f} {med:13.3f} {n:12d}")
        del d_base, d_inj, rec
    return 0


if __name__ == "__main__":
    sys.exit(main())
