#!/usr/bin/env python3
"""Is the band limit a defect, or optimal estimation? The Wiener bound, measured.

    python scripts/n2n_wiener_bound.py \
        --half-a local/n2n_multichannel/stacks/ngc6888__O-III_train0.npy \
        --half-b local/n2n_multichannel/stacks/ngc6888__O-III_train1.npy \
        --label "ngc6888 O-III" \
        --models pooledNB=local/models/n2n_pooledNB_300s.pt

Step 29 measured the denoiser's transfer function falling from ~0.98 at large
scales to ~0.33 at 4-8 px and called the discarded band a defect. Nobody checked
whether an *optimal* estimator would also discard it. For a scene with signal
spectrum S(k) under noise N(k), the MMSE linear filter is

    T*(k) = S(k) / (S(k) + N(k))

and any additive perturbation — which is exactly what the fractal injection
adds — is passed with that same gain. If the measured T(k) sits at T*(k), the
model is doing the best any estimator with the correct scene prior can do at
this SNR, and no loss or architecture change beats it; the lever is photons. If
it sits well below, the gap is real headroom, quantified band by band.

**Both spectra come from the half-stack pair, with no model in the loop.** For
two independent half-stacks A = s + n_A and B = s + n_B of equal depth:

    cross-spectrum   Re( FFT(A) · conj(FFT(B)) )  ->  S(k)   (noise cancels)
    difference       P[A - B](k) / 2              ->  N(k)   (signal cancels)

the same independence trick as the correlation ceiling (step 5 of the runbook),
taken per frequency band. As a consistency check, P[A](k) should equal
S(k) + N(k) band by band; the script prints the ratio.

**Stars are masked before the spectra are taken.** The question is the model's
response to *extended* structure — its point-source response is separately
measured and near-perfect (0.1% at the core, step 29) — and a CNN is spatially
adaptive, so the fair prior for the extended-structure band is the scene
spectrum with point sources removed. Both halves get the union mask, filled
with sep's smooth background, so the masking cannot manufacture cross-power.

**The measured T(k) is taken on the same half-stack the bound is computed
for.** Injecting into half A and denoising half A means the input the model
sees has exactly the N(k) in the denominator of the bound. Reusing step 29/30's
full-stack numbers here would compare a bound at one noise level against a
measurement at another.

Caveats, stated rather than hidden: T*(k) is the *linear* MMSE bound with a
global prior. A nonlinear, spatially adaptive estimator can beat it locally
(and demonstrably does around stars). So measured > T* in a band is possible
and means the model exploits structure a global linear filter cannot; measured
well below T* is unambiguous headroom either way, which is the decision this
exists to make.
"""

import argparse
import os
import sys

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

BANDS = [(1024, 4096), (512, 1024), (256, 512), (128, 256),
         (64, 128), (32, 64), (16, 32), (8, 16), (4, 8)]


def log(m: str = "") -> None:
    print(m, flush=True)


def star_fill(a: np.ndarray, mask: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """Replace masked pixels with the smooth local background so the FFT sees no
    holes. `smooth` is sep's own background mesh — already computed, already the
    right thing for a hole where a star was."""
    out = a.copy()
    out[mask] = smooth[mask]
    return out


def band_power(F: np.ndarray, scale: np.ndarray) -> dict:
    out = {}
    for lo, hi in BANDS:
        sel = (scale >= lo) & (scale < hi)
        out[(lo, hi)] = float((F[sel]).sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-a", required=True)
    ap.add_argument("--half-b", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rms", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write a PNG comparison here")
    args = ap.parse_args()

    import importlib.util

    import sep
    import torch
    from scipy.ndimage import binary_dilation

    from nn import denoiser
    from nn.noise2noise_model import UNet
    sep.set_extract_pixstack(5_000_000)

    spec = importlib.util.spec_from_file_location(
        "inj", os.path.join(_root, "scripts", "n2n_fractal_injection.py"))
    inj = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(inj)
    sys.argv = argv

    A = np.load(args.half_a).astype(np.float32)
    B = np.load(args.half_b).astype(np.float32)
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    A, B = A[:h, :w], B[:h, :w]
    label = args.label or os.path.basename(args.half_a)

    # --- spectra from the pair -------------------------------------------
    a_s, _ = denoiser.subtract_background(A)
    b_s, _ = denoiser.subtract_background(B)
    fill = (A == np.median(A)) | (B == np.median(B))

    def stars(x):
        """Point-source mask that leaves the nebula alone.

        On an emission field, sep at 4 sigma detects the filaments themselves as
        sources with enormous ellipses; painting 3*sqrt(a*b) for those masks the
        very structure whose spectrum this script needs, and the paint loop over
        thousand-pixel slices ran for 20+ minutes before this was capped. The
        radius cap at 25 px (~3.5x seeing FWHM) keeps stars fully masked while a
        filament detection loses only a small core patch, which is negligible in
        band power and stated bias-side: any residual star wings INFLATE S(k),
        i.e. push the verdict toward "headroom", never toward "optimal".
        """
        bg = sep.Background(np.ascontiguousarray(x))
        s = x - bg.back()
        src = sep.extract(np.ascontiguousarray(s), 4.0 * float(bg.globalrms), minarea=5)
        m = np.zeros(x.shape, bool)
        for r in src[:60000]:
            rad = int(np.clip(3.0 * np.sqrt(max(r["a"] * r["b"], 1)), 4, 25))
            y0, y1 = int(max(0, r["y"] - rad)), int(min(x.shape[0], r["y"] + rad + 1))
            x0, x1 = int(max(0, r["x"] - rad)), int(min(x.shape[1], r["x"] + rad + 1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            m[y0:y1, x0:x1] |= ((yy - r["y"]) ** 2 + (xx - r["x"]) ** 2) <= rad * rad
        return m, np.array(bg.back(), dtype=np.float32)

    ma, sm_a = stars(a_s)
    mb, sm_b = stars(b_s)
    mask = binary_dilation(ma | mb | fill, iterations=2)
    log(f"{label}: masked {100 * mask.mean():.1f}% (point sources + fill), frame {A.shape}")

    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    af = star_fill(a_s, mask, sm_a).astype(np.float64)
    bf = star_fill(b_s, mask, sm_b).astype(np.float64)
    FA = np.fft.rfft2((af - af.mean()) * win)
    FB = np.fft.rfft2((bf - bf.mean()) * win)

    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = 1e-9
    scale = 1.0 / k

    S = band_power(np.real(FA * np.conj(FB)), scale)          # scene signal
    N = band_power(0.5 * np.abs(FA - FB) ** 2, scale)          # noise per half
    PA = band_power(np.abs(FA) ** 2, scale)                    # check: ~ S + N

    # --- measured T(k): inject into half A, denoise half A ----------------
    good = ~fill
    sig = float(1.4826 * np.median(np.abs(a_s[good] - np.median(a_s[good]))))
    truth = inj.fractal_field(A.shape, args.beta, args.rms, sig, seed=args.seed)
    injected = A + truth
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    measured = {}
    for spec_s in args.models:
        name, _, path = spec_s.partition("=")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = UNet(residual="linear")
        m.load_state_dict(ck["model_state"])
        m.eval()
        d_base = denoiser.denoise_frame(A, m, device=dev)
        d_inj = denoiser.denoise_frame(injected, m, device=dev)
        measured[name] = {(lo, hi): t for lo, hi, t
                          in inj.transfer((d_inj - d_base).astype(np.float32), truth)}
        del d_base, d_inj

    # --- report ------------------------------------------------------------
    names = list(measured)
    log(f"\n{label} — Wiener bound vs measured transfer (input: one half-stack, "
        f"sky sigma {sig:.3f} ADU)")
    hdr = (f"  {'band (px)':>11s} {'S/N':>8s} {'T*(k)':>7s} "
           + "".join(f"{n:>12s}" for n in names)
           + f" {'headroom':>9s} {'S+N/PA':>7s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    rows = []
    for band in BANDS:
        s, n_, pa = max(S[band], 0.0), N[band], PA[band]
        tstar = s / (s + n_) if (s + n_) > 0 else 0.0
        best = max(measured[n].get(band, float("nan")) for n in names)
        cells = "".join(f"{measured[n].get(band, float('nan')):12.3f}" for n in names)
        check = (s + n_) / pa if pa > 0 else float("nan")
        lo, hi = band
        log(f"  {f'{lo}-{hi}':>11s} {s / n_ if n_ > 0 else float('inf'):8.2f} "
            f"{tstar:7.3f} {cells} {tstar - best:+9.3f} {check:7.2f}")
        rows.append((band, s / n_ if n_ > 0 else np.inf, tstar,
                     {n: measured[n].get(band) for n in names}))

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mids = [np.sqrt(lo * hi) for (lo, hi), *_ in rows]
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
        ax.plot(mids, [r[2] for r in rows], "k--", lw=1.8,
                label="Wiener bound T*(k) = S/(S+N)")
        colors = ["#c1443c", "#2f6f9f", "#3fa34d"]
        for i, n in enumerate(names):
            ax.plot(mids, [r[3][n] for r in rows], "o-", ms=4, lw=1.6,
                    color=colors[i % 3], label=f"measured — {n}")
        ax.set_xscale("log")
        ax.set_xlabel("spatial scale (px per cycle)")
        ax.set_ylabel("transfer")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label}: measured transfer vs the linear-MMSE bound\n"
                     "gap below the dashed line = headroom an estimator could still claim",
                     fontsize=11)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.expanduser(args.out))
        log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
