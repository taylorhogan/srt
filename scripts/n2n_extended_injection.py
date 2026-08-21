#!/usr/bin/env python3
"""Inject extended structure of known brightness and measure what survives.

    python scripts/n2n_extended_injection.py --raw local/n2n_lrgb_render/ic1396_O-III_raw.npy \
        --models pooledNB=local/models/n2n_pooledNB_300s.pt

Every extended-structure measurement in this manual before now is relative —
denoised against raw, with no ground truth, because no clean image of these
fields exists. That leaves one question unanswerable: when retention reads 0.30,
is the model failing, or is the information genuinely not there?

Injection answers it. Add a synthetic structure of known amplitude and scale,
denoise, and measure how much of the *known* signal comes back.

    recovered = denoise(base + synthetic) - denoise(base)

Differencing two denoiser runs cancels the underlying field, so what is measured
is the model's response to the added structure alone. The estimator is the
optimal linear one for a known profile, sum(recovered * g) / sum(g * g) with g
the unit-amplitude template — not aperture flux, which is noisier and biased by
whatever the field already holds.

**Why this discriminates.** Per-pixel SNR and integrated SNR are wildly
different for extended sources. A Gaussian of peak A (in sky sigma) and scale s
carries integrated SNR ~ A*s*sqrt(pi): at A=1, s=30 px that is ~53, i.e. not
marginal at all. So:

- recovery ~1.0 at high integrated SNR and falling only where integrated SNR is
  genuinely low  ->  the model is near-optimal and the answer to faint nebulae
  is more integration.
- recovery ~0.3 at integrated SNR of 50+  ->  the model is discarding structure
  it has ample evidence for, the prior is overriding the likelihood, and that is
  an implementation problem worth fixing.

A raw control (injected minus base, no denoiser) is measured alongside and must
read 1.000; anything else means the estimator itself is broken.
"""

import argparse
import os
import sys

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

AMPS = [0.5, 1.0, 2.0, 4.0, 8.0]      # peak, in sky sigma
SCALES = [10, 30, 100]                 # Gaussian sigma, pixels


def log(m: str = "") -> None:
    print(m, flush=True)


def build_sites(shape, scales, amps, margin=400, spacing=700, seed=0):
    """Well-separated injection sites, cycling through (amp, scale) combos."""
    h, w = shape
    rng = np.random.default_rng(seed)
    ys = list(range(margin, h - margin, spacing))
    xs = list(range(margin, w - margin, spacing))
    grid = [(y, x) for y in ys for x in xs]
    rng.shuffle(grid)
    combos = [(a, s) for s in scales for a in amps]
    return [(y, x, combos[i % len(combos)][0], combos[i % len(combos)][1])
            for i, (y, x) in enumerate(grid)]


def inject(base, sites, sig):
    """Add Gaussians; return the synthetic layer and per-site templates."""
    syn = np.zeros_like(base, dtype=np.float32)
    keep = []
    h, w = base.shape
    for (y, x, amp, s) in sites:
        r = int(4 * s)
        if y - r < 0 or y + r >= h or x - r < 0 or x + r >= w:
            continue
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        g = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * s * s)).astype(np.float32)
        syn[y - r:y + r + 1, x - r:x + r + 1] += amp * sig * g
        keep.append((y, x, amp, s, r, g))
    return syn, keep


def recover(delta, keep, sig):
    """Optimal linear amplitude estimate per site, as a fraction of injected."""
    out = []
    for (y, x, amp, s, r, g) in keep:
        patch = delta[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        gg = g.astype(np.float64)
        est = float((patch * gg).sum() / (gg * gg).sum())   # in ADU
        out.append((amp, s, est / (amp * sig)))
    return out


def summarise(rows, label):
    log(f"\n{label}")
    log(f"  {'scale':>6s} {'peak':>6s} {'int.SNR':>8s} {'n':>4s} {'recovered':>10s} {'spread':>9s}")
    by = {}
    for amp, s, frac in rows:
        by.setdefault((s, amp), []).append(frac)
    for (s, amp) in sorted(by):
        v = np.array(by[(s, amp)])
        snr = amp * s * np.sqrt(np.pi)
        log(f"  {s:6d} {amp:6.1f} {snr:8.1f} {len(v):4d} {np.median(v):10.3f} "
            f"{np.percentile(v,84)-np.percentile(v,16):9.3f}")
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from nn import denoiser
    from nn.noise2noise_model import UNet

    base = np.load(args.raw).astype(np.float32)
    sub, _ = denoiser.subtract_background(base)
    good = base != np.median(base)
    sig = float(1.4826 * np.median(np.abs(sub[good] - np.median(sub[good]))))
    log(f"base {os.path.basename(args.raw)}  sky sigma {sig:.3f} ADU  shape {base.shape}")

    sites = build_sites(base.shape, SCALES, AMPS, seed=args.seed)
    syn, keep = inject(base, sites, sig)
    injected = base + syn
    log(f"{len(keep)} injection sites, "
        f"{len(AMPS)}x{len(SCALES)} combinations, "
        f"~{len(keep)//(len(AMPS)*len(SCALES))} per combination")

    # Control: the estimator on the noiseless difference must return exactly 1.
    summarise(recover(syn, keep, sig), "RAW CONTROL (injected - base, no denoiser) — must read 1.000")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            log(f"{name}: {path} missing")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = UNet(residual="linear")
        m.load_state_dict(ck["model_state"])
        m.eval()
        d_base = denoiser.denoise_frame(base, m, device=dev)
        d_inj = denoiser.denoise_frame(injected, m, device=dev)
        summarise(recover(d_inj - d_base, keep, sig),
                  f"{name}  (epoch {ck.get('epoch')}, loss {ck.get('loss')})")
        del d_base, d_inj
    return 0


if __name__ == "__main__":
    sys.exit(main())
