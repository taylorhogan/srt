#!/usr/bin/env python3
"""Where a learned deconvolver puts a bright star's wing: the halo check.

    python scripts/deconv_halo_check.py --model local/deconv/models/learned_default.pt
    python scripts/deconv_halo_check.py --model X.pt --stacks ngc1961__L_train0_den,m92__L_train0_den --lo 800 --hi 6000

For the brightest UNSATURATED stars (peak in [--lo, --hi] ADU over the sky)
of each prepared stack, three radial tables in units of the stack's sky
sigma, annulus means from the star outwards:

    frame          the star as stacked (its wing and halo)
    train out-tgt  the model on the TRAINING pair (the frame re-blurred by its
                   own kernel, factor 1.0) against the frame: what the loss saw
    infer out-in   the model on the frame itself, as inference runs, against
                   the frame: what the picture gets
    cycle          K * infer - frame: re-blurring the answer should give the
                   frame back (the one-kernel assumption, checked)

2026-10-22: the promoted model (whole measured kernel) was right to a sigma
on the training pairs and, on the frame, halved every bright star's wing at
30-60 px into a dark moat inside the halo beyond the kernel's 72 px stamp;
nn/deblur_data.taper_kernel is the answer, and this is its acceptance.
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

EDGES = [0, 4, 8, 16, 24, 32, 45, 60, 80, 110, 150]
DEFAULT_STACKS = "ngc1961__L_train0_den,ic1396__Ha_train0_den,m92__L_train0_den,ngc5907__L_train0_den"


def annuli(d, r):
    return [float(np.mean(d[(r >= a) & (r < b)])) for a, b in zip(EDGES[:-1], EDGES[1:])]


def fmt(vals):
    return " ".join(f"{v:9.2f}" for v in vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stacks", default=DEFAULT_STACKS)
    ap.add_argument("--lo", type=float, default=800.0)
    ap.add_argument("--hi", type=float, default=6000.0)
    ap.add_argument("--stars", type=int, default=3)
    a = ap.parse_args()
    import torch
    from scipy.ndimage import maximum_filter
    from nn import deblur_data as dd, deblur_model as dm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = dm.load(a.model, dev)
    print(f"{a.model}: kernel {'whole' if model.kernel_taper is None else 'core %s FWHM' % (model.kernel_taper,)}")
    items = {it.name: it for it in dd.list_prepared()}
    names = [n for n in a.stacks.split(",") if n]
    missing = [n for n in names if n not in items]
    if missing:
        print(f"not prepared: {missing}")
        return 1
    ds = dd.DeblurDataset([items[n] for n in names], patch_size=512, seed=0, kernel_taper=model.kernel_taper)
    P, M = 512, dd.MARGIN
    yy, xx = np.ogrid[:P, :P]
    rc = np.hypot(yy - P // 2, xx - P // 2)
    print(f"annuli (px): {EDGES}")
    summary = []
    for it in ds.items:
        f, side = it.frame, it.side
        sig, s = side["sky_sigma"], it.scale
        k = dd.taper_kernel(side["psf"], model.kernel_taper)
        sub = np.asarray(f[::2, ::2], dtype=np.float32)
        pk = (sub == maximum_filter(sub, 61)) & (sub > a.lo) & (sub < a.hi)
        ys, xs = np.nonzero(pk)
        print(f"\n== {it.name}  sigma {sig:.3f}  FWHM {side['fwhm']:.2f}")
        done = 0
        for o in np.argsort(-sub[ys, xs]):
            cy, cx = int(ys[o]) * 2, int(xs[o]) * 2
            H, W = f.shape
            if not (M + P // 2 < cy < H - M - P // 2 and M + P // 2 < cx < W - M - P // 2):
                continue
            win = np.asarray(f[cy - 3:cy + 4, cx - 3:cx + 4])
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            cy, cx = cy - 3 + dy, cx - 3 + dx
            if side["flat"][cy - 40:cy + 41, cx - 40:cx + 41].any():
                continue
            y0, x0 = cy - P // 2, cx - P // 2
            big = np.asarray(f[y0 - M:y0 + P + M, x0 - M:x0 + P + M], dtype=np.float64)
            big = np.where(np.isfinite(big), big, 0.0)
            a0 = big[M:M + P, M:M + P]
            peak = float(a0[P // 2, P // 2])
            if (a0 > peak).any():
                continue
            inp, tgt, _mask = ds.make_pair(it, y0, x0, factor=1.0)
            ka = dd.conv(big, k)
            inp2 = np.stack([np.arcsinh(a0 / s), np.arcsinh(ka[M:M + P, M:M + P] / s)]).astype(np.float32)
            with torch.no_grad():
                out = model(torch.from_numpy(inp)[None].to(dev))[0, 0].cpu().numpy()
                out2 = model(torch.from_numpy(inp2)[None].to(dev))[0, 0].cpu().numpy()
            o_lin, t_lin = np.sinh(out) * s, np.sinh(tgt[0]) * s
            o2 = np.sinh(out2) * s
            full = big.copy()
            full[M:M + P, M:M + P] = o2
            cyc = dd.conv(full, k)[M:M + P, M:M + P] - a0
            core = o2[rc < 4].sum() / a0[rc < 4].sum()
            print(f"  star @({cy},{cx}) peak {peak:.0f} ADU ({peak / sig:.0f} sigma)  core r<4: x{core:.2f}")
            print(f"    frame          {fmt(annuli(a0 / sig, rc))}")
            print(f"    train out-tgt  {fmt(annuli((o_lin - t_lin) / sig, rc))}")
            inf = annuli((o2 - a0) / sig, rc)
            print(f"    infer out-in   {fmt(inf)}")
            print(f"    cycle          {fmt(annuli(cyc / sig, rc))}")
            fr = annuli(a0 / sig, rc)
            # the moat: the fraction of the wing at 32-60 px taken away
            w_in = fr[5] * 1 + fr[6]
            w_out = inf[5] + inf[6]
            summary.append((it.name, cy, cx, core, -w_out / max(w_in, 1e-9)))
            done += 1
            if done >= a.stars:
                break
    print("\nsummary: core gain and the fraction of the 32-60 px wing removed on the frame")
    for n, cy, cx, core, frac in summary:
        print(f"  {n:28s} @({cy},{cx})  core x{core:.2f}  wing 32-60 px removed {100 * frac:5.1f}%")
    if summary:
        print(f"  mean: core x{np.mean([s[3] for s in summary]):.2f}  wing removed {100 * np.mean([s[4] for s in summary]):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
