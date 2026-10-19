#!/usr/bin/env python3
"""Phase 0 of docs/LEARNED_DEBLUR_PLAN.md: does one-kernel removal compose?

    python scripts/deconv_phase0.py synth   [--kernel K.npz] [--train 6 --test 2]
    python scripts/deconv_phase0.py train   --head asinh|loggain|linear --seed 0 [--epochs 20]
    python scripts/deconv_phase0.py eval    --model local/deconv_phase0/models/<name>.pt
    python scripts/deconv_phase0.py baseline            # RL40 + starless on the test fields
    python scripts/deconv_phase0.py report              # table over everything in results/

The learned deconvolver's training signal is manufactured: blur a real stack X
once more with the field's own kernel K, train f(K*X + n) -> X, then apply f to
X itself. That removes "one K" and is exact for a Gaussian kernel. Whether it
holds for the REAL kernel -- Moffat-like core, 4% of the flux beyond 24 px,
diffraction spikes -- is the plan's first question, and it is answered here on
fully synthetic fields where the sharp scene S is known, before any real
training.

Everything runs under local/deconv_phase0/ (gitignored).

Synthetic field (per `synth`): Sérsic galaxies with a nucleus, knots and dust
lanes; point stars over five decades of flux, the brightest clipped at the
stack's measured saturation level; the measured ngc6946 kernel (core + wing +
spikes); Gaussian noise with Poisson-like variance at the stack's sky level.
X = clip(K*S + sky) + noise is what the pipeline sees; S is the truth no real
field has.

Training pairs: Y = K*X + n', with n' restoring the fine-band noise K smoothed
away so the only difference between the training input and the inference
input is the blur. The network is told the kernel by a second input plane K*Y
(at inference K*X), see the plan. Loss: L2 in asinh space, masked within
SAT_DILATE px of a plateau.

Heads (--head): the backbone is UNet(in_ch=2, out_ch=1, residual="none") and
the head parametrisation is applied here, on the asinh-normalised first plane
t0 (x0 = sinh(t0) is the linear frame in sky-sigma units):

    asinh     out = t0 + h                      N2N's "asinh" residual
    linear    out = asinh(x0 + h)               N2N's default, bright pixels pinned
    loggain   out = asinh(x0 * exp(h))          multiplicative, RL's own form
    scatter   out = asinh(fold(x0 * softmax(h)))  flux transport: every input pixel's
              flux is redistributed over a SCATTER x SCATTER window by non-negative
              weights that sum to one, so flux is conserved by construction and
              nothing can go negative or ring. Added after the three heads above
              all CREATED flux at bright stars (1.4-1.8x, seed-dependent): the
              wing that holds the flux is below the noise, so a pointwise head
              can only learn a core gain, and the gain right for K*K is wrong
              for K.

Metrics (`eval`), identical for the net, RL40 (stacking/deblur.py with the true
kernel) and the starless default (stacking/starless.py), all against S:
stars by amplitude bin -- FWHM, aperture flux ratio, ring depth in sigma --
and the galaxy residual per octave in sigma. Pass criteria in the plan.
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

BASE = Path(_root) / "local" / "deconv_phase0"
KERNEL_DEFAULT = BASE / "kernel.npz"
FIELD = 2048
PAD = 72                    # reflect pad for the convolutions: the wing's reach
N_STARS = 1500              # amplitude 2..2e4 ADU above sky, dN/dlog(amp) ~ amp^-STAR_SLOPE
N_BRIGHT = 6                # log-uniform 2e4..3e5 ADU: about half clip
# A log-uniform draw put 9x ngc6946's density of >2000 ADU stars on the field
# (measure_psf found 29 and 9 isolated stars where the real stack gives 400)
# and the unresolved wings raised the "sky" by 2 ADU. With -0.6 the field has
# one star above 2000 ADU per ~200k px; ngc6946, a Cygnus field, has one per 120k.
STAR_SLOPE = 0.6
N_GALAXIES = 10
SAT_DILATE = 6
SCATTER = 15                # the scatter head's window: reach 7 px, the kernel's FWHM
STAR_BINS = [(10, 100), (100, 1000), (1000, 10000), (10000, 64000), (64000, 1e9)]   # last = clipped


# ---------------------------------------------------------------- kernel ----
def load_kernel(path=KERNEL_DEFAULT) -> dict:
    z = np.load(path)
    k = np.asarray(z["psf"], dtype=np.float64)
    return {"psf": k / k.sum(), "spikes": np.asarray(z["spikes"], dtype=np.float64),
            "fwhm": float(z["fwhm"]), "sat": float(z["sat"]), "sky_sigma": float(z["sky_sigma"]),
            "sky": float(z["sky"])}


def conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """'same' convolution with reflect padding so the edge does not wrap."""
    from scipy.signal import fftconvolve
    p = k.shape[0] // 2
    xp = np.pad(x, p, mode="reflect")
    return fftconvolve(xp, k, mode="same")[p:-p, p:-p]


def profile_fwhm(k: np.ndarray) -> float:
    """FWHM of a centred kernel from its azimuthal profile (sub-bin interpolation)."""
    c = k.shape[0] // 2
    yy, xx = np.mgrid[:k.shape[0], :k.shape[1]]
    r = np.hypot(yy - c, xx - c)
    rb = r.astype(int)
    prof = np.bincount(rb.ravel(), weights=k.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    prof = prof / prof[0]
    i = int(np.argmax(prof < 0.5))
    if i == 0:
        return 0.0
    f = (prof[i - 1] - 0.5) / (prof[i - 1] - prof[i])
    return 2.0 * (i - 1 + f)


# ------------------------------------------------------------- synthetic ----
def _sersic(yy, xx, y0, x0, amp, re, n, q, pa):
    ct, st = np.cos(pa), np.sin(pa)
    dx, dy = xx - x0, yy - y0
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    r = np.hypot(u, v / q)
    bn = 2 * n - 1 / 3
    return amp * np.exp(-bn * ((r / re) ** (1 / n) - 1))


def make_scene(rng: np.random.Generator, K: dict, size: int = FIELD) -> dict:
    """Return S (sharp), its galaxy-only part, the star catalogue, and X."""
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    gal = np.zeros((size, size))
    for _ in range(N_GALAXIES):
        y0, x0 = rng.uniform(200, size - 200, 2)
        re = rng.uniform(15, 150)
        n = rng.uniform(0.8, 4.0)
        q = rng.uniform(0.35, 1.0)
        pa = rng.uniform(0, np.pi)
        amp = rng.uniform(2, 12) * K["sky_sigma"]          # disc SNR/px 2..12 at Re
        g = _sersic(yy, xx, y0, x0, amp, re, n, q, pa)
        # a sharper nucleus, knots along the disc, one or two dust lanes
        g += _sersic(yy, xx, y0, x0, 8 * amp, 1.5, 1.0, 1.0, 0.0)
        for _k in range(int(rng.integers(4, 14))):
            rr = rng.uniform(0.3, 1.5) * re
            th = rng.uniform(0, 2 * np.pi)
            ky, kx = y0 + rr * np.sin(th) * q, x0 + rr * np.cos(th)
            sig = rng.uniform(1.2, 4.0)
            g += rng.uniform(3, 30) * K["sky_sigma"] * np.exp(-((yy - ky) ** 2 + (xx - kx) ** 2) / (2 * sig ** 2))
        for _d in range(int(rng.integers(1, 3))):
            dpa = pa + rng.uniform(-0.4, 0.4)
            off = rng.uniform(0.2, 0.8) * re
            w = rng.uniform(2, 8)
            u = -(xx - x0) * np.sin(dpa) + (yy - y0) * np.cos(dpa)
            g *= 1 - 0.6 * np.exp(-((u - off) ** 2) / (2 * w ** 2)) * (np.hypot(xx - x0, yy - y0) < 1.5 * re)
        gal += g
    # stars: sub-pixel deltas, bilinear-split over the 2x2 neighbourhood
    n_all = N_STARS + N_BRIGHT
    pos = rng.uniform(PAD, size - PAD - 1, (n_all, 2))
    lo, hi, g = 2.0, 2e4, STAR_SLOPE
    u = rng.uniform(0, 1, N_STARS)
    faint = (u * (hi ** -g - lo ** -g) + lo ** -g) ** (-1 / g)
    amp = np.concatenate([faint, 10 ** rng.uniform(np.log10(2e4), np.log10(3e5), N_BRIGHT)])
    kpeak = float(K["psf"].max())
    flux = amp / kpeak                                     # so K*delta peaks at amp
    stars = np.zeros((size, size))
    for (y, x), f in zip(pos, flux):
        iy, ix = int(y), int(x)
        fy, fx = y - iy, x - ix
        stars[iy, ix] += f * (1 - fy) * (1 - fx)
        stars[iy, ix + 1] += f * (1 - fy) * fx
        stars[iy + 1, ix] += f * fy * (1 - fx)
        stars[iy + 1, ix + 1] += f * fy * fx
    S = gal + stars
    blurred = conv(S, K["psf"]) + K["sky"]
    clipped = blurred > 0.98 * K["sat"]
    blurred = np.minimum(blurred, 0.98 * K["sat"])
    var = K["sky_sigma"] ** 2 * np.maximum(blurred, 1.0) / K["sky"]
    X = blurred + rng.normal(0, 1, blurred.shape) * np.sqrt(var)
    return {"S": S.astype(np.float32), "gal": gal.astype(np.float32),
            "X": X.astype(np.float32), "clipped": clipped,
            "star_y": pos[:, 0], "star_x": pos[:, 1], "star_amp": amp, "star_flux": flux}


def cmd_synth(a):
    K = load_kernel(a.kernel)
    out = BASE / "fields"
    out.mkdir(parents=True, exist_ok=True)
    fk = BASE / "kernel.npz"
    if Path(a.kernel).resolve() != fk.resolve():
        import shutil
        shutil.copy(a.kernel, fk)
    for split, n, seed0 in (("train", a.train, 100), ("test", a.test, 900)):
        for i in range(n):
            rng = np.random.default_rng(seed0 + i)
            t = time.time()
            sc = make_scene(rng, K)
            np.savez(out / f"{split}{i}.npz", **sc)
            print(f"{split}{i}: {int(sc['clipped'].sum())} clipped px, "
                  f"X range {sc['X'].min():.0f}..{sc['X'].max():.0f}, {time.time() - t:.0f}s")
    print(f"kernel FWHM {K['fwhm']:.2f} px, K*K FWHM {profile_fwhm(conv(_centre_pad(K['psf']), K['psf'])):.2f} px")


def _centre_pad(k):
    p = k.shape[0] // 2
    return np.pad(k, p, mode="constant")


# ---------------------------------------------------------------- model ----
def _torch():
    import torch
    return torch


class Deconv:
    """Backbone + head parametrisation, shared by train and eval."""

    def __init__(self, head: str, device: str):
        torch = _torch()
        from nn.noise2noise_model import UNet
        if head not in ("asinh", "linear", "loggain", "scatter"):
            raise ValueError(head)
        self.head = head
        out_ch = SCATTER * SCATTER if head == "scatter" else 1
        self.net = UNet(in_ch=2, out_ch=out_ch, residual="none").to(device)
        torch.nn.init.zeros_(self.net.head.weight)
        torch.nn.init.zeros_(self.net.head.bias)         # start from identity
        if head == "scatter":
            # identity = all weight on the central offset
            with torch.no_grad():
                self.net.head.bias[(SCATTER * SCATTER) // 2] = 6.0
        self.device = device

    def forward(self, inp):
        torch = _torch()
        h = self.net(inp)
        t0 = inp[:, :1]
        if self.head == "asinh":
            return t0 + h
        x0 = torch.sinh(t0)
        if self.head == "linear":
            return torch.asinh(x0 + h)
        if self.head == "scatter":
            import torch.nn.functional as F
            w = torch.softmax(h, dim=1)                              # (B, S*S, H, W), sums to 1
            B, _, H, W = w.shape
            contrib = (x0 * w).reshape(B, SCATTER * SCATTER, H * W)   # each pixel's flux, split
            out = F.fold(contrib, output_size=(H, W), kernel_size=SCATTER, padding=SCATTER // 2)
            return torch.asinh(out)
        return torch.asinh(x0 * torch.exp(torch.clamp(h, -6, 6)))

    def state(self):
        return {"head": self.head, "net": self.net.state_dict()}

    @classmethod
    def load(cls, path, device):
        torch = _torch()
        st = torch.load(path, map_location=device)
        m = cls(st["head"], device)
        m.net.load_state_dict(st["net"])
        return m


def _norm_scale(X, K):
    from nn import denoiser
    _, scale = denoiser.normalise((X - K["sky"]).astype(np.float32))
    return float(scale)


def _planes(img, K, scale):
    """(img, K*img) sky-subtracted and asinh-normalised: the network's input."""
    a = img - K["sky"]
    return np.stack([np.arcsinh(a / scale), np.arcsinh(conv(a, K["psf"]) / scale)]).astype(np.float32)


def _make_pair(X, K, scale, rng):
    """Training input planes and target for one field: Y = K*X + n'."""
    a = X - K["sky"]
    KX = conv(a, K["psf"])
    # restore the fine-band noise the blur removed: white approximation,
    # var(K*n) = sigma^2 * sum(K^2)
    add = K["sky_sigma"] * np.sqrt(max(1.0 - float((K["psf"] ** 2).sum()), 0.0))
    Y = KX + rng.normal(0, add, X.shape)
    inp = np.stack([np.arcsinh(Y / scale), np.arcsinh(conv(Y, K["psf"]) / scale)]).astype(np.float32)
    tgt = np.arcsinh(a / scale).astype(np.float32)[None]
    return inp, tgt


def _loss_mask(X, K):
    from scipy.ndimage import binary_dilation
    from stacking import deblur as db
    sat = db.saturated_mask(X, K["sat"], K["sky_sigma"])
    return ~binary_dilation(sat, iterations=SAT_DILATE)


def cmd_train(a):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    K = load_kernel(BASE / "kernel.npz")
    fields = [dict(np.load(p)) for p in sorted((BASE / "fields").glob("train*.npz"))]
    scales = [_norm_scale(f["X"], K) for f in fields]
    masks = [_loss_mask(f["X"], K) for f in fields]
    model = Deconv(a.head, device)
    opt = torch.optim.Adam(model.net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    name = a.name or f"{a.head}_s{a.seed}"
    (BASE / "models").mkdir(parents=True, exist_ok=True)
    log = open(BASE / "models" / f"{name}.log", "w")

    def say(s):
        print(s); log.write(s + "\n"); log.flush()

    say(f"{name}: head={a.head} loss={a.loss} lam={a.lam} seed={a.seed} epochs={a.epochs} patch={a.patch} "
        f"pairs={a.pairs} fields={len(fields)} device={device}")
    P = a.patch
    t0 = time.time()
    for ep in range(a.epochs):
        # fresh noise draw every epoch so the pairs are never memorised
        pairs = [_make_pair(f["X"], K, s, rng) for f, s in zip(fields, scales)]
        model.net.train()
        tot, n = 0.0, 0
        order = rng.permutation(a.pairs)
        for b in range(0, a.pairs, a.batch):
            xs, ys, ms = [], [], []
            for _ in order[b:b + a.batch]:
                i = int(rng.integers(len(fields)))
                inp, tgt = pairs[i]
                y0 = int(rng.integers(0, FIELD - P)); x0 = int(rng.integers(0, FIELD - P))
                xs.append(inp[:, y0:y0 + P, x0:x0 + P]); ys.append(tgt[:, y0:y0 + P, x0:x0 + P])
                ms.append(masks[i][None, y0:y0 + P, x0:x0 + P])
            xb = torch.from_numpy(np.stack(xs)).to(device)
            yb = torch.from_numpy(np.stack(ys)).to(device)
            mb = torch.from_numpy(np.stack(ms).astype(np.float32)).to(device)
            out = model.forward(xb)
            loss = (((out - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1)
            if a.loss == "mixed":
                # Poisson-weighted L2 in LINEAR sky-sigma units. The asinh loss
                # alone left the bright stars' amplitude unconstrained (a star
                # at 5000 sigma is 9.2 in asinh units; a 2x flux error is 0.7
                # there, the size of one sky pixel's noise) and seed-0 models
                # created flux: aperture ratios 1.6-1.8x the truth at
                # 1000-64000 ADU. chi^2 against the target's own variance
                # charges a relative error at every amplitude alike.
                xo, xt = torch.sinh(out), torch.sinh(yb)
                chi2 = (((xo - xt) ** 2) / (xt.abs() + 1.0) * mb).sum() / mb.sum().clamp(min=1)
                loss = loss + a.lam * chi2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xs); n += len(xs)
        sched.step()
        say(f"  epoch {ep + 1:3d}  loss {tot / max(n, 1):.5f}  {time.time() - t0:.0f}s")
    torch.save(model.state(), BASE / "models" / f"{name}.pt")
    say(f"saved {BASE / 'models' / (name + '.pt')}")


# ----------------------------------------------------------------- eval ----
def infer(model, X, K):
    torch = _torch()
    scale = _norm_scale(X, K)
    inp = torch.from_numpy(_planes(X, K, scale)[None]).to(model.device)
    model.net.eval()
    with torch.no_grad():
        out = model.forward(inp)[0, 0].cpu().numpy()
    O = (np.sinh(out.astype(np.float64)) * scale + K["sky"]).astype(np.float32)
    # A plateau has no "one kernel sharper" answer: the input goes back inside
    # the clipped discs, as deblur.py does (the plan's design choice 2).
    keep = ~_loss_mask(X, K)
    O[keep] = X[keep]
    return O


def star_metrics(O, sc, K, sig):
    """Per amplitude bin: FWHM, flux ratio, ring depth, on isolated stars."""
    ys, xs, amps, fl = sc["star_y"], sc["star_x"], sc["star_amp"], sc["star_flux"]
    gal = sc["gal"]
    H, W = O.shape
    yy, xx = np.mgrid[-40:41, -40:41]
    r = np.hypot(yy, xx)
    rows = {}
    order = np.argsort(-amps)
    for i in order:
        y, x, A = ys[i], xs[i], amps[i]
        iy, ix = int(round(y)), int(round(x))
        if not (150 <= iy < H - 150 and 150 <= ix < W - 150):
            continue
        d = np.hypot(ys - y, xs - x)
        near = (d > 0) & (d < 40) & (amps > 0.1 * A)
        if near.any() or gal[iy, ix] > 0.2 * sig:
            continue
        st = O[iy - 40:iy + 41, ix - 40:ix + 41].astype(np.float64)
        rr = np.hypot(yy - (y - iy), xx - (x - ix))
        ann = (rr > 24) & (rr < 36)
        sky = np.median(st[ann]); lsig = 1.4826 * np.median(np.abs(st[ann] - sky))
        s = st - sky
        rb = np.minimum(rr.astype(int), 40)
        prof = np.bincount(rb.ravel(), weights=s.ravel(), minlength=41) / np.bincount(rb.ravel(), minlength=41)
        peak = float(s[rr < 1.5].max())
        fwhm = np.nan
        if peak > 0:
            below = np.nonzero(prof[:24] < 0.5 * peak)[0]
            if len(below) and below[0] > 0:
                j = below[0]; f = (prof[j - 1] - 0.5 * peak) / (prof[j - 1] - prof[j])
                fwhm = 2.0 * (j - 1 + f)
        flux_ratio = float(s[rr <= 12].sum() / fl[i])
        ring = float(prof[6:21].min() / max(lsig, 1e-6))
        for lo, hi in STAR_BINS:
            if lo <= A < hi:
                key = f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"
                rows.setdefault(key, []).append((fwhm, flux_ratio, ring))
    out = {}
    for key, v in rows.items():
        v = np.array(v, dtype=float)
        out[key] = {"n": int(len(v)), "fwhm": float(np.nanmedian(v[:, 0])),
                    "flux": float(np.median(v[:, 1])), "ring_min": float(np.min(v[:, 2])),
                    "ring_med": float(np.median(v[:, 2]))}
    return out


def galaxy_metrics(O, sc, K):
    """RMS of (O - sky - S) per octave in sigma, on galaxy pixels away from stars."""
    from scipy.ndimage import gaussian_filter
    sig = K["sky_sigma"]
    S = sc["S"].astype(np.float64)
    res = O.astype(np.float64) - K["sky"] - S
    mask = sc["gal"] > 2 * sig
    yy, xx = np.mgrid[:O.shape[0], :O.shape[1]]
    for y, x, A in zip(sc["star_y"], sc["star_x"], sc["star_amp"]):
        # S holds a delta where O holds a blurred star, so every star is a
        # residual; a bright one's wing reaches 2 sigma at 100 px
        rad = 200 if A > 3e4 else 100 if A > 3e3 else 40 if A > 300 else 16 if A > 30 else 6
        if rad:
            iy, ix = int(y), int(x)
            y0, y1, x0, x1 = max(iy - rad, 0), iy + rad + 1, max(ix - rad, 0), ix + rad + 1
            sub = (yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2 < rad * rad
            mask[y0:y1, x0:x1] &= ~sub
    res = np.where(np.isfinite(res), res, 0.0)
    out = {}
    prev = res
    for j, s in enumerate((1, 2, 4, 8, 16, 32)):
        low = gaussian_filter(res, s)
        band = prev - low
        out[f"{s}-{2 * s}px"] = float(np.sqrt(np.mean(band[mask] ** 2)) / sig)
        prev = low
    out[">64px"] = float(np.sqrt(np.mean(prev[mask] ** 2)) / sig)
    out["total"] = float(np.sqrt(np.mean(res[mask] ** 2)) / sig)
    out["n_px"] = int(mask.sum())
    return out


def evaluate(name, images, K):
    """images: {field: O}. Writes results/<name>.json."""
    sig = K["sky_sigma"]
    stars, gals = {}, {}
    for fname, O in images.items():
        sc = dict(np.load(BASE / "fields" / f"{fname}.npz"))
        stars[fname] = star_metrics(O, sc, K, sig)
        gals[fname] = galaxy_metrics(O, sc, K)
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    with open(BASE / "results" / f"{name}.json", "w") as f:
        json.dump({"stars": stars, "galaxy": gals}, f, indent=1)
    print(f"wrote results/{name}.json")


def cmd_eval(a):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = load_kernel(BASE / "kernel.npz")
    model = Deconv.load(a.model, device)
    name = a.name or Path(a.model).stem
    images = {}
    for p in sorted((BASE / "fields").glob("test*.npz")):
        sc = np.load(p)
        t = time.time()
        images[p.stem] = infer(model, sc["X"], K)
        print(f"{p.stem}: inferred in {time.time() - t:.1f}s")
        if a.save:
            np.save(BASE / "results" / f"{name}_{p.stem}.npy", images[p.stem])
    evaluate(name, images, K)


def cmd_baseline(a):
    from stacking import deblur as db
    from stacking import starless
    K = load_kernel(BASE / "kernel.npz")
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    inputs, rl, sl = {}, {}, {}
    for p in sorted((BASE / "fields").glob("test*.npz")):
        sc = np.load(p)
        X = sc["X"].astype(np.float32)
        inputs[p.stem] = X
        if a.save:
            np.save(BASE / "results" / f"input_{p.stem}.npy", X)
        t = time.time()
        rl[p.stem], info = db.deblur(X, iters=40, psf=K["psf"], spikes=K["spikes"], clamp_to="sky")
        print(f"{p.stem}: rl40 in {time.time() - t:.0f}s  {info.get('why', '')}")
        t = time.time()
        res, infos = starless.deblur_set({"L": X}, iters=40, log=lambda s: None)
        sl[p.stem] = res["L"].astype(np.float32)
        print(f"{p.stem}: starless in {time.time() - t:.0f}s  {infos['L'].get('why', '')} "
              f"stars={infos['L'].get('stars')}")
        if a.save:
            np.save(BASE / "results" / f"rl40_{p.stem}.npy", rl[p.stem])
            np.save(BASE / "results" / f"starless_{p.stem}.npy", sl[p.stem])
    evaluate("input", inputs, K)
    evaluate("rl40", rl, K)
    evaluate("starless", sl, K)


def cmd_reeval(a):
    """Re-score saved outputs results/<name>_<field>.npy (after a metric change)."""
    K = load_kernel(BASE / "kernel.npz")
    for name in a.names:
        images = {p.stem[len(name) + 1:]: np.load(p)
                  for p in sorted((BASE / "results").glob(f"{name}_test*.npy"))}
        if not images:
            print(f"{name}: no saved outputs"); continue
        evaluate(name, images, K)


def cmd_report(a):
    K = load_kernel(BASE / "kernel.npz")
    fk = profile_fwhm(K["psf"])                       # same estimator as the star metric
    kk = profile_fwhm(conv(_centre_pad(K["psf"]), K["psf"]))
    pred = fk * fk / kk
    print(f"kernel FWHM {fk:.2f} px (profile; Gaussian fit {K['fwhm']:.2f}); K*K {kk:.2f} px; "
          f"one-kernel-removal prediction {pred:.2f} px (Gaussian kernel would give {fk / np.sqrt(2):.2f})\n")
    files = sorted((BASE / "results").glob("*.json"))
    bins = [f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}" for lo, hi in STAR_BINS]
    print("STARS  (median FWHM px / median flux ratio r<=12 / deepest ring in sigma), per amplitude bin")
    print(f"{'run':18s}" + "".join(f"{b:>22s}" for b in bins))
    for f in files:
        d = json.load(open(f))
        row = f"{f.stem:18s}"
        for b in bins:
            vals = [fd[b] for fd in d["stars"].values() if b in fd]
            if not vals:
                row += f"{'-':>22s}"; continue
            fw = np.nanmedian([v["fwhm"] for v in vals]); fl = np.median([v["flux"] for v in vals])
            rg = min(v["ring_min"] for v in vals)
            row += f"{fw:7.2f} {fl:6.3f} {rg:7.2f}"
        print(row)
    print("\nGALAXY residual rms vs S, in sigma, per octave (mean over test fields)")
    keys = ["1-2px", "2-4px", "4-8px", "8-16px", "16-32px", "32-64px", ">64px", "total"]
    print(f"{'run':18s}" + "".join(f"{k:>9s}" for k in keys))
    for f in files:
        d = json.load(open(f))
        print(f"{f.stem:18s}" + "".join(f"{np.mean([g[k] for g in d['galaxy'].values()]):9.3f}" for k in keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth"); s.add_argument("--kernel", default=str(KERNEL_DEFAULT))
    s.add_argument("--train", type=int, default=6); s.add_argument("--test", type=int, default=2)
    s.set_defaults(fn=cmd_synth)
    t = sub.add_parser("train"); t.add_argument("--head", required=True); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--patch", type=int, default=256)
    t.add_argument("--pairs", type=int, default=2000); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--name")
    t.add_argument("--loss", default="asinh", choices=("asinh", "mixed")); t.add_argument("--lam", type=float, default=1.0)
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval"); e.add_argument("--model", required=True); e.add_argument("--name")
    e.add_argument("--save", action="store_true"); e.set_defaults(fn=cmd_eval)
    b = sub.add_parser("baseline"); b.add_argument("--save", action="store_true"); b.set_defaults(fn=cmd_baseline)
    r = sub.add_parser("report"); r.set_defaults(fn=cmd_report)
    v = sub.add_parser("reeval"); v.add_argument("names", nargs="+"); v.set_defaults(fn=cmd_reeval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
