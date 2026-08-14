"""Injection-recovery for two denoisers, on identical stacks and sites.

ARM 1  the trained N2N model (L1, 60 epochs, split-half stacks, residual=linear)
ARM 2  a coincidence filter: keep a pixel only where BOTH independent half-stacks
       exceed k*sigma_half, otherwise set it to background.

Arm 2 is the non-learned baseline this investigation has been missing. If a
deterministic filter with no training matches the network, the network is not
earning its complexity.

Note on why the test is min(A,B) and not |A-B|: a real source contributes
equally to both halves, so it CANCELS in the difference, leaving pure noise at
sigma_half*sqrt(2). Thresholding the difference therefore selects on noise alone
and says nothing about whether a source is present. The AND/coincidence form is
what actually carries the information — a noise spike lands in one half, a real
source in both.

Both arms are scored the same way: denoise the clean stack and the injected
stack, then measure the incremental response D(S+I) - D(S) in an aperture
against the true injected signal. Denoising is non-linear, so that difference is
the response to the added source alone, with host, sky and any underlying star
cancelled by construction.
"""
import sys
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/taylor/Documents/srt")
import sep
import torch
from nn import denoiser, registration
from nn.noise2noise_model import UNet

sep.set_extract_pixstack(3000000)
OUT = "/tmp/claude-1000/-home-taylor-Documents-srt/da3595b7-58db-41eb-aa9d-5c4610a03787/scratchpad"
MODEL = "/home/taylor/Documents/srt/local/models/n2n_stack_R_300s.pt"
LEVELS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0]
N_PER = 90
FWHM, APER, R = 6.0, 8.0, 20
KS = [1.0, 1.5, 2.0]          # coincidence thresholds, in sigma_half

# ---- build m92 half-stacks (held out; never trained on) -------------------
import glob
frames = []
for p in sorted(glob.glob("/home/taylor/Desktop/Targets/m92/**/LIGHT/*.fits", recursive=True)):
    with fits.open(p) as hh:
        hd = hh[0].header
        if str(hd.get("FILTER", "")).strip() != "R":
            continue
        if round(float(hd.get("EXPTIME", 0))) != 300:
            continue
        frames.append(np.squeeze(hh[0].data).astype(np.float32))
print(f"m92: {len(frames)} frames", flush=True)
frames = registration.register_frames(frames, ["m92"] * len(frames),
                                      progress_cb=lambda m: print("  " + m, flush=True))
rng = np.random.default_rng(0)
order = rng.permutation(len(frames))
half = len(order) // 2
A = np.mean([frames[i] for i in order[:half]], axis=0).astype(np.float64)
B = np.mean([frames[i] for i in order[half:2 * half]], axis=0).astype(np.float64)
del frames
S = (A + B) / 2.0
h, w = S.shape
print(f"half-stacks of {half}; full = {S.shape}", flush=True)

bkS = sep.Background(S); rms = float(bkS.globalrms)
bkA = sep.Background(A); rmsA = float(bkA.globalrms)
print(f"rms: full {rms:.4f}  half {rmsA:.4f}", flush=True)

# ---- injection sites ------------------------------------------------------
real = sep.extract(S - bkS, thresh=5.0 * rms, err=rms)
tree = cKDTree(np.column_stack([real["x"], real["y"]]))
sigma = FWHM / 2.3548
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

inj = np.zeros_like(S)
sites = []
for lvl in LEVELS:
    placed, guard = 0, 0
    while placed < N_PER and guard < N_PER * 300:
        guard += 1
        y = int(rng.integers(400, h - 400)); x = int(rng.integers(400, w - 400))
        if tree.query_ball_point([x, y], 30.0):
            continue
        if sites and min((y - sy) ** 2 + (x - sx) ** 2 for sy, sx, _ in sites) < 60 ** 2:
            continue
        inj[y - R:y + R + 1, x - R:x + R + 1] += psf * (lvl * rms)
        sites.append((y, x, lvl)); placed += 1
    print(f"  level {lvl:5.1f}: {placed}", flush=True)

ys = np.array([s[0] for s in sites], float)
xs = np.array([s[1] for s in sites], float)
lv = np.array([s[2] for s in sites])
f_true, _, _ = sep.sum_circle(inj, xs, ys, APER)
snr = f_true / (rms * np.sqrt(np.pi * APER ** 2))

def curve(resp, tag):
    f, _, _ = sep.sum_circle(resp, xs, ys, APER)
    out = []
    for lvl in LEVELS:
        m = lv == lvl
        if m.sum() < 3:
            continue
        r = f[m] / f_true[m]
        out.append((lvl, float(np.median(snr[m])), float(np.median(r))))
    print(f"\n--- {tag} ---", flush=True)
    print(f"{'peak/rms':>9} {'aperSNR':>8} {'recovered':>10}", flush=True)
    for lvl, s_, r_ in out:
        print(f"{lvl:9.1f} {s_:8.2f} {r_:10.4f}", flush=True)
    ok = [o for o in out if o[2] >= 0.97]
    print(f"  floor (recovery>=0.97): "
          + (f"{min(ok)[0]:.1f}x rms, aperSNR {min(ok)[1]:.2f}" if ok else "NONE"), flush=True)
    return out

# ---- ARM 1: the network ---------------------------------------------------
ck = torch.load(MODEL, map_location="cpu", weights_only=True)
model = UNet(); model.load_state_dict(ck["model_state"]); model.eval()
print(f"\nmodel epoch {ck['epoch']} residual={model.residual}", flush=True)
d0 = denoiser.denoise_frame(S.astype(np.float32), model).astype(np.float64)
d1 = denoiser.denoise_frame((S + inj).astype(np.float32), model).astype(np.float64)
res = {"n2n": curve(d1 - d0, "ARM 1 — N2N network")}

# ---- ARM 2: coincidence filter -------------------------------------------
# A source is present in every sub, so it is present in both halves: inject the
# full amplitude into each. Then (A+inj + B+inj)/2 = S + inj, consistent with arm 1.
As, Bs = A - sep.Background(A), B - sep.Background(B)
Ai, Bi = As + inj, Bs + inj
for k in KS:
    def coinc(a, b):
        keep = (np.minimum(a, b) > k * rmsA)
        return np.where(keep, (a + b) / 2.0, 0.0)
    res[f"coinc k={k}"] = curve(coinc(Ai, Bi) - coinc(As, Bs), f"ARM 2 — coincidence, k={k}")

np.save(f"{OUT}/inject2.npy", np.array(
    [(name, l, s, r) for name, rows in res.items() for l, s, r in rows], dtype=object))
print("\ndone", flush=True)
