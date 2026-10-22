"""The learned deconvolver: a U-Net that moves flux, and its tiled inference.

docs/LEARNED_DEBLUR_PLAN.md, Phase 1, after Phase 0 (2026-10-19) measured
that every pointwise head -- a residual in asinh units, a residual in linear
units, a multiplicative gain, with or without a chi-square flux term --
either created star flux (1.4-1.8x the truth, seed-dependent) or did not
sharpen. The flux that has to move into a star's core sits in the wing,
below the noise, so a pointwise head can only learn a core gain, and the
gain that is right for the training input (two kernels of blur) is wrong
for the real one. Flux conservation has to be structural.

So the network does not predict pixel values. For every input pixel it
predicts a set of non-negative weights that sum to one, and that pixel's
flux is scattered over its neighbourhood by those weights (`F.fold` is the
scatter-add). Total flux is conserved exactly, nothing can go negative,
nothing can ring. Two windows share one softmax: a dense 15x15 (reach 7 px,
the core) and a dilation-3 15x15 (reach 21 px, the wing, where 96% of the
kernel's flux lives inside 24 px). Phase 0's single dense window left a
halo at its edge on bright stars; the dilated window is the reach fix.

Input: two planes, the frame and the frame convolved with the kernel to
remove, both sky-subtracted and asinh-normalised by denoiser.normalise's
scale. The kernel to remove is the measured kernel's CORE (deblur_data.
taper_kernel, by the taper stamped on the checkpoint; checkpoints before
2026-10-22 carry none and remove the whole stamp, which halved every bright
star's halo out to the stamp's edge). The second plane is how the network is told the kernel (the plan's
design choice 1): the difference between the planes shows what one kernel
of blur does to this image, locally. A narrower kernel than measured gives
a milder correction; that is the strength knob.

Output: asinh of the scattered linear frame, so the loss lives in the same
units as the denoiser's and is not dominated by bright pixels. Identity at
init: the dense window's centre logit starts at IDENTITY_LOGIT, which puts
~98% of each pixel's flux on itself, so training starts from the input.

`deblur_frame` is the inference path: sep background off, planes built on the
WHOLE frame (so no tile ever sees a padded edge in its kernel plane), 1024
px tiles with 128 px overlap and the crop stitch (a bright star's wing needs
more context than the denoiser's 64), the input written back inside clipped
plateaus (a plateau has no "one kernel sharper" answer), background on
again. Spark only: it needs torch, like the denoiser.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Callable, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from nn import denoiser  # noqa: E402

DENSE = 15               # dense window side (odd): reach (DENSE-1)/2 px
DILATED = 15             # dilated window side
DILATION = 3             # ...its step: reach 21 px
IDENTITY_LOGIT = 10.0    # softmax(10 vs 0 x 449) = 0.98 on the centre at init
TILE = 1024
OVERLAP = 128
SAT_DILATE = 6
# The checkpoint the morning render's fourth composite uses: a symlink under
# local/deconv/models/ pointed at the arm that scored best (lam 1.0, seed 0 on
# 2026-10-21). Missing file = no learned composite, the render says so.
DEFAULT_MODEL = Path(_root) / "local" / "deconv" / "models" / "learned_default.pt"


def n_weights() -> int:
    return DENSE * DENSE + DILATED * DILATED


def scatter(x0, logits):
    """Move the flux of *x0* (B,1,H,W, linear units) by *logits* (B,n_weights(),H,W)."""
    import torch
    import torch.nn.functional as F
    w = torch.softmax(logits, dim=1)
    B, _, H, W = w.shape
    nd = DENSE * DENSE
    dense = (x0 * w[:, :nd]).reshape(B, nd, H * W)
    out = F.fold(dense, output_size=(H, W), kernel_size=DENSE, padding=DENSE // 2)
    dil = (x0 * w[:, nd:]).reshape(B, DILATED * DILATED, H * W)
    out = out + F.fold(dil, output_size=(H, W), kernel_size=DILATED, dilation=DILATION,
                       padding=(DILATED // 2) * DILATION)
    return out


def build(features=(32, 64, 128, 256)):
    """The model: a U-Net backbone (nn/noise2noise_model) with the scatter head."""
    import torch
    from torch import nn as tnn
    from nn.noise2noise_model import UNet

    class ScatterDeconv(tnn.Module):
        def __init__(self):
            super().__init__()
            self.net = UNet(features=features, in_ch=2, out_ch=n_weights(), residual="none")
            tnn.init.zeros_(self.net.head.weight)
            tnn.init.zeros_(self.net.head.bias)
            with torch.no_grad():
                self.net.head.bias[(DENSE * DENSE) // 2] = IDENTITY_LOGIT

        def forward(self, inp, return_logits: bool = False):
            logits = self.net(inp)
            x0 = torch.sinh(inp[:, :1])
            out = torch.asinh(scatter(x0, logits))
            return (out, logits) if return_logits else out

    return ScatterDeconv()


def save(model, path: Path, meta: Optional[dict] = None, kernel_taper: Optional[tuple] = None) -> None:
    """*kernel_taper* is the (inner, outer) FWHM taper the model was trained to
    remove (nn/deblur_data.taper_kernel); None = the whole measured kernel.
    It is stamped on the checkpoint so inference removes the same kernel."""
    import torch
    torch.save({"state": model.state_dict(), "asinh_sigma_mult": denoiser.ASINH_SIGMA_MULT,
                "dense": DENSE, "dilated": DILATED, "dilation": DILATION,
                "kernel_taper": None if kernel_taper is None else [float(kernel_taper[0]), float(kernel_taper[1])],
                "meta": meta or {}}, path)


def load(path: Path, device: Optional[str] = None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)
    for k, v in (("dense", DENSE), ("dilated", DILATED), ("dilation", DILATION)):
        if ck.get(k, v) != v:
            raise ValueError(f"checkpoint {k}={ck.get(k)} does not match this module's {v}")
    if float(ck.get("asinh_sigma_mult", denoiser.ASINH_SIGMA_MULT)) != float(denoiser.ASINH_SIGMA_MULT):
        raise ValueError("checkpoint was trained at another asinh scale; refusing to silently rescale")
    m = build().to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    taper = ck.get("kernel_taper", (ck.get("meta") or {}).get("kernel_taper"))
    # checkpoints from before 2026-10-22 carry no taper and were trained on the
    # whole measured kernel: None keeps their behaviour exactly
    m.kernel_taper = None if taper is None else (float(taper[0]), float(taper[1]))
    return m, ck.get("meta", {})


# -------------------------------------------------------------- inference ----
def _conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    p = k.shape[0] // 2
    return fftconvolve(np.pad(x, p, mode="reflect"), k, mode="same")[p:-p, p:-p]


def deblur_frame(frame: np.ndarray, model, psf: np.ndarray, device: Optional[str] = None,
                 tile: int = TILE, overlap: int = OVERLAP, sat_level: Optional[float] = None,
                 sky_sigma: Optional[float] = None,
                 log: Callable[[str], None] = lambda s: None,
                 plateau: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """Deconvolve *frame* (linear ADU, sky in) by one *psf*. Returns (out, info).

    *plateau* is an extra mask of pixels to keep as given, united with the
    frame's own saturation/flat-top detection. A synthetic luminance (a
    weighted sum of planes) needs it: a core clipped in Ha alone sits in the
    sum at Ha's weight times the ceiling plus whatever the other planes put
    there, which is neither at a ceiling nor flat, so the sum's own detector
    finds nothing (trunk SHO, 2026-10-22: 0 plateau px) and the network
    deconvolves the clipped star as a real profile into a black moat with a
    lit centre. The planes' own masks, united, are what the sum inherits.
    """
    import torch
    from stacking import deblur as db
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.asarray(frame, dtype=np.float32)
    h, w = x.shape
    med = float(np.nanmedian(x))
    x = np.where(np.isfinite(x), x, med)
    sub, bg = denoiser.subtract_background(x)
    t, scale = denoiser.normalise(sub)
    from nn import deblur_data as dd
    taper = getattr(model, "kernel_taper", None)
    k = dd.taper_kernel(np.asarray(psf, dtype=np.float64), taper)    # the kernel the model was trained on
    plane2 = np.arcsinh(_conv(sub.astype(np.float64), k) / scale).astype(np.float32)
    planes = np.stack([t, plane2])
    pad = overlap
    pp = np.pad(planes, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros((h, w), dtype=np.float32)
    core = tile - 2 * overlap
    n = 0
    model.eval()
    with torch.no_grad():
        for y in range(0, h, core):
            for xx in range(0, w, core):
                ty, tx = y, xx                     # padded coords of the tile's origin
                tv = pp[:, ty:ty + tile, tx:tx + tile]
                th, tw = tv.shape[1], tv.shape[2]
                if th < tile or tw < tile:         # ragged edge: pad the tile, crop after
                    tv = np.pad(tv, ((0, 0), (0, tile - th), (0, tile - tw)), mode="reflect")
                o = model(torch.from_numpy(np.ascontiguousarray(tv))[None].to(device))[0, 0].cpu().numpy()
                o = o[overlap:overlap + core, overlap:overlap + core]
                oh, ow = min(core, h - y), min(core, w - xx)
                out[y:y + oh, xx:xx + ow] = o[:oh, :ow]
                n += 1
    lin = (np.sinh(out.astype(np.float64)) * scale).astype(np.float32) + bg
    # the plateau guard
    sig = sky_sigma if sky_sigma is not None else db.sky_sigma(x)
    level = sat_level if sat_level is not None else db.saturation_level(x)
    sat = db.saturated_mask(x, level, sig) if level else db.flat_tops(x, sig)
    if plateau is not None:
        sat = sat | np.asarray(plateau, dtype=bool)
    if sat.any():
        from scipy.ndimage import binary_dilation
        keep = binary_dilation(sat, iterations=SAT_DILATE)
        lin[keep] = x[keep]
    info = {"tiles": n, "scale": float(scale), "plateau_px": int(sat.sum()), "sky_sigma": float(sig),
            "kernel_taper": taper}
    log(f"learned deblur: {n} tiles of {tile}, scale {scale:.2f}, {int(sat.sum())} plateau px kept, "
        f"kernel {'whole' if taper is None else 'core to %.1f-%.1f FWHM' % taper}")
    return lin, info


def deblur_channels(channels: dict, model, device: Optional[str] = None,
                    log: Callable[[str], None] = lambda s: None,
                    planes: Optional[Iterable[str]] = None,
                    plateau: Optional[dict] = None) -> tuple[dict, dict]:
    """The selected *planes* of *channels* (linear ADU, sky in, one pixel grid)
    through the learned deconvolver, each with ITS OWN measured kernel; the
    rest come back as given. Returns (out, info per channel); a selected
    channel whose field gives no kernel comes back as given with info["why"]
    "no kernel", an unselected one with "not selected", so the set never
    loses a plane.

    *plateau* maps a channel name to an extra keep-as-given mask (see
    deblur_frame). *planes* None means every channel. The production choice (2026-10-21) is
    the luminance alone: deconvolving R, G and B independently was measured
    on ngc6946 HALRGB to sharpen each plane's own residual noise into
    single-pixel specks that do not coincide across planes (coloured dots
    over the faint sky), and to spread one plane's saturated core while the
    others kept theirs (a cyan-cored bright star). The colour planes left as
    denoised keep the colour where the blur put it, under a sharpened L,
    which is what the eye reads. Per-channel deconvolution is kept because
    the head conserves flux per pixel, so when it IS wanted the colour ratio
    survives. The registration border (NaN) is filled for the work and put
    back, as stacking/starless does.
    """
    from nn import deblur_data as dd
    out, infos = {}, {}
    chosen = None if planes is None else set(planes)
    for name, frame in channels.items():
        if chosen is not None and name not in chosen:
            out[name] = frame
            infos[name] = {"why": "not selected"}
            continue
        x = np.asarray(frame, dtype=np.float32)
        nan = ~np.isfinite(x)
        side = dd.measure_sidecar(x, log=lambda m: log(f"{name}: {m.strip()}"))
        if side is None:
            out[name] = frame
            infos[name] = {"why": "no kernel"}
            continue
        lin, info = deblur_frame(x, model, side["psf"], device=device, sat_level=side["sat"] or None,
                                 sky_sigma=side["sky_sigma"], log=lambda m: log(f"{name}: {m}"),
                                 plateau=None if plateau is None else plateau.get(name))
        if nan.any():
            lin = lin.copy(); lin[nan] = np.nan
        out[name] = lin
        infos[name] = {**info, "fwhm_px": side["fwhm"], "stars": side["stars"], "why": None}
        log(f"{name}: kernel {side['fwhm']:.2f} px from {side['stars']} stars")
    return out, infos
