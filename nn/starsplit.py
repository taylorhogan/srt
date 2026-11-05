"""Inference for the learned star/nebula split: a plane in, the starless
plane out, the star layer being the difference (2026-11-05, nn/starsplit_data
has the recipe). Tiled like the learned deblur: 1024 px tiles, 128 px
overlap, the inner 768 px of each kept (a crop stitch, no blending), the
whole frame asinh-normalised once by its own sky sigma so the tiles sit on
one scale.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "local" / "starsplit" / "models" / "starsplit_default.pt"
TILE = 1024
OVERLAP = 128
# The network's one defect (2026-11-05, four recipes judged on the held-out
# Trunk): a compact nub of a few hundred ADU inside a faint ring at the two
# or three stars per field above ~1500 ADU, the halo and arms already
# gone. A PSF-fit hybrid (deblur.star_layer) left a bright annulus instead,
# so the fix is a fill: at each compact peak above FILL_PEAK_ADU over the
# sky, the disc where the starless differs from its surroundings by more
# than 3 sigma (8-30 px, +2) is replaced by starless.normalised_fill of
# its neighbourhood, feathered over 4 px. Trunk: residual rms within 12 px
# 69/40 -> 0/0 ADU, within 40 px 21/12 -> 1.1/1.0 (the sky's 0.98).
FILL_PEAK_ADU = 1200.0
FILL_R_MIN, FILL_R_MAX, FILL_BOX = 8, 30, 48


def build(features=(32, 64, 128, 256)):
    from nn.noise2noise_model import UNet
    return UNet(features=features, in_ch=1, out_ch=1, residual="linear", upsample="bilinear")


def save(model, path: Path, meta: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "kind": "starsplit"}, path)


def load(path: Path = DEFAULT_MODEL, device: Optional[str] = None):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "starsplit":
        raise ValueError(f"{path} is not a starsplit checkpoint")
    model = build(tuple(ck["meta"].get("features", (32, 64, 128, 256))))
    model.load_state_dict(ck["state_dict"])
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval(), ck["meta"], dev


def fill_bright_residuals(frame: np.ndarray, starless: np.ndarray, peak_adu: float = FILL_PEAK_ADU,
                          log=print) -> tuple[np.ndarray, int]:
    """The compact residual the network leaves under a bright star, filled
    from its surroundings (see FILL_PEAK_ADU). Returns (starless, count).
    Pure in its inputs; NaN outside the footprint is kept."""
    import sep

    from stacking import starless as sl_mod
    fin = np.isfinite(starless)
    if not fin.any():
        return starless, 0
    med = float(np.nanmedian(frame[fin]))
    a = np.ascontiguousarray(np.where(np.isfinite(frame), frame, med), dtype=np.float32)
    try:
        bkg = sep.Background(a)
        sub = a - bkg.back()
        rms = float(bkg.globalrms)
        obj = sep.extract(sub, max(peak_adu / 10.0 / max(rms, 1e-6), 20.0), err=rms, minarea=5)
    except Exception as e:  # noqa: BLE001
        log(f"    bright-star fill skipped: {e}")
        return starless, 0
    sel = obj[(obj["peak"] > peak_adu) & (obj["b"] / obj["a"] > 0.5)]
    out = starless.copy()
    h, w = a.shape
    n = 0
    for o in sel:
        y, x = int(round(o["y"])), int(round(o["x"]))
        R = FILL_BOX
        y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, x - R), min(w, x + R + 1)
        c = np.where(np.isfinite(out[y0:y1, x0:x1]), out[y0:y1, x0:x1], med).astype(np.float64)
        base = float(np.median(c))
        yy, xx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x]
        rr = np.hypot(yy, xx)
        bad = np.abs(c - base) > 3 * rms
        r_fill = float(FILL_R_MAX)
        for r in range(FILL_R_MIN, FILL_R_MAX):
            ring = (rr >= r) & (rr < r + 2)
            if ring.any() and bad[ring].mean() < 0.2:
                r_fill = float(r)
                break
        r_fill += 2
        hole = rr <= r_fill
        if not hole.any() or hole.all():
            continue
        filled = sl_mod.normalised_fill(c, hole)
        feather = np.clip((rr - r_fill) / 4.0, 0, 1)
        patch = feather * c + (1 - feather) * filled
        region = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = np.where(np.isfinite(region), patch, region).astype(out.dtype)
        n += 1
    if n:
        log(f"    bright-star fill: {n} star(s) above {peak_adu:.0f} ADU")
    return out, n


def split_frame(frame: np.ndarray, model=None, device: Optional[str] = None,
                tile: int = TILE, overlap: int = OVERLAP, log=print,
                fill_bright: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(starless, stars) for one background-subtracted-or-not plane, linear
    ADU; stars = frame - starless exactly. The frame's background (sep mesh)
    is taken off for the network and put back; with *fill_bright* the
    compact residual under each bright star is filled (fill_bright_residuals)."""
    import torch

    from nn import denoiser
    from stacking import color_process
    if model is None:
        model, _, device = load(device=device)
    device = device or next(model.parameters()).device.type
    a = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(a)
    fill = float(np.nanmedian(a)) if not finite.all() else 0.0
    a = np.where(finite, a, fill).astype(np.float32)
    bg = a - np.asarray(color_process._remove_gradient(a, mesh=4), dtype=np.float32)
    sub = a - bg
    _, scale = denoiser.normalise(sub)
    t = np.arcsinh(sub / scale).astype(np.float32)
    h, w = t.shape
    pad = overlap
    tp = np.pad(t, pad, mode="reflect")
    out = np.empty_like(t)
    core = tile - 2 * overlap
    n_ty = int(np.ceil(h / core))
    n_tx = int(np.ceil(w / core))
    log(f"    starsplit: {n_ty * n_tx} tiles of {tile}, scale {scale:.3g}")
    with torch.no_grad():
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, x0 = ty * core, tx * core
                blk = tp[y0:y0 + tile, x0:x0 + tile]
                bh, bw = blk.shape
                if bh < tile or bw < tile:
                    blk = np.pad(blk, ((0, tile - bh), (0, tile - bw)), mode="reflect")
                x = torch.from_numpy(np.ascontiguousarray(blk))[None, None].to(device)
                y = model(x)[0, 0].cpu().numpy()
                yc = y[overlap:overlap + core, overlap:overlap + core]
                hh = min(core, h - y0)
                ww = min(core, w - x0)
                out[y0:y0 + hh, x0:x0 + ww] = yc[:hh, :ww]
    starless = (np.sinh(out) * scale + bg).astype(np.float32)
    starless = np.where(finite, starless, np.nan)
    if fill_bright:
        starless, _ = fill_bright_residuals(np.asarray(frame, dtype=np.float32), starless, log=log)
    stars = np.where(finite, np.asarray(frame, dtype=np.float32) - starless, np.nan)
    return starless, stars
