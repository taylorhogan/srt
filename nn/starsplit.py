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


def split_frame(frame: np.ndarray, model=None, device: Optional[str] = None,
                tile: int = TILE, overlap: int = OVERLAP, log=print) -> tuple[np.ndarray, np.ndarray]:
    """(starless, stars) for one background-subtracted-or-not plane, linear
    ADU; stars = frame - starless exactly. The frame's background (sep mesh)
    is taken off for the network and put back."""
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
    stars = np.where(finite, np.asarray(frame, dtype=np.float32) - starless, np.nan)
    return starless, stars
