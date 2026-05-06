"""Inference and frame collection for Noise2Noise denoising.

denoise_frame() uses tiled inference with overlap-blending so the full
QHY600 sensor (9576×6388 px) can be processed without hitting GPU VRAM limits.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from nn.noise2noise_model import UNet


def best_device() -> str:
    """'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_path(filter_name: str, exptime_s: int) -> Path:
    """Return local/models/n2n_{filter}_{exptime}s.pt relative to project root."""
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    safe = filter_name.replace(" ", "_").replace("/", "_")
    return root / "local" / "models" / f"n2n_{safe}_{exptime_s}s.pt"


def load_model(model_path: Path) -> nn.Module:
    """Load a saved checkpoint and return the model (on CPU)."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = UNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def denoise_frame(
    frame: np.ndarray,
    model: nn.Module,
    device: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Denoise a single 2-D float32 FITS frame using tiled inference.

    The frame is normalised once using global percentiles before tiling so all
    tiles share the same scale — this prevents seam artefacts. Edge tiles are
    zero-padded to tile_size and the padding is discarded after inference.
    Tiles are blended with a raised-cosine weight for smooth transitions.
    Input and output are in the original ADU scale.
    """
    if device is None:
        device = best_device()
    model = model.to(device)

    # Normalise the whole frame once — keeps all tiles on the same scale
    p1  = float(np.percentile(frame, 1))
    p99 = float(np.percentile(frame, 99))
    scale = max(p99 - p1, 1.0)
    norm_frame = ((frame - p1) / scale).astype(np.float32)

    h, w = frame.shape
    step = tile_size - overlap
    output = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    # 1-D raised cosine window → 2-D blend weight
    def _window1d(n: int) -> np.ndarray:
        x = np.linspace(0, np.pi, n)
        return 0.5 - 0.5 * np.cos(x)

    win2d = np.outer(_window1d(tile_size), _window1d(tile_size)).astype(np.float32)

    ys = list(range(0, max(1, h - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    with torch.no_grad():
        for y0 in ys:
            y1 = min(y0 + tile_size, h)
            for x0 in xs:
                x1 = min(x0 + tile_size, w)
                tile = norm_frame[y0:y1, x0:x1]
                th, tw = tile.shape

                # Pad edge tiles to tile_size so the model always sees the
                # same input shape it was trained on
                if th < tile_size or tw < tile_size:
                    padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                    padded[:th, :tw] = tile
                    tile = padded

                inp = torch.from_numpy(tile[None, None]).to(device)
                out = model(inp).cpu().numpy()[0, 0, :th, :tw]  # crop padding

                w2d = win2d[:th, :tw]
                output[y0:y1, x0:x1] += out * w2d
                weight[y0:y1, x0:x1] += w2d

    weight = np.where(weight == 0, 1.0, weight)
    norm_output = (output / weight).astype(np.float32)

    # Denormalise using the same global scale
    return norm_output * scale + p1


def collect_all_frames(
    subs_dir: Path,
    filter_name: str,
    exptime_s: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Walk all DSO subdirs under subs_dir and collect matching LIGHT frames.

    Only frames whose FITS FILTER header matches filter_name AND whose EXPTIME
    rounds to exptime_s are returned.

    Returns (frames, dso_names) where dso_names[i] is the DSO directory name
    for frames[i]. This is used by N2NDataset to restrict pairs to within the
    same DSO — N2N requires pairs to be two noisy views of the same scene.
    """
    from astropy.io import fits as _fits

    frames: list[np.ndarray] = []
    dso_names: list[str] = []

    for fits_path in sorted(subs_dir.rglob("*.fits")):
        if fits_path.parent.name.upper() != "LIGHT":
            continue
        # Directory layout: subs_dir/{DSO}/LIGHT/{frame}.fits
        dso_dir_name = fits_path.parent.parent.name
        try:
            with _fits.open(fits_path) as hdul:
                hdr = hdul[0].header
                if str(hdr.get("FILTER", "")).strip() != filter_name:
                    continue
                if round(float(hdr.get("EXPTIME", 0))) != exptime_s:
                    continue
                data = np.squeeze(hdul[0].data).astype(np.float32)
                if data.ndim != 2:
                    continue
                frames.append(data)
                dso_names.append(dso_dir_name)
        except Exception:
            pass

    return frames, dso_names
