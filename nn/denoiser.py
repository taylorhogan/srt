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


# Multiple of the robust sky sigma used as the asinh scale. Larger values push
# the bright end toward asinh's linear regime, which reduces how badly sinh
# amplifies network error on inversion (the amplification is cosh(t)) — at the
# cost of shrinking the sky noise the network has to work on. Measured on m92:
#
#   mult   max t   sky noise   amplification at max t
#      1    9.43      0.5007                    6235x
#     10    7.13      0.0540                     624x
#    100    4.83      0.0054                      62x
#
# A single module-level constant because training and inference MUST agree on
# it; every other way of expressing this has drifted at least once already
# (c617047, c61cd26). It is stamped into the checkpoint by trainer.train() and
# checked on load.
ASINH_SIGMA_MULT = 1.0


def subtract_background(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame - smooth sky background, background).

    Shared by training and inference so both see the same kind of image. That
    sharing is the point: inference started subtracting the background in
    c617047 to kill tile seams, which silently undid c61cd26's alignment of the
    two normalisation paths and left the model trained on frames that still had
    the vignetting gradient in them. Measured on real frames the resulting
    scale error runs 1.0-1.7x and varies by field, so the model was applied
    off-distribution by an amount that changed from target to target.

    sep.Background is a mesh of box medians interpolated to full resolution.
    """
    import sep
    background = np.array(sep.Background(frame.astype(np.float64))).astype(np.float32)
    return frame - background, background


def normalise(frame_sub: np.ndarray) -> tuple[np.ndarray, float]:
    """asinh-compress a background-subtracted frame to a trainable range.

    Returns (t, scale); invert with denormalise(). Used by both the training
    dataset and tiled inference — one definition, so the two cannot drift
    apart again.

    This was a linear (x - p1) / (p99 - p1) scaling until 2026-08-12, and the
    docstring claimed it produced ~[0,1]. It did not: 99% of pixels are sky, so
    p99 - p1 spans the sky *noise*, and measured on real frames the output
    reached 837 (m92) and 631 (ngc5907). Feeding that to a conv stack with no
    normalisation layers (BatchNorm was removed in 91b43b9, correctly, because
    the absolute output level carries the photometry) collapsed training onto
    the trivial solution: the trained net emitted a constant, uncorrelated with
    its input (corr -0.02), erasing every star. Its val loss, 0.2230, was the
    constant-predictor baseline of 0.2263 — while a perfect denoiser scores
    0.1276, so it captured ~1.5% of the 43.6% available.

    A linear scale cannot fix that. The dynamic range here is ~4000:1, so any
    single factor either overflows the stars (p99 -> max 837) or underflows the
    sky into nothing (p100 -> sky sigma 0.0025). asinh compresses the bright end
    while staying linear near zero, which is where the sky noise lives:
    max ~9.4 with sky noise at 0.50.

    It is *exactly* invertible (round-trip error 4e-7, float32 precision), so
    the photometry is recoverable — which a stretch applied for looks would not
    be. Note sinh grows exponentially, so network error at the bright end is
    amplified on inversion; that is a measurable failure the step-4 gate exists
    to catch, traded against a guaranteed one.

    Scaling on the robust sky sigma rather than a percentile also makes fields
    comparable: m92 and ngc5907 peak at 9.4 and 8.7 here, against 837 and 631
    before. The old scaling was field-dependent by ~1.3x, the same class of
    drift fix #2 of 91b43b9 repaired.
    """
    # Subsampled: a sky-sigma estimate does not need all 61 Mpx, and the two
    # full-frame medians this replaces cost ~2 s per frame across 204 frames.
    s = frame_sub[::4, ::4]
    sigma = 1.4826 * float(np.median(np.abs(s - float(np.median(s)))))
    scale = max(sigma * ASINH_SIGMA_MULT, 1e-3)
    return np.arcsinh(frame_sub / scale).astype(np.float32), scale


def denormalise(t: np.ndarray, scale: float) -> np.ndarray:
    """Invert normalise(). Kept beside it so the pair cannot drift apart.

    The inverse used to be open-coded at the end of denoise_frame(), which is
    exactly the shape of bug that let training and inference diverge twice
    before (c617047, c61cd26).
    """
    return (scale * np.sinh(t.astype(np.float64))).astype(np.float32)


def denoise_frame(
    frame: np.ndarray,
    model: nn.Module,
    device: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Denoise a single 2-D float32 FITS frame using tiled inference.

    Background (vignetting gradient) is estimated with sep and subtracted
    before tiling so every tile sees a flat sky — this eliminates the tile-
    boundary seam artefacts caused by per-tile DC-level disagreement.  The
    background is added back after inference.

    Edge tiles are zero-padded to tile_size and the padding is discarded after
    inference.  Tiles are blended with a raised-cosine weight for smooth
    transitions.  Input and output are in the original ADU scale.
    """
    if device is None:
        device = best_device()
    model = model.to(device)

    # Flatten the sky so every tile sees the same DC level (this is what killed
    # the seam artefacts) and normalise once so all tiles share a scale. Both
    # steps are the shared helpers above, which is what keeps training and
    # inference on the same footing.
    frame_sub, background = subtract_background(frame)
    norm_frame, scale = normalise(frame_sub)

    h, w = frame.shape
    step = tile_size - overlap
    output = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    # Tukey (cosine-tapered) window: flat 1.0 in the centre, cosine roll-off
    # over exactly `overlap` pixels at each end.  With step = tile_size - overlap
    # this puts the 50/50 cross-over point at the seam midpoint, giving seamless
    # blending regardless of per-tile DC differences.
    def _window1d(n: int) -> np.ndarray:
        win = np.ones(n, dtype=np.float32)
        for i in range(overlap):
            w = 0.5 - 0.5 * np.cos(np.pi * i / overlap)
            win[i] = w
            win[n - overlap + i] = 1.0 - w
        return win

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

    # Denormalise and restore the background
    return denormalise(norm_output, scale) + background


def collect_all_frames(
    subs_dir: Path,
    filter_name: str,
    exptime_s: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Walk all DSO subdirs under subs_dir and collect matching LIGHT frames.

    Only frames whose FITS FILTER header matches filter_name AND whose EXPTIME
    rounds to exptime_s are returned.

    Returns (frames, dso_names) where dso_names[i] is the normalized DSO key
    for frames[i]. This is used by N2NDataset to restrict pairs to within the
    same DSO — N2N requires pairs to be two noisy views of the same scene.
    """
    from astropy.io import fits as _fits

    frames: list[np.ndarray] = []
    dso_names: list[str] = []

    for fits_path in sorted(subs_dir.rglob("*.fits")):
        if fits_path.parent.name.upper() != "LIGHT":
            continue
        # Layout varies in depth: subs_dir/{DSO}/[rig/]{date}/LIGHT/{frame}.fits.
        # The DSO is always the first path component under subs_dir — NOT
        # parent.parent, which lands on the date when a rig/date level exists.
        # Normalize (lowercase, no spaces) so "M 13" and "m13" group as one
        # scene, matching the canonical DSO key used elsewhere in the codebase.
        try:
            dso_dir_name = fits_path.relative_to(subs_dir).parts[0].lower().replace(" ", "")
        except ValueError:
            dso_dir_name = fits_path.parent.parent.name.lower().replace(" ", "")
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
