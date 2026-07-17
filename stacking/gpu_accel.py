"""Optional GPU (torch CUDA) acceleration for the registration hot spots.

Registration profile on the DGX Spark (61 MP QHY600M frames, m92):
despike (scipy 3x3 median) 4.1 s + find_transform 2.6 s + apply_transform
(skimage warp) 1.6 s per frame. The two image-wide raster ops — the median
filter and the affine warp — are the GPU wins; transform *estimation*
(sep detection + astroalign triangle matching) stays on CPU where it is
cheap and already validated.

Every helper returns None when torch/CUDA is unavailable so callers fall
back to the CPU path — the Windows observatory box runs this same code
without torch installed.
"""

import logging

import numpy as np

_logger = logging.getLogger(__name__)

# None = not yet probed, False = unavailable, module = ready
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            if torch.cuda.is_available():
                _TORCH = torch
                _logger.info("GPU acceleration active: %s",
                             torch.cuda.get_device_name(0))
            else:
                _TORCH = False
        except Exception:
            _TORCH = False
    return _TORCH


def available() -> bool:
    return bool(_torch())


def median3(frame: np.ndarray):
    """3x3 median filter on the GPU, or None when CUDA is unavailable.

    Matches scipy.ndimage.median_filter(frame, size=3) exactly in the
    interior; the 1-px border differs (replicate vs reflect padding), which
    is irrelevant for its only use — star detection for registration.
    """
    torch = _torch()
    if not torch:
        return None
    try:
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            t = t.cuda().unsqueeze(0).unsqueeze(0)
            p = torch.nn.functional.pad(t, (1, 1, 1, 1), mode="replicate")
            u = p.unfold(2, 3, 1).unfold(3, 3, 1)
            m = u.reshape(*u.shape[:4], 9).median(dim=-1).values
            return m.squeeze(0).squeeze(0).cpu().numpy().astype(
                frame.dtype, copy=False)
    except Exception:
        _logger.exception("GPU median3 failed — falling back to CPU")
        return None


def apply_affine(matrix: np.ndarray, source: np.ndarray, out_shape):
    """Warp ``source`` into the reference grid via a 3x3 affine, on the GPU.

    ``matrix`` maps source pixel coords -> target pixel coords (astroalign's
    transform.params, x-then-y row convention). Returns (aligned float32,
    footprint bool) matching astroalign.apply_transform's contract —
    footprint True where the output pixel falls outside the source frame —
    or None when CUDA is unavailable. Bicubic sampling, like skimage's
    default order-3 warp (kernels differ slightly; validated to sub-0.1 %
    aperture-flux agreement).
    """
    torch = _torch()
    if not torch:
        return None
    try:
        h_out, w_out = int(out_shape[0]), int(out_shape[1])
        h_in, w_in = source.shape
        inv = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
        with torch.no_grad():
            ys, xs = torch.meshgrid(
                torch.arange(h_out, device="cuda", dtype=torch.float32),
                torch.arange(w_out, device="cuda", dtype=torch.float32),
                indexing="ij")
            m = torch.from_numpy(inv.astype(np.float32)).cuda()
            sx = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
            sy = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
            gx = 2.0 * sx / (w_in - 1) - 1.0
            gy = 2.0 * sy / (h_in - 1) - 1.0
            grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
            src = torch.from_numpy(
                np.ascontiguousarray(source, dtype=np.float32))
            src = src.cuda().unsqueeze(0).unsqueeze(0)
            out = torch.nn.functional.grid_sample(
                src, grid, mode="bicubic", padding_mode="zeros",
                align_corners=True)
            oob = (gx < -1) | (gx > 1) | (gy < -1) | (gy > 1)
            aligned = out.squeeze(0).squeeze(0).cpu().numpy()
            footprint = oob.cpu().numpy()
        return aligned.astype(np.float32, copy=False), footprint
    except Exception:
        _logger.exception("GPU apply_affine failed — falling back to CPU")
        return None
