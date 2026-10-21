"""Training data for the learned deconvolver: real stacks, re-blurred on the fly.

docs/LEARNED_DEBLUR_PLAN.md, Phase 1. The training signal is manufactured
from the stacks themselves: a background-subtracted stack X carries one
kernel K of blur, so Y = K*X + n' is "one kernel blurrier", and a network
trained to recover X from Y removes one kernel; applied to X it estimates
the sharp scene. Phase 0 measured that this composes on the real kernel for
extended structure, and that only a flux-conserving head (nn/deblur_model)
gets the stars right.

Two files per prepared stack, written once by `prepare_stack`:

    <name>_sub.npy     background-subtracted stack, float32, memory-mapped
                       at training time (a full frame is 245 MB; the pool
                       is ~30 GB and never loaded whole)
    <name>.psf.npz     the sidecar: the kernel and spikes (deblur.measure_psf),
                       its FWHM, the saturation level, the sky sigma, the
                       plateau mask (deblur.flat_tops, packed) and the
                       coverage-fill mask (exact-median pixels, packed)

The sidecar is measured ONCE so every arm trains on the same kernels, and on
the stack it belongs to: a denoised stack's kernel is not the raw stack's
(the denoiser erodes faint stars' wings).

`DeblurDataset` draws source-biased 512 px patches. Per patch, with the
stack's own kernel scaled by a random factor in `kernel_scale`:

    a      = patch of X, sky-subtracted, linear ADU, read with a margin so the
             two convolutions below see real context, not a padded edge
    Y      = K'*a + n'          n' white, restoring the fine-band noise the
                                blur removed (var(K*n) = sigma^2 * sum(K^2))
    input  = asinh([Y, K'*Y] / scale)      the second plane tells the network
                                            which kernel to remove (plan,
                                            design choice 1)
    target = asinh(a / scale)
    mask   = not within SAT_DILATE px of a plateau, not coverage fill, finite

`scale` is denoiser.normalise's sky-sigma scale for the whole stack, the same
rule inference uses, so the two cannot drift apart (the denoiser paid for
that lesson twice).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from nn import denoiser  # noqa: E402

PREPARED = Path(_root) / "local" / "deconv" / "prepared"
PATCH = 512
SAT_DILATE = 6              # px round a plateau that the loss ignores
KERNEL_SCALE = (0.8, 1.25)  # the training kernel is the measured one times U(lo, hi)
MARGIN = 160                # px read round a patch: two kernel radii (72) and change
BLOCK = 64                  # px: the source-bias map's block
MIN_STARS = 15              # fewer isolated stars and the stack has no kernel


# ---------------------------------------------------------------- sidecar ----
def measure_sidecar(stack: np.ndarray, log: Callable[[str], None] = print,
                    extra_flat: Optional[np.ndarray] = None) -> Optional[dict]:
    """Kernel, spikes, saturation, sky sigma, plateau and fill masks of *stack*
    (with its sky in). None when the field gives no kernel.

    *extra_flat* is OR-ed into the plateau mask: the raw half's plateaus, the
    same pixels, passed in when a denoised stack is prepared, because the raw
    half is where a ceiling is certain to be seen (the denoiser may round a
    plateau's edge). A precaution; m13's denoised half found its own.
    """
    from stacking import deblur as db
    x = np.asarray(stack, dtype=np.float32)
    fill = x == np.median(x)                      # the stacker's coverage fill
    med = float(np.nanmedian(x))
    work = np.where(np.isfinite(x), x, med).astype(np.float64)
    psf, info = db.measure_psf(work)
    if psf is None or info.get("stars", 0) < MIN_STARS:
        log(f"  no kernel: {info.get('why', 'too few stars')} ({info.get('stars', 0)} stars)")
        return None
    sat = db.saturation_level(work)
    sig = db.sky_sigma(work)
    flat = db.saturated_mask(work, sat, sig) if sat else db.flat_tops(work, sig)
    if extra_flat is not None and extra_flat.shape == flat.shape:
        flat = flat | extra_flat
    return {"psf": psf.astype(np.float32), "spikes": info["spikes"].astype(np.float32),
            "fwhm": float(info["fwhm_px"]), "sat": float(sat or 0.0), "sky_sigma": float(sig),
            "stars": int(info["stars"]), "wing_stars": int(info.get("wing_stars", 0)),
            "flat": flat.astype(bool), "fill": fill.astype(bool), "shape": x.shape}


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.ravel())


def _unpack(packed: np.ndarray, shape) -> np.ndarray:
    n = int(shape[0]) * int(shape[1])
    return np.unpackbits(packed)[:n].reshape(int(shape[0]), int(shape[1])).astype(bool)


def prepare_stack(src: np.ndarray, name: str, out_dir: Path = PREPARED,
                  denoise_model=None, device: Optional[str] = None,
                  log: Callable[[str], None] = print,
                  extra_flat: Optional[np.ndarray] = None) -> Optional[dict]:
    """Write <name>_sub.npy and <name>.psf.npz from a stack with its sky in.

    *denoise_model* (a loaded N2N model) turns the raw stack into the
    denoised domain first; the sidecar is then measured on the denoised
    stack, which is what the model will be applied to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(src, dtype=np.float32)
    if denoise_model is not None:
        h, w = x.shape
        x = denoiser.denoise_frame(x, denoise_model, device=device)[:h, :w]
    side = measure_sidecar(x, log=log, extra_flat=extra_flat)
    if side is None:
        return None
    sub, _bg = denoiser.subtract_background(np.where(np.isfinite(x), x, np.nanmedian(x)))
    np.save(out_dir / f"{name}_sub.npy", sub.astype(np.float32))
    np.savez(out_dir / f"{name}.psf.npz",
             psf=side["psf"], spikes=side["spikes"], fwhm=side["fwhm"], sat=side["sat"],
             sky_sigma=side["sky_sigma"], stars=side["stars"], wing_stars=side["wing_stars"],
             flat=_pack(side["flat"]), fill=_pack(side["fill"]), shape=np.array(side["shape"]))
    log(f"  {name}: FWHM {side['fwhm']:.2f} px, {side['stars']} stars, sigma {side['sky_sigma']:.2f}, "
        f"sat {side['sat']:.0f}, plateau {side['flat'].sum()} px, fill {100 * side['fill'].mean():.1f}%")
    return side


def load_sidecar(path: Path) -> dict:
    z = np.load(path)
    shape = tuple(int(v) for v in z["shape"])
    return {"psf": np.asarray(z["psf"], dtype=np.float64), "spikes": np.asarray(z["spikes"], dtype=np.float64),
            "fwhm": float(z["fwhm"]), "sat": float(z["sat"]), "sky_sigma": float(z["sky_sigma"]),
            "stars": int(z["stars"]), "flat": _unpack(z["flat"], shape), "fill": _unpack(z["fill"], shape),
            "shape": shape}


# ---------------------------------------------------------------- kernels ----
def scale_kernel(psf: np.ndarray, factor: float) -> np.ndarray:
    """The kernel resampled by *factor* (>1 broader), same stamp size, unit sum."""
    from scipy.ndimage import zoom
    if abs(factor - 1.0) < 1e-3:
        return psf / psf.sum()
    n = psf.shape[0]
    z = zoom(psf, factor, order=1)
    m = z.shape[0]
    out = np.zeros_like(psf)
    if m >= n:
        o = (m - n) // 2
        out[:] = z[o:o + n, o:o + n]
    else:
        o = (n - m) // 2
        out[o:o + m, o:o + m] = z
    out = np.clip(out, 0, None)
    return out / out.sum()


def conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """'same' convolution, reflect-padded so the edge does not wrap."""
    from scipy.signal import fftconvolve
    p = k.shape[0] // 2
    xp = np.pad(x, p, mode="reflect")
    return fftconvolve(xp, k, mode="same")[p:-p, p:-p]


# ---------------------------------------------------------------- dataset ----
@dataclass
class Prepared:
    """One prepared stack: paths, sidecar, and the lazily opened frame."""
    name: str
    dso: str
    filt: str
    domain: str                      # "raw" or "den"
    sub_path: Path
    side_path: Path
    holdout: bool = False
    side: dict = field(default_factory=dict)
    _frame: Optional[np.ndarray] = None
    scale: float = 0.0
    _cdf: Optional[np.ndarray] = None

    @property
    def frame(self) -> np.ndarray:
        if self._frame is None:
            self._frame = np.load(self.sub_path, mmap_mode="r")
        return self._frame

    def open(self) -> "Prepared":
        if not self.side:
            self.side = load_sidecar(self.side_path)
        f = self.frame
        if not self.scale:
            # the same rule as inference (denoiser.normalise reads [::4, ::4] of
            # what it is given); a 4x subsample of the frame keeps this cheap
            _, self.scale = denoiser.normalise(np.asarray(f[::4, ::4], dtype=np.float32))
        return self

    def source_cdf(self, patch: int) -> Optional[np.ndarray]:
        """CDF over patch origins on a BLOCK grid, weighted by significant
        pixels (the N2N dataset's rule: a count, not a max, because the max of
        a block of pure sky already sits at 3 sigma)."""
        if self._cdf is None:
            f = self.frame
            h, w = f.shape
            if h <= patch + 2 * MARGIN or w <= patch + 2 * MARGIN:
                return None
            d = BLOCK // 4
            s = np.asarray(f[::4, ::4], dtype=np.float32)
            sig = self.side["sky_sigma"]
            H, W = s.shape[0] // d, s.shape[1] // d
            m = (s[:H * d, :W * d] > 5 * sig).reshape(H, d, W, d).sum(axis=(1, 3)).astype(np.float64)
            pb = patch // BLOCK
            lo = MARGIN // BLOCK + 1
            ii = np.zeros((H + 1, W + 1)); ii[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)
            oh, ow = H - pb - 2 * lo, W - pb - 2 * lo
            if oh < 1 or ow < 1:
                return None
            box = (ii[lo + pb:lo + pb + oh, lo + pb:lo + pb + ow] - ii[lo:lo + oh, lo + pb:lo + pb + ow]
                   - ii[lo + pb:lo + pb + oh, lo:lo + ow] + ii[lo:lo + oh, lo:lo + ow])
            w_ = box.ravel() + 0.05 * box.mean() + 1e-9      # a floor: blank sky is a scene too
            self._cdf = np.cumsum(w_ / w_.sum())
            self._cdf_shape = (oh, ow, lo)
        return self._cdf


def list_prepared(out_dir: Path = PREPARED, manifest: Optional[Path] = None) -> list[Prepared]:
    """Every prepared stack, with holdout flags from the manifest beside them."""
    man = {}
    mp = manifest or (out_dir / "manifest.json")
    if mp.exists():
        man = json.load(open(mp))
    items = []
    for side in sorted(out_dir.glob("*.psf.npz")):
        name = side.name[: -len(".psf.npz")]
        sub = out_dir / f"{name}_sub.npy"
        if not sub.exists():
            continue
        meta = man.get("stacks", {}).get(name, {})
        dso, filt, domain = meta.get("dso", name.split("__")[0]), meta.get("filter", ""), meta.get("domain", "")
        if not filt or not domain:
            # name convention: <dso>__<filter>_<half>_<domain>
            try:
                rest = name.split("__", 1)[1]
                filt, _half, domain = rest.rsplit("_", 2)
            except ValueError:
                pass
        items.append(Prepared(name=name, dso=dso, filt=filt, domain=domain, sub_path=sub,
                              side_path=side, holdout=bool(meta.get("holdout", False))))
    return items


class DeblurDataset:
    """torch Dataset of (input planes, target, mask) from prepared stacks."""

    def __init__(self, items: list[Prepared], patch_size: int = PATCH, pairs_per_epoch: int = 2000,
                 kernel_scale: tuple = KERNEL_SCALE, source_bias: float = 0.7,
                 seed: Optional[int] = None):
        import torch
        from torch.utils.data import Dataset
        self._Dataset = Dataset
        self.items = [it.open() for it in items]
        if not self.items:
            raise ValueError("no prepared stacks")
        self.patch = int(patch_size)
        self.pairs = int(pairs_per_epoch)
        self.kernel_scale = kernel_scale
        self.source_bias = float(np.clip(source_bias, 0, 1))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._pid = None

    # -- the torch Dataset protocol, without subclassing at import time so the
    #    module imports without torch (tests, the observatory PC)
    def __len__(self) -> int:
        return self.pairs

    def _worker_rng(self) -> None:
        """One generator per loader worker, keyed on the worker's index.

        Keyed on the OS pid (as it was until 2026-10-20) the patch stream
        differed on every launch at the same seed, so a seeded run was not
        recoverable. Worker ids run 0..n-1 and are the same on every launch;
        the main process (no worker info) keeps the seed itself.
        """
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            base = self.seed if self.seed is not None else int.from_bytes(os.urandom(4), "little")
            wid = 0
            try:
                import torch.utils.data as tud
                info = tud.get_worker_info()
                wid = 0 if info is None else int(info.id) + 1
            except ImportError:
                pass
            self.rng = np.random.default_rng([base, wid])

    def _origin(self, it: Prepared) -> tuple[int, int]:
        h, w = it.frame.shape
        P = self.patch
        cdf = it.source_cdf(P)
        if cdf is not None and self.rng.random() < self.source_bias:
            oh, ow, lo = it._cdf_shape
            k = min(int(np.searchsorted(cdf, self.rng.random())), cdf.size - 1)
            by, bx = divmod(k, ow)
            y0 = (by + lo) * BLOCK + int(self.rng.integers(0, BLOCK))
            x0 = (bx + lo) * BLOCK + int(self.rng.integers(0, BLOCK))
        else:
            y0 = int(self.rng.integers(MARGIN, h - P - MARGIN))
            x0 = int(self.rng.integers(MARGIN, w - P - MARGIN))
        y0 = int(np.clip(y0, MARGIN, h - P - MARGIN))
        x0 = int(np.clip(x0, MARGIN, w - P - MARGIN))
        return y0, x0

    def make_pair(self, it: Prepared, y0: int, x0: int, factor: Optional[float] = None):
        """(input (2,P,P), target (1,P,P), mask (1,P,P)) as float32 arrays."""
        P, M = self.patch, MARGIN
        a = np.asarray(it.frame[y0 - M:y0 + P + M, x0 - M:x0 + P + M], dtype=np.float64)
        a = np.where(np.isfinite(a), a, 0.0)
        side = it.side
        if factor is None:
            factor = float(self.rng.uniform(*self.kernel_scale))
        k = scale_kernel(side["psf"], factor)
        sig = side["sky_sigma"]
        add = sig * np.sqrt(max(1.0 - float((k ** 2).sum()), 0.0))
        Y = conv(a, k) + self.rng.normal(0.0, add, a.shape)
        KY = conv(Y, k)
        s = it.scale
        inp = np.stack([np.arcsinh(Y[M:M + P, M:M + P] / s),
                        np.arcsinh(KY[M:M + P, M:M + P] / s)]).astype(np.float32)
        tgt = np.arcsinh(a[M:M + P, M:M + P] / s).astype(np.float32)[None]
        from scipy.ndimage import binary_dilation
        flat = side["flat"][y0:y0 + P, x0:x0 + P]
        fill = side["fill"][y0:y0 + P, x0:x0 + P]
        bad = fill | ~np.isfinite(it.frame[y0:y0 + P, x0:x0 + P])
        if flat.any():
            bad |= binary_dilation(flat, iterations=SAT_DILATE)
        mask = (~bad).astype(np.float32)[None]
        return inp, tgt, mask

    def __getitem__(self, _idx: int):
        import torch
        self._worker_rng()
        for _try in range(8):
            it = self.items[int(self.rng.integers(len(self.items)))]
            y0, x0 = self._origin(it)
            fill = it.side["fill"][y0:y0 + self.patch, x0:x0 + self.patch]
            if fill.mean() < 0.05:
                break
        inp, tgt, mask = self.make_pair(it, y0, x0)
        return torch.from_numpy(inp), torch.from_numpy(tgt), torch.from_numpy(mask)

    def torch_dataset(self):
        """A real torch Dataset view, for DataLoader."""
        parent = self

        class _DS(self._Dataset):
            def __len__(self_):
                return len(parent)

            def __getitem__(self_, i):
                return parent[i]

        return _DS()
