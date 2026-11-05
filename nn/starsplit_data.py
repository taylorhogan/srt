"""Training pairs for the learned star/nebula split (2026-11-05).

The network takes one background-subtracted plane and returns it without
its stars; the star layer is the difference. It replaces the palette's
hand rules (stacking/color_process star_mask + smooth_fill: discs sized by
a peak rule, a block-median fill, painted spike arms) and gives a starless
render as a product in its own right.

Why synthetic injection and not hand-made starless targets: a target made
by the current rules teaches the network the rules' faults (flat fills,
rings at bright stars), and a teacher-student net can only reproduce the
teacher (LEARNED_DEBLUR_PLAN). So the target is a REAL background with
KNOWN stars added:

  background  = the plane with its detected stars replaced by smooth_fill
                (3 sigma discs, the palette's own finder), so what is left
                is the nebula, the sky, the noise and stars below the
                noise; 10 % of patches are a flat sky or a fractal field
                for diversity; synthetic knots (1.3-3x the PSF, Sersic-ish)
                are added as BACKGROUND so a compact HII knot is never
                learned as a star;
  stars       = the field's own measured PSF (stacking/deblur.measure_psf:
                median star stamp to 72 px, wing and spikes in it), peak
                amplitudes log-uniform from 1.5 sigma to 30x saturation
                and clipped at saturation (a plateau the net must fill),
                density drawn around the field's own star density;
  input       = background + stars (+ a little white noise so the noise is
                at the pixel scale after any zoom), target = background
                with the same noise, both asinh-normalised by the target's
                sky sigma (nn/denoiser.normalise, as every model here).

Limitations of this first recipe, recorded so nobody rediscovers them:
the stamp ends at 72 px, so the far arms of a 3-4 dex star are not in
the input and the net will leave what it never saw; the fill under a
detected star is a block median, so the background under an injected
star is smoother than nature where it overlaps a filled disc.
"""
from __future__ import annotations

import json
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
PREPARED = ROOT / "local" / "starsplit" / "prepared"
MODELS = ROOT / "local" / "starsplit" / "models"
PATCH = 512
HOLDOUT = {"trunk", "ngc7380"}          # never trained on; trunk is N2N's holdout too
VAL = {"m76"}                             # epoch selection
FLAT_FRACTION = 0.05
FRACTAL_FRACTION = 0.05
KNOTS_PER_PATCH = (0, 8)
STAR_DENSITY_JITTER = (0.5, 2.0)
AMP_SIGMA_MIN = 1.5
AMP_SAT_MAX = 30.0
BRIGHT_SLOPE = 0.6                       # power-law slope of the amplitude distribution, as deconv_phase0


# ---------------------------------------------------------------------------
# prepare: one plane -> fill (starless-ish background) + psf sidecar
# ---------------------------------------------------------------------------

def prepare_plane(plane: np.ndarray, name: str, out_dir: Path = PREPARED, log=print) -> dict:
    """Write `<name>_bg.npy` (the background-subtracted plane with its
    detected stars filled) and `<name>_psf.npz` (psf, spikes, sat, sky
    sigma, star density), and return the manifest entry."""
    from stacking import color_process, deblur
    out_dir.mkdir(parents=True, exist_ok=True)
    a = np.asarray(plane, dtype=np.float32)
    sub = np.asarray(color_process._remove_gradient(a, mesh=4), dtype=np.float32)
    mask, info = color_process.star_mask(
        sub, color_process.PALETTE_STAR_THRESH_SIGMA, color_process.PALETTE_STAR_MAX_NPIX,
        color_process.PALETTE_STAR_RADIUS_A, color_process.PALETTE_STAR_MAX_RADIUS_PX,
        feather=0.6, min_peak_flux=color_process.PALETTE_STAR_MIN_PEAK_FLUX,
        radius_from="peak", feather_max=color_process.PALETTE_STAR_FEATHER_MAX_PX)
    fill = color_process.smooth_fill(sub, color_process.PALETTE_FILL_BLOCK, mask=mask)
    m = np.clip(mask, 0, 1).astype(np.float32)
    bg = (sub * (1 - m) + fill * m).astype(np.float32)
    psf, pinfo = deblur.measure_psf(a)
    if psf is None:
        raise RuntimeError(f"{name}: no PSF ({pinfo.get('why')})")
    sat = deblur.saturation_level(a)
    rms = float(info.get("rms", np.nanstd(sub[::8, ::8])))
    np.save(out_dir / f"{name}_bg.npy", bg)
    np.savez_compressed(out_dir / f"{name}_psf.npz", psf=psf.astype(np.float32),
                        spikes=np.asarray(pinfo.get("spikes"), dtype=np.float32),
                        sat=float(sat or 65535.0), rms=rms,
                        density=float(info.get("detected", 0)) / float(a.size))
    entry = {"name": name, "shape": list(a.shape), "stars_filled": int(info.get("kept", 0)),
             "detected": int(info.get("detected", 0)), "rms": rms, "sat": float(sat or 65535.0),
             "fwhm_px": float(pinfo.get("fwhm_px", 0.0)), "psf_stars": int(pinfo.get("stars", 0)),
             "wing_beyond_24px": float(pinfo.get("wing_flux_beyond_24px", 0.0))}
    log(f"  {name}: {entry['stars_filled']} stars filled of {entry['detected']}, rms {rms:.2f}, "
        f"fwhm {entry['fwhm_px']:.2f} px from {entry['psf_stars']} stars, sat {entry['sat']:.0f}")
    return entry


def write_manifest(entries: list, out_dir: Path = PREPARED) -> Path:
    p = out_dir / "manifest.json"
    old = json.loads(p.read_text()) if p.exists() else {"planes": {}}
    for e in entries:
        old["planes"][e["name"]] = e
    p.write_text(json.dumps(old, indent=2))
    return p


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def inject_stars(bg_shape, psf: np.ndarray, rng, n: int, rms: float, sat: float):
    """A star field: *n* stars at sub-pixel positions, peak amplitudes from
    AMP_SIGMA_MIN*rms to AMP_SAT_MAX*sat on a power law, each the measured
    PSF shifted by its fractional offset. Returns (stars, table[n,3] y x amp)."""
    from scipy.ndimage import shift as ndshift
    h, w = bg_shape
    R = psf.shape[0] // 2
    out = np.zeros((h + 2 * R, w + 2 * R), dtype=np.float32)
    lo, hi = AMP_SIGMA_MIN * rms, AMP_SAT_MAX * sat
    # power law p(A) ~ A^-(1+slope) between lo and hi, via inverse CDF
    u = rng.random(n)
    s = BRIGHT_SLOPE
    amps = (lo ** -s + u * (hi ** -s - lo ** -s)) ** (-1.0 / s)
    ys = rng.uniform(0, h - 1, n)
    xs = rng.uniform(0, w - 1, n)
    pk = float(psf.max())
    table = np.zeros((n, 3), dtype=np.float64)
    for k, (y, x, A) in enumerate(zip(ys, xs, amps)):
        iy, ix = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - iy, x - ix
        stamp = ndshift(psf, (fy, fx), order=1, mode="constant", cval=0.0) if (fy > 0.02 or fx > 0.02) else psf
        out[iy:iy + 2 * R + 1, ix:ix + 2 * R + 1] += (A / pk) * stamp
        table[k] = (y, x, A)
    return out[R:R + h, R:R + w], table


def synthetic_knots(shape, rng, fwhm_px: float, n: int, rms: float):
    """Compact diffuse knots, 1.3-3x the PSF, Gaussian-ish, as BACKGROUND.
    Each drawn on its own +-4 sigma window, not the whole patch."""
    h, w = shape
    out = np.zeros(shape, dtype=np.float32)
    for _ in range(max(n, 0)):
        y, x = rng.uniform(0, h), rng.uniform(0, w)
        sig = rng.uniform(1.3, 3.0) * fwhm_px / 2.3548
        amp = rng.uniform(2, 40) * rms
        e = rng.uniform(0.6, 1.0)
        th = rng.uniform(0, np.pi)
        r = int(np.ceil(4 * sig)) + 1
        y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, h)
        x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - x, yy - y
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        out[y0:y1, x0:x1] += (amp * np.exp(-0.5 * ((u / sig) ** 2 + (v / (sig * e)) ** 2))).astype(np.float32)
    return out


def fractal_field(shape, rng, rms: float):
    """A power-law random field, nebula-like, at a few to tens of sigma."""
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(fy ** 2 + fx ** 2)
    k[0, 0] = 1.0
    spec = rng.normal(size=(h, fx.shape[1])) + 1j * rng.normal(size=(h, fx.shape[1]))
    f = np.fft.irfft2(spec * k ** -1.6, s=(h, w))
    f = f - f.min()
    f = f / max(f.max(), 1e-9)
    return (f * rng.uniform(5, 60) * rms).astype(np.float32)


class StarSplitDataset:
    """Pairs (input, target), both (1, P, P) float32 in asinh units, and
    the target's scale; `make_pair(rng)` draws one."""

    def __init__(self, names: list, prepared: Path = PREPARED, patch: int = PATCH):
        self.patch = patch
        self.planes = []
        for n in names:
            bg = np.load(prepared / f"{n}_bg.npy", mmap_mode="r")
            z = np.load(prepared / f"{n}_psf.npz")
            man = json.loads((prepared / "manifest.json").read_text())["planes"][n]
            self.planes.append({"name": n, "bg": bg, "psf": z["psf"], "sat": float(z["sat"]),
                                "rms": float(z["rms"]), "density": float(z["density"]),
                                "fwhm": float(man.get("fwhm_px") or 5.0)})
        if not self.planes:
            raise ValueError("no planes")

    def make_pair(self, rng, with_truth: bool = False):
        from scipy.ndimage import zoom

        from nn import denoiser
        P = self.patch
        pl = self.planes[rng.integers(len(self.planes))]
        bg_full, rms = pl["bg"], pl["rms"]
        u = rng.random()
        if u < FLAT_FRACTION:
            bg = np.zeros((P, P), dtype=np.float32)
        elif u < FLAT_FRACTION + FRACTAL_FRACTION:
            bg = fractal_field((P, P), rng, rms)
        else:
            z = rng.uniform(1.0, 1.3)
            n_nat = int(np.ceil(P / z)) + 2
            h, w = bg_full.shape
            y0 = rng.integers(0, h - n_nat)
            x0 = rng.integers(0, w - n_nat)
            nat = np.asarray(bg_full[y0:y0 + n_nat, x0:x0 + n_nat], dtype=np.float32)
            nat = np.nan_to_num(nat, nan=float(np.nanmedian(nat)))
            bg = zoom(nat, z, order=1)[:P, :P] if z > 1.0 else nat[:P, :P]
            bg = bg - np.median(bg)
        bg = bg + synthetic_knots((P, P), rng, pl["fwhm"], int(rng.integers(*KNOTS_PER_PATCH)), rms)
        noise = rng.normal(0.0, 0.5 * rms, (P, P)).astype(np.float32)
        target = bg + noise
        dens = pl["density"] * rng.uniform(*STAR_DENSITY_JITTER)
        n = int(rng.poisson(max(dens * P * P, 1.0)))
        stars, table = inject_stars((P, P), pl["psf"], rng, n, rms, pl["sat"])
        inp = np.minimum(target + stars, pl["sat"])            # saturation plateau
        _, scale = denoiser.normalise(target)
        t_in = np.arcsinh(inp / scale).astype(np.float32)[None]
        t_tg = np.arcsinh(target / scale).astype(np.float32)[None]
        if with_truth:
            return t_in, t_tg, scale, {"stars": stars, "table": table, "target": target, "input": inp,
                                       "rms": rms, "name": pl["name"]}
        return t_in, t_tg, scale
