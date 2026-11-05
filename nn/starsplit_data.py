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
# v2 (2026-11-05, after the epoch-13 look at IC 410: bright stars left a
# nub inside a dark ring, a faint far spike survived): the star stamp is
# shifted with a cubic spline, not bilinearly, so the injected core is as
# sharp as a real star's; BRIGHT_FRACTION of the stars are drawn
# log-uniform from 100 sigma to the cap, since the power law makes a
# bright star a 1-in-500 event; and a star past ARM_FROM_SIGMA gets four
# arms beyond the stamp along the measured spike axes, amplitude falling
# as 1/r from the stamp's edge out to the palette's length rule.
SHIFT_ORDER = 3
BRIGHT_FRACTION = 0.25
BRIGHT_FLOOR_SIGMA = 100.0
ARM_FROM_SIGMA = 1500.0                   # below this the wing past 62 px and the arms past 100 px are under 0.05 sigma
ARM_L0, ARM_L_PER_DEX2, ARM_MAX = 40.0, 100.0, 400.0


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
    axes = pinfo.get("spike_axes") or []
    np.savez_compressed(out_dir / f"{name}_psf.npz", psf=psf.astype(np.float32),
                        spikes=np.asarray(pinfo.get("spikes"), dtype=np.float32),
                        sat=float(sat or 65535.0), rms=rms,
                        density=float(info.get("detected", 0)) / float(a.size),
                        spike_axes=np.asarray(axes, dtype=np.float64),
                        arm_sigma=float(pinfo.get("arm_sigma") or 0.0))
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

def _amplitudes(rng, n: int, lo: float, hi: float):
    """Peak amplitudes: a power law from lo to hi (slope BRIGHT_SLOPE) for
    most stars, log-uniform from BRIGHT_FLOOR_SIGMA*lo/AMP_SIGMA_MIN to hi
    for BRIGHT_FRACTION of them."""
    s = BRIGHT_SLOPE
    u = rng.random(n)
    amps = (lo ** -s + u * (hi ** -s - lo ** -s)) ** (-1.0 / s)
    bright = rng.random(n) < BRIGHT_FRACTION
    b_lo = min(lo * BRIGHT_FLOOR_SIGMA / AMP_SIGMA_MIN, hi / 2)
    amps[bright] = np.exp(rng.uniform(np.log(b_lo), np.log(hi), bright.sum()))
    return amps


def _wing_extension(psf: np.ndarray, r_max: float):
    """A power-law continuation of the psf's azimuthal wing beyond its stamp:
    the slope is fitted in log-log between 40 px and the taper's start and
    clipped to a halo's range; the LEVEL is the measured ring median just
    inside the taper, so the continuation starts where the stamp leaves
    off. Returns (slope, level at r0, r0) or None."""
    R = psf.shape[0] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    r0 = R - 10
    rs, vs = [], []
    for r in range(40, r0, 4):
        ring = (rr >= r - 2) & (rr < r + 2)
        v = float(np.median(psf[ring]))
        if v > 0:
            rs.append(r); vs.append(v)
    anchor = (rr >= r0 - 4) & (rr < r0)
    v0 = float(np.median(psf[anchor])) if anchor.any() else 0.0
    if len(rs) < 3 or v0 <= 0:
        return None
    slope = float(np.clip(np.polyfit(np.log(rs), np.log(vs), 1)[0], -3.5, -1.5))
    return slope, v0, r0


def _far_arms(out: np.ndarray, y: float, x: float, amp: float, psf: np.ndarray,
              spikes: Optional[np.ndarray], axes, rms: float, arm_sigma: float = 3.0) -> None:
    """What a bright star has beyond the PSF stamp, in place on *out* (which
    carries a psf-radius border): the azimuthal wing continued as a power
    law (the stamp ends in a 10 px cosine taper at 72 px; a real halo does
    not), measure_psf's spike template (201 px) blended in where the stamp
    tapers out, and past the template a Gaussian ribbon of the template's
    own width per half-axis, its peak continuing the template's edge level
    as 1/r out to the palette's length rule. A four-vane spider's axes are
    90 deg apart, so the second axis is the first folded, as the palette's
    spike_axis assumes (measure_psf's second peak was 50 deg off on IC 1396)."""
    R = psf.shape[0] // 2
    pk = float(psf.max())
    h, w = out.shape
    cy, cx = int(np.floor(y)) + R, int(np.floor(x)) + R
    dex = np.log10(max(amp / max(rms, 1e-6), 1.0))
    length = min(ARM_L0 + ARM_L_PER_DEX2 * max(dex - 2.5, 0.0) ** 2, ARM_MAX)
    # 1. the wing beyond the stamp
    ext = _wing_extension(psf, length)
    if ext is not None:
        slope, v0, r0 = ext
        Re = int(min(max(length, R + 40), 400))
        yy, xx = np.mgrid[-Re:Re + 1, -Re:Re + 1].astype(np.float32)
        rr = np.hypot(yy, xx)
        r1 = np.arange(0, Re + 2, dtype=np.float64)
        p1 = np.where(r1 > r0, v0 * (np.maximum(r1, 1.0) / r0) ** slope, 0.0)
        tt = np.clip((r1 - r0) / 10.0, 0.0, 1.0)
        p1 *= 1.0 - 0.5 * (1.0 + np.cos(np.pi * tt))           # the complement of the stamp's cosine taper
        p1 *= np.clip((Re - r1) / 20.0, 0.0, 1.0)               # and fades at its own edge
        halo = (amp / pk) * np.interp(rr, r1, p1).astype(np.float32)
        if halo.max() > 0.02 * rms:
            y0, x0 = cy - Re, cx - Re
            ys0, xs0 = max(0, -y0), max(0, -x0)
            ys1, xs1 = min(2 * Re + 1, h - y0), min(2 * Re + 1, w - x0)
            if ys1 > ys0 and xs1 > xs0:
                out[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1] += halo[ys0:ys1, xs0:xs1].astype(np.float32)
    # 2. the arms: ribbons from where the stamp's taper starts, along the
    # first measured axis and that axis + 90 deg (IC 410's and Melotte 15's
    # brightest stars both show arms at 65 and 155 deg: a four-vane spider;
    # measure_psf's second peak read 119 deg on IC 1396, so it is not used),
    # at the arm level the spike template shows along the first axis at
    # 48-58 px, continued as 1/r. The template's pixels themselves are not
    # blended in: they carry that wrong second pair.
    if axes is None or len(axes) == 0:
        return
    th0 = float(axes[0])
    axes2 = (th0, th0 + np.pi / 2)
    src = spikes if spikes is not None and spikes.shape[0] > psf.shape[0] else psf
    Rt = src.shape[0] // 2
    yy, xx = np.mgrid[-Rt:Rt + 1, -Rt:Rt + 1]
    rr = np.hypot(yy, xx)
    dang = np.abs((np.arctan2(yy, xx) - th0 + np.pi / 2) % np.pi - np.pi / 2)
    band = (rr > 48) & (rr < 58)
    on_axis = band & (dang < np.radians(8))
    off_axis = band & (dang > np.radians(30)) & (dang < np.radians(60))
    if not on_axis.any() or not off_axis.any():
        return
    excess = float(np.max(src[on_axis]) - np.median(src[off_axis]))
    r_edge = R - 10
    edge = excess * (amp / pk) * (53.0 / r_edge)
    if edge <= 0.05 * rms or length <= r_edge:
        return
    sig = float(arm_sigma) if arm_sigma and arm_sigma > 0.5 else 3.0
    L = int(np.ceil(length)) + 4
    yy, xx = np.mgrid[-L:L + 1, -L:L + 1].astype(np.float32)
    fy, fx = (y - np.floor(y)), (x - np.floor(x))
    yy -= fy; xx -= fx
    ribbon = np.zeros_like(yy)
    for th in axes2:
        s_ = yy * np.sin(th) + xx * np.cos(th)          # along the axis, signed
        d_ = -yy * np.cos(th) + xx * np.sin(th)         # across it
        sa = np.abs(s_)
        on = (sa >= r_edge) & (sa <= length)
        tt = np.clip((sa - r_edge) / 10.0, 0.0, 1.0)
        ramp = 1.0 - 0.5 * (1.0 + np.cos(np.pi * tt))        # in as the stamp's cosine taper goes out
        with np.errstate(divide="ignore", invalid="ignore"):
            ribbon += np.where(on, ramp * edge * (r_edge / np.maximum(sa, 1.0)) * np.exp(-0.5 * (d_ / sig) ** 2), 0.0)
    y0, x0 = cy - L, cx - L
    ys0, xs0 = max(0, -y0), max(0, -x0)
    ys1, xs1 = min(2 * L + 1, h - y0), min(2 * L + 1, w - x0)
    if ys1 > ys0 and xs1 > xs0:
        out[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1] += ribbon[ys0:ys1, xs0:xs1]


def inject_stars(bg_shape, psf: np.ndarray, rng, n: int, rms: float, sat: float,
                 spikes: Optional[np.ndarray] = None, axes=None, arm_sigma: float = 3.0):
    """A star field: *n* stars at sub-pixel positions, peak amplitudes from
    AMP_SIGMA_MIN*rms to AMP_SAT_MAX*sat (see _amplitudes), each the measured
    PSF spline-shifted by its fractional offset, with far arms past the stamp
    for the bright ones. Returns (stars, table[n,3] y x amp)."""
    from scipy.ndimage import shift as ndshift
    h, w = bg_shape
    R = psf.shape[0] // 2
    out = np.zeros((h + 2 * R, w + 2 * R), dtype=np.float32)
    amps = _amplitudes(rng, n, AMP_SIGMA_MIN * rms, AMP_SAT_MAX * sat)
    ys = rng.uniform(0, h - 1, n)
    xs = rng.uniform(0, w - 1, n)
    pk = float(psf.max())
    table = np.zeros((n, 3), dtype=np.float64)
    for k, (y, x, A) in enumerate(zip(ys, xs, amps)):
        iy, ix = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - iy, x - ix
        stamp = (np.clip(ndshift(psf, (fy, fx), order=SHIFT_ORDER, mode="constant", cval=0.0), 0, None)
                 if (fy > 0.02 or fx > 0.02) else psf)
        out[iy:iy + 2 * R + 1, ix:ix + 2 * R + 1] += (A / pk) * stamp
        table[k] = (y, x, A)
        if A > ARM_FROM_SIGMA * rms:
            _far_arms(out, y, x, A, psf, spikes, axes, rms, arm_sigma)
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
                                "fwhm": float(man.get("fwhm_px") or 5.0),
                                "spikes": z["spikes"] if "spikes" in z.files else None,
                                "axes": (z["spike_axes"].tolist() if "spike_axes" in z.files else []),
                                "arm_sigma": float(z["arm_sigma"]) if "arm_sigma" in z.files else 3.0})
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
        stars, table = inject_stars((P, P), pl["psf"], rng, n, rms, pl["sat"],
                                    spikes=pl["spikes"], axes=pl["axes"], arm_sigma=pl["arm_sigma"])
        inp = np.minimum(target + stars, pl["sat"])            # saturation plateau
        _, scale = denoiser.normalise(target)
        t_in = np.arcsinh(inp / scale).astype(np.float32)[None]
        t_tg = np.arcsinh(target / scale).astype(np.float32)[None]
        if with_truth:
            return t_in, t_tg, scale, {"stars": stars, "table": table, "target": target, "input": inp,
                                       "rms": rms, "name": pl["name"]}
        return t_in, t_tg, scale
