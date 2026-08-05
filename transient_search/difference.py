"""Transient / supernova discovery via image differencing.

Given a DSO name and a filter, this:

1. finds every saved LIGHT sub for that target+filter (reusing the transit
   pipeline's frame discovery),
2. splits them by observing night,
3. builds a deep **template** from all nights except the last and treats the
   **last (most recent) night** as the **science** image,
4. registers template+science onto a common grid, PSF-matches, and subtracts,
5. detects positive residuals (new / brightened sources), rejects artifacts
   (edge/coverage, hot pixels, elongated blobs, subtraction dipoles), and
   down-ranks anything that matches a catalogued Gaia point source (a variable
   star, not a supernova),
6. reports the ranked candidates as a template|science|difference triptych and
   persists them to local/transients.json.

This is pure read-only post-processing of saved FITS — it moves no hardware.

Reuses, without modification:
  - stacking/stacker.py           registration + FITS load  (via _register_only)
  - transit_search/transit.py     _find_dso_dir, _obs_time_mjd, _register_only,
                                   _plate_solve_wcs, _identify_candidates
  - photometry/cmd_diagram.py     the sep detection pattern (re-implemented here)

PSF matching uses a dependency-light scipy Gaussian match by default; if the
optional ``ois`` package (Alard–Lupton optimal image subtraction) is installed
it is used instead for a cleaner subtraction. ``ois`` is intentionally NOT a
hard requirement so a failed build can't break the deploy.
"""

import json
import logging
import math
import os
import sys
import tempfile
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config as _config
from utils.cancellation import Cancelled

_logger = logging.getLogger(__name__)

_DARK_BG = "#0d0d1a"
_DARK_AXES = "#1a1a2e"


class TransientCancelled(Cancelled):
    """Raised inside run_transient_search when the job's cancel flag is set."""


# Serialises the read-modify-write in save_transients so concurrent searches
# can't lose each other's entry (mirrors transit_search.save_transits).
_save_lock = threading.Lock()


def _transient_cfg() -> dict:
    return _config.data().get("transient", {})


def _transient_path() -> Path:
    rel = _transient_cfg().get("file", "local/transients.json")
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return root / rel


def load_transients() -> dict:
    path = _transient_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        _logger.exception("Failed to load transients.json")
        return {}


def save_transients(dso_name: str, filter_name: str, entry: dict) -> None:
    """Merge entry under data[dso][filter] and atomically rewrite the file."""
    path = _transient_path()
    dso_key = dso_name.lower().replace(" ", "")
    with _save_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = load_transients()
        existing = data.get(dso_key, {})
        existing[filter_name] = entry
        data[dso_key] = existing
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(path)
        except Exception:
            _logger.exception("Failed to save transients.json")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Frame selection helpers
# --------------------------------------------------------------------------- #
def _select_filter(by_filter: dict, filter_name: str) -> tuple:
    """Return ``(paths, canonical_filter)`` for ``filter_name``, matching the
    FITS FILTER header case-insensitively so ``r`` and ``R`` resolve to the same
    subs. ``canonical_filter`` is the actual header spelling (e.g. ``R``) and is
    used for storage/reporting so the two spellings don't split into separate
    transients.json entries. Raises ValueError listing the available filters."""
    if filter_name in by_filter:
        return by_filter[filter_name], filter_name
    for key, val in by_filter.items():
        if key.lower() == filter_name.lower():
            return val, key
    available = ", ".join(sorted(by_filter)) or "(none)"
    raise ValueError(f"No '{filter_name}' subs found; available filters: {available}")


def _cluster_nights(paths: list, mjds: dict, gap_days: float) -> list:
    """Group paths into observing nights.

    Sorts by MJD and starts a new night wherever consecutive frames differ by
    more than ``gap_days``. Returns a list of (night_mjd, [paths]) ordered
    oldest→newest.
    """
    ordered = sorted(paths, key=lambda p: mjds[p])
    nights: list = []
    cur: list = []
    prev: Optional[float] = None
    for p in ordered:
        m = mjds[p]
        if prev is not None and (m - prev) > gap_days:
            nights.append(cur)
            cur = []
        cur.append(p)
        prev = m
    if cur:
        nights.append(cur)
    return [(mjds[n[0]], n) for n in nights]


# --------------------------------------------------------------------------- #
# Image building / PSF match / subtract
# --------------------------------------------------------------------------- #
def _nanmedian_and_coverage(frames: list) -> tuple:
    """Return (median image, per-pixel finite-frame count) for registered
    frames (which carry NaN at astroalign footprints)."""
    arr = np.stack(frames, axis=0).astype(np.float32)
    coverage = np.sum(np.isfinite(arr), axis=0).astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        img = np.nanmedian(arr, axis=0).astype(np.float32)
    return img, coverage


def _fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace NaN / uncovered pixels with the image median (stable input to
    convolution and subtraction)."""
    out = np.array(image, dtype=np.float32)
    bad = ~np.isfinite(out) | ~mask
    if bad.any():
        med = float(np.nanmedian(out[~bad])) if (~bad).any() else 0.0
        out[bad] = med
    return out


def _frame_fwhm(image: np.ndarray, default: float = 3.0) -> float:
    """Median stellar FWHM (pixels) from a sep detection, or ``default``."""
    try:
        objs, _ = _sep_detect(image, thresh_sigma=8.0)
    except Exception:
        return default
    if objs is None or len(objs) == 0:
        return default
    a = np.asarray(objs["a"], dtype=np.float64)
    b = np.asarray(objs["b"], dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not good.any():
        return default
    fwhm = 2.3548 * np.sqrt(a[good] * b[good])
    val = float(np.median(fwhm))
    return val if np.isfinite(val) and val > 0 else default


def _gaussian_match(image: np.ndarray, fwhm: float, target_fwhm: float) -> np.ndarray:
    """Blur ``image`` so its PSF matches ``target_fwhm`` (no-op if already ≥)."""
    from scipy.ndimage import gaussian_filter
    if target_fwhm <= fwhm:
        return image
    sigma = math.sqrt(max(0.0, target_fwhm ** 2 - fwhm ** 2)) / 2.3548
    if sigma < 0.3:
        return image
    return gaussian_filter(image, sigma)


def _robust_scale_bg(science: np.ndarray, template: np.ndarray,
                     mask: np.ndarray) -> tuple:
    """Fit science ≈ s·template + b over covered pixels with sigma-clipping."""
    m = mask & np.isfinite(science) & np.isfinite(template)
    xs = template[m].astype(np.float64).ravel()
    ys = science[m].astype(np.float64).ravel()
    if xs.size < 100:
        return 1.0, 0.0
    if xs.size > 200_000:
        idx = np.random.default_rng(0).choice(xs.size, 200_000, replace=False)
        xs, ys = xs[idx], ys[idx]
    s, b = 1.0, 0.0
    for _ in range(3):
        if xs.size < 10:
            break
        A = np.vstack([xs, np.ones_like(xs)]).T
        (s, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - (s * xs + b)
        std = float(np.std(resid))
        if std <= 0:
            break
        keep = np.abs(resid) < 3.0 * std
        xs, ys = xs[keep], ys[keep]
    return float(s), float(b)


def _psf_match_and_subtract(science: np.ndarray, template: np.ndarray,
                            coverage: np.ndarray) -> np.ndarray:
    """Difference image (science − PSF-matched template); positive ⇒ new flux."""
    sci = _fill(science, coverage)
    tmpl = _fill(template, coverage)

    # Optional: Alard–Lupton optimal subtraction if `ois` is installed.
    try:
        import ois  # noqa: F401
        diff, _opt, _krn, _bkg = ois.optimal_system(
            image=sci, refimage=tmpl, kernelshape=(11, 11), method="Bramich")
        return np.asarray(diff, dtype=np.float32)
    except Exception:
        pass  # fall back to the dependency-light Gaussian match below

    fwhm_s = _frame_fwhm(sci)
    fwhm_t = _frame_fwhm(tmpl)
    target = max(fwhm_s, fwhm_t)
    sci_m = _gaussian_match(sci, fwhm_s, target)
    tmpl_m = _gaussian_match(tmpl, fwhm_t, target)
    s, b = _robust_scale_bg(sci_m, tmpl_m, coverage)
    return (sci_m - (s * tmpl_m + b)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Detection + artifact rejection
# --------------------------------------------------------------------------- #
def _sep_detect(image: np.ndarray, thresh_sigma: float) -> tuple:
    """sep background-subtract + extract (positive sources). Returns
    (objs structured array, background rms). Mirrors cmd_diagram._detect_and_measure."""
    import sep
    try:
        sep.set_extract_pixstack(4_000_000)
    except Exception:
        pass
    data = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(data)
    data_sub = data - bkg.back()
    objs = None
    thresh = thresh_sigma
    for _attempt in range(4):
        try:
            objs = sep.extract(data_sub, thresh=thresh, err=bkg.globalrms)
            break
        except Exception as exc:
            if "pixel buffer full" not in str(exc):
                raise
            thresh *= 1.5
            _logger.warning("sep.extract pixstack overflow; retrying at thresh=%.1fσ", thresh)
    if objs is None:
        return np.array([]), float(bkg.globalrms)
    return objs, float(bkg.globalrms)


def _local_rms(diff: np.ndarray, ix: int, iy: int, r_in: int, r_out: int,
               fallback: float) -> float:
    """Robust background noise in an annulus around (ix, iy) of the difference
    image. Using a *local* rms instead of the global one keeps a bright galaxy
    core or a saturated-star residual from deflating the noise estimate and
    inflating SNR into the thousands. Falls back to the global rms if too few
    annulus pixels are available."""
    h, w = diff.shape
    y0, y1 = max(0, iy - r_out), min(h, iy + r_out + 1)
    x0, x1 = max(0, ix - r_out), min(w, ix + r_out + 1)
    box = diff[y0:y1, x0:x1]
    if box.size == 0:
        return fallback
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - ix, yy - iy)
    ann = box[(rr >= r_in) & (rr <= r_out) & np.isfinite(box)]
    if ann.size < 20:
        return fallback
    med = float(np.median(ann))
    mad = float(np.median(np.abs(ann - med)))
    rms = 1.4826 * mad
    return rms if rms > 0 else fallback


def _bright_source_mask(science: np.ndarray, template: np.ndarray,
                        coverage: np.ndarray, cfg: dict) -> np.ndarray:
    """Mask a grown region around the brightest (saturated) sources.

    Saturated stars clip and don't scale linearly, so their subtraction leaves a
    bright+dark dipole that looks like a new source. We flag the top-percentile
    pixels of the TEMPLATE and dilate, then reject candidates landing inside —
    killing the dominant false positive.

    Template only, deliberately. This used to threshold the science image too,
    which is self-defeating: a bright transient IS among the brightest pixels of
    the science image, so it masked itself out. Measured with synthetic
    injections on ngc5907 R, everything at or above 3200 ADU was rejected as a
    "saturated star" — the brighter the supernova, the more certainly it was
    discarded. Saturated stars are in the template as well (they were there on
    every prior night), so masking from the template alone loses nothing: the
    same injection run kept the false-positive count unchanged at 3 while
    recovery went from 8/21 to 14/21."""
    from scipy.ndimage import binary_dilation
    grow = int(cfg.get("bright_mask_grow_px", 10))
    pct = float(cfg.get("bright_mask_percentile", 99.9))
    mask = np.zeros(science.shape, dtype=bool)
    for img in (template,):
        finite = np.isfinite(img) & coverage
        if not finite.any():
            continue
        thr = float(np.percentile(img[finite], pct))
        mask |= finite & (img >= thr)
    if grow > 0 and mask.any():
        mask = binary_dilation(mask, iterations=grow)
    return mask


def _has_negative_lobe(diff: np.ndarray, ix: int, iy: int, r: int,
                       rms: float, dipole_sigma: float) -> bool:
    """True if a strong negative residual sits next to the positive peak — the
    signature of a mis-registration / bad-subtraction dipole, not a real source."""
    h, w = diff.shape
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    box = diff[y0:y1, x0:x1]
    if box.size == 0:
        return False
    return float(np.nanmin(box)) < -dipole_sigma * rms


def _reject_and_rank(diff: np.ndarray, objs, coverage: np.ndarray, rms: float,
                     cfg: dict, bright_mask: Optional[np.ndarray] = None) -> list:
    """Filter raw residuals to plausible new point sources, ranked by *local* SNR.

    A real supernova is a stellar point source that appears where the template is
    dark. The dominant false positives are (a) single hot/warm pixels and cosmic
    rays (sub-pixel ``a``), (b) saturated bright stars whose subtraction leaves an
    extended bright+dark dipole (large ``a``, sits inside ``bright_mask``). We
    bound the source size on both ends, reject anything in the bright-star mask,
    scale the dipole search to the source, and score by a locally-measured SNR so
    a bright galaxy core can't inflate it into the thousands."""
    if objs is None or len(objs) == 0:
        return []
    h, w = diff.shape
    margin = int(cfg.get("edge_margin_px", 16))
    # Shape cuts bounding the stellar locus. A supernova is a point source, so
    # we keep only compact, round residuals — rejecting single hot pixels (below
    # min_a) and extended galaxy-structure knots (above max_a or max_elong).
    #
    # The window was [0.6, 1.3], said to match "measured stellar a≈0.7". That
    # figure does not describe this system. Measured 2026-08-05 on the ngc5907 R
    # stack, real stars have median a=2.51 (FWHM 5.6px) and difference-image
    # residuals median a=2.83, so the old window admitted only 15% of genuine
    # point sources and rejected the rest as "extended" — it was selecting for
    # the sub-pixel artifacts it was meant to exclude. Widening it to the
    # measured stellar spread (p10 1.12 → p90 4.28) raised injection recovery
    # from 2/21 to 8/21 while *lowering* false positives from 5 to 3.
    #
    # KNOWN LIMITATION: sep's `a` is a second moment over pixels above the
    # detection threshold, so it grows with brightness at fixed PSF — the same
    # injected profile measures a=1.06 at 400 ADU and a=2.66 at 6400. Any fixed
    # window is therefore partly a brightness cut. A brightness-independent
    # shape test (compare against the frame's own stellar locus at matched flux)
    # would be the real fix; this window is only bounded by measurement.
    max_elong = float(cfg.get("max_elongation", 1.4))
    min_a = float(cfg.get("min_semimajor_px", 1.0))
    max_a = float(cfg.get("max_semimajor_px", 5.0))
    detect_sigma = float(cfg.get("detect_sigma", 5.0))
    dipole_sigma = float(cfg.get("dipole_sigma", 4.0))
    dipole_r = int(cfg.get("dipole_radius_px", 14))
    rms_r_in = int(cfg.get("local_rms_r_in_px", 6))
    rms_r_out = int(cfg.get("local_rms_r_out_px", 20))

    cands: list = []
    for o in objs:
        x, y = float(o["x"]), float(o["y"])
        a, b, peak = float(o["a"]), float(o["b"]), float(o["peak"])
        ix, iy = int(round(x)), int(round(y))
        if not (margin <= ix < w - margin and margin <= iy < h - margin):
            continue
        if not coverage[iy, ix]:
            continue
        if bright_mask is not None and bright_mask[iy, ix]:
            continue                        # saturated / bright-star residual
        if a < min_a:                       # hot pixel / cosmic ray (sub-pixel)
            continue
        if a > max_a:                       # extended: galaxy / bright-star blob
            continue
        elong = a / max(b, 1e-6)
        if elong > max_elong:               # trail / edge / galaxy residual
            continue
        local_rms = _local_rms(diff, ix, iy, rms_r_in, rms_r_out, rms)
        snr = peak / max(local_rms, 1e-6)
        if snr < detect_sigma:
            continue
        # Scale the dipole search with the source so extended residuals whose
        # negative lobe sits beyond the fixed radius are still caught. Compared
        # against the *global* rms: the local annulus is contaminated by the
        # lobe itself, which would mask the very dipole we're testing for.
        dr = max(dipole_r, int(math.ceil(3.0 * a)))
        if _has_negative_lobe(diff, ix, iy, dr, rms, dipole_sigma):
            continue
        # Roundish, bright residuals score highest.
        score = snr / (1.0 + (elong - 1.0))
        cands.append({
            "x": round(x, 2), "y": round(y, 2),
            "peak": round(peak, 3), "snr": round(snr, 2),
            "a": round(a, 2), "b": round(b, 2),
            "elongation": round(elong, 2), "score": round(score, 2),
        })
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def _apply_gaia_rejection(cands: list, cfg: dict) -> None:
    """Down-rank candidates that coincide with a catalogued Gaia point source
    (a variable star), in place. _identify_candidates has already annotated
    gaia_source_id / gaia_sep_arcsec."""
    reject_radius = float(cfg.get("gaia_reject_arcsec", 2.0))
    for c in cands:
        sep_as = c.get("gaia_sep_arcsec")
        if c.get("gaia_source_id") and sep_as is not None and sep_as <= reject_radius:
            c["gaia_rejected"] = True
            c["score"] = round(c["score"] * 0.1, 2)   # heavy down-rank, not deleted
    cands.sort(key=lambda c: c["score"], reverse=True)


def _resolve_simbad(cands: list, cfg: dict,
                    progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """Annotate each candidate that already has RA/Dec with the nearest SIMBAD
    object (``simbad_name`` + ``simbad_otype``) within ``simbad_radius_arcsec``.

    Where Gaia only yields a catalogue number, SIMBAD gives a human-readable name
    AND an object type — so a hit usually means the source is a *known* object
    (named variable star, galaxy nucleus, AGN) rather than a supernova. Fully
    fail-safe: any import/network/query failure leaves the candidates unchanged."""
    radius_as = float(cfg.get("simbad_radius_arcsec", 10.0))
    todo = [c for c in cands if c.get("ra_deg") is not None]
    if not todo:
        return
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad
    except Exception:
        return
    sim = Simbad()
    try:
        sim.add_votable_fields("otype")
    except Exception:
        pass
    if progress_cb:
        progress_cb(f"identify: SIMBAD lookup for {len(todo)} candidate(s)…")
    for c in todo:
        try:
            sky = SkyCoord(c["ra_deg"], c["dec_deg"], unit="deg")
            res = sim.query_region(sky, radius=radius_as * u.arcsec)
        except Exception:
            _logger.exception("SIMBAD query failed")
            if progress_cb:
                progress_cb("identify: SIMBAD query failed, leaving names blank")
            break
        if res is None or len(res) == 0:
            continue
        cols = {col.lower(): col for col in res.colnames}
        mid, ot = cols.get("main_id"), cols.get("otype")
        try:
            if mid is not None:
                name = res[mid][0]
                name = name.decode() if isinstance(name, bytes) else str(name)
                c["simbad_name"] = name.strip()
            if ot is not None:
                otype = res[ot][0]
                otype = otype.decode() if isinstance(otype, bytes) else str(otype)
                c["simbad_otype"] = otype.strip()
        except Exception:
            continue


def _apply_newness_gate(cands: list, template: np.ndarray, science: np.ndarray,
                        coverage: np.ndarray, cfg: dict) -> list:
    """Keep only genuinely *new* sources.

    A transient is present in the science image but absent from the template; an
    imperfectly-subtracted existing star (the dominant false positive) is present
    in BOTH. For each candidate we aperture-measure the background-subtracted flux
    in the template and the science image and drop it unless the science has a
    real positive excess AND the template is near-dark there. Annotates each kept
    candidate with template_flux / science_flux.
    """
    if not cands:
        return cands
    import sep
    aper_r = float(cfg.get("newness_aperture_px", 4.0))
    frac = float(cfg.get("newness_max_template_frac", 0.5))
    t = np.ascontiguousarray(_fill(template, coverage), dtype=np.float32)
    s = np.ascontiguousarray(_fill(science, coverage), dtype=np.float32)
    t_sub = t - sep.Background(t).back()
    s_sub = s - sep.Background(s).back()
    xs = np.array([c["x"] for c in cands], dtype=np.float64)
    ys = np.array([c["y"] for c in cands], dtype=np.float64)
    tf, _e1, _f1 = sep.sum_circle(t_sub, xs, ys, aper_r)
    sf, _e2, _f2 = sep.sum_circle(s_sub, xs, ys, aper_r)
    kept: list = []
    for c, tflux, sflux in zip(cands, tf, sf):
        c["template_flux"] = round(float(tflux), 1)
        c["science_flux"] = round(float(sflux), 1)
        if sflux <= 0:                         # no positive source in science
            continue
        if tflux > frac * sflux:               # already present in template
            continue
        kept.append(c)
    return kept


def _find_nucleus(science: np.ndarray, coverage: np.ndarray) -> tuple:
    """Approximate the host-galaxy nucleus as the brightest smoothed pixel in
    the central region (for reporting candidate offsets)."""
    from scipy.ndimage import gaussian_filter
    h, w = science.shape
    filled = _fill(science, coverage)
    sm = gaussian_filter(filled, 3.0)
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    sub = sm[cy0:cy1, cx0:cx1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return float(ix + cx0), float(iy + cy0)


# --------------------------------------------------------------------------- #
# Synthetic injection (for the acceptance self-test)
# --------------------------------------------------------------------------- #
def _inject_into(frame: np.ndarray, sources: list, fwhm: float) -> np.ndarray:
    """Add Gaussian point sources of given total flux into a registered frame."""
    out = np.array(frame, dtype=np.float32)
    sigma = max(0.8, fwhm / 2.3548)
    r = int(math.ceil(4 * sigma))
    h, w = out.shape
    for src in sources:
        cx, cy, flux = float(src["x"]), float(src["y"]), float(src["flux_adu"])
        amp = flux / (2.0 * math.pi * sigma * sigma)
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        psf = amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)))
        patch = out[y0:y1, x0:x1]
        finite = np.isfinite(patch)
        patch[finite] += psf[finite].astype(np.float32)
        out[y0:y1, x0:x1] = patch
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _stamp(image: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    h, w = image.shape
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half)
    return image[y0:y1, x0:x1]


def _show(ax, data: np.ndarray, title: str) -> None:
    ax.set_facecolor(_DARK_AXES)
    if data.size:
        vmin, vmax = ZScaleInterval().get_limits(np.nan_to_num(data))
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
    ax.set_title(title, color="white", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _triptych(template: np.ndarray, science: np.ndarray, diff: np.ndarray,
              cands: list, out_path: Path, half: int) -> None:
    """template | science | difference panels, one row per top candidate (or a
    single full-frame difference row if there are none)."""
    rows = max(1, len(cands))
    fig = Figure(figsize=(9, 3 * rows), dpi=130)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(_DARK_BG)

    if not cands:
        for j, (img, name) in enumerate(
                [(template, "Template"), (science, "Science (latest night)"),
                 (diff, "Difference")]):
            ax = fig.add_subplot(1, 3, j + 1)
            _show(ax, img, name)
    else:
        for i, c in enumerate(cands):
            cx, cy = c["x"], c["y"]
            panels = [
                (_stamp(template, cx, cy, half), "Template"),
                (_stamp(science, cx, cy, half), "Science"),
                (_stamp(diff, cx, cy, half),
                 f"Diff  SNR {c['snr']:.1f}" + (" [Gaia]" if c.get("gaia_rejected") else "")),
            ]
            for j, (img, name) in enumerate(panels):
                ax = fig.add_subplot(rows, 3, i * 3 + j + 1)
                _show(ax, img, name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpeg", facecolor=fig.get_facecolor())


def _notify_push(dso: str, filt: str, top: dict, image_path: Path) -> None:
    """Best-effort Pushover alert for a strong candidate. Never raises."""
    try:
        from utils import pushover
        loc = (f"RA {top['ra_deg']:.4f} Dec {top['dec_deg']:+.4f}"
               if top.get("ra_deg") is not None else f"({top['x']:.0f},{top['y']:.0f})")
        msg = f"Possible new source in {dso} [{filt}]: SNR {top['snr']:.1f} at {loc}"
        img = str(image_path) if image_path and os.path.exists(image_path) else None
        pushover.push_message(msg, img)
    except Exception:
        _logger.exception("transient push notification failed")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_transient_search(
    dso_name: str,
    filter_name: str,
    image_dir: Path,
    output_plot_path: Path,
    astap_exe: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    inject: Optional[list] = None,
    top_n: Optional[int] = None,
    identify: Optional[bool] = None,
) -> dict:
    """Difference the most recent night against all prior nights and return the
    entry saved to transients.json. ``inject`` (list of {x,y,flux_adu}) adds
    synthetic point sources into the science frames before differencing — used
    by the acceptance self-test. ``top_n`` overrides how many candidates are
    reported (the injection test raises it to account for every injection).
    ``identify`` overrides the config's ASTAP+Gaia cross-match toggle (pass False
    to skip the slow plate-solve/Gaia lookup).
    """
    from stacking import stacker  # local import avoids circular at module load
    from transit_search.transit import (
        _find_dso_dir, _obs_time_mjd, _register_only, _identify_candidates,
        _solve_field_wcs)

    def _notify(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def _ck() -> None:
        if cancel_cb and cancel_cb():
            raise TransientCancelled()

    cfg = _transient_cfg()
    gap_days = float(cfg.get("night_gap_days", 0.3))
    top_n = int(top_n if top_n is not None else cfg.get("top_n", 5))
    triptych_max_rows = int(cfg.get("triptych_max_rows", 8))
    detect_sigma = float(cfg.get("detect_sigma", 5.0))
    min_cov_frac = float(cfg.get("min_coverage_fraction", 0.5))
    stamp_half = int(cfg.get("stamp_half_px", 30))
    identify = bool(identify if identify is not None else cfg.get("identify_candidates", True))
    gaia_match_radius = float(cfg.get("gaia_match_radius_arcsec", 3.0))
    alert_snr = float(cfg.get("alert_snr", 8.0))
    if not astap_exe:
        astap_exe = _config.data().get("hardware", {}).get("astap_exe", "")

    dso_dir = _find_dso_dir(image_dir, dso_name)
    if dso_dir is None:
        raise ValueError(f"No image directory found for '{dso_name}'")
    light_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if f.parent.name.upper() == "LIGHT"),
        key=lambda p: p.stat().st_mtime,
    )
    if not light_files:
        raise ValueError(f"No LIGHT frames under {dso_dir}")

    by_filter = stacker.group_by_filter(light_files)
    # Canonicalise the filter to the header spelling so `diff … r` and `diff … R`
    # read the same subs AND persist under the same key.
    paths, filter_name = _select_filter(by_filter, filter_name)
    _ck()

    # --- split into nights ------------------------------------------------- #
    mjds = {p: _obs_time_mjd(p) for p in paths}
    nights = _cluster_nights(paths, mjds, gap_days)
    if len(nights) < 2:
        raise ValueError(
            f"transient needs ≥2 nights of {dso_name}/{filter_name}; found "
            f"{len(nights)} ({len(paths)} frames). Image it again on another night.")
    *template_nights, science_night = nights
    template_paths = [p for _m, ps in template_nights for p in ps]
    science_paths = science_night[1]
    baseline_days = float(nights[-1][0] - nights[0][0])
    _notify(f"{len(nights)} nights: template={len(template_paths)} frames over "
            f"{len(template_nights)} night(s), science={len(science_paths)} frames "
            f"(latest), baseline {baseline_days:.1f}d")
    _ck()

    # --- register template+science on one common grid ---------------------- #
    all_paths = list(template_paths) + list(science_paths)
    registered, accepted = _register_only(all_paths, progress_cb=progress_cb)
    if len(registered) < 2:
        raise ValueError("registration failed — too few frames aligned")
    tmpl_set = {str(p) for p in template_paths}
    tmpl_reg = [r for r, p in zip(registered, accepted) if str(p) in tmpl_set]
    sci_reg = [r for r, p in zip(registered, accepted) if str(p) not in tmpl_set]
    if not tmpl_reg or not sci_reg:
        raise ValueError(
            f"after registration: {len(tmpl_reg)} template / {len(sci_reg)} science "
            "frames survived — need ≥1 of each")
    _ck()

    if inject:
        fwhm = _frame_fwhm(_nanmedian_and_coverage(sci_reg)[0])
        sci_reg = [_inject_into(f, inject, fwhm) for f in sci_reg]
        _notify(f"injected {len(inject)} synthetic source(s) at FWHM≈{fwhm:.1f}px")

    template_img, tmpl_cov = _nanmedian_and_coverage(tmpl_reg)
    science_img, sci_cov = _nanmedian_and_coverage(sci_reg)
    coverage = (
        (tmpl_cov >= max(1, math.ceil(min_cov_frac * len(tmpl_reg)))) &
        (sci_cov >= max(1, math.ceil(min_cov_frac * len(sci_reg)))))

    _notify("PSF-matching and subtracting…")
    diff = _psf_match_and_subtract(science_img, template_img, coverage)
    diff = np.where(coverage, diff, 0.0).astype(np.float32)
    _ck()

    # Debug artifacts for inspection (best-effort). The science median is written
    # WITH a real science-frame header so ASTAP can plate-solve it directly — and
    # because it lives on the same registered grid as the candidate (x,y), its
    # WCS maps them correctly. (Solving an arbitrary raw frame, as before, gave a
    # WCS for a different grid and so matched no Gaia sources.)
    scratch = output_plot_path.parent
    try:
        sci_header = fits.getheader(science_paths[-1])
    except Exception:
        sci_header = None
    science_fits = scratch / "transient_science.fits"
    for name, arr, hdr in (("transient_template.fits", template_img, None),
                           ("transient_science.fits", science_img, sci_header),
                           ("transient_diff.fits", diff, sci_header)):
        try:
            fits.writeto(str(scratch / name), arr, header=hdr, overwrite=True)
        except Exception:
            _logger.warning("could not write %s", name)

    bright_mask = _bright_source_mask(science_img, template_img, coverage, cfg)
    objs, rms = _sep_detect(diff, detect_sigma)
    _notify(f"{0 if objs is None else len(objs)} raw residuals; rejecting artifacts…")
    cands = _reject_and_rank(diff, objs, coverage, rms, cfg, bright_mask=bright_mask)
    cands = _apply_newness_gate(cands, template_img, science_img, coverage, cfg)
    _notify(f"{len(cands)} candidate(s) after artifact + newness cuts")
    _ck()

    if identify and cands and science_fits.exists():
        top_for_id = cands[:top_n]
        try:
            # The WCS has to be solved here and passed in. _identify_candidates
            # used to take (path, astap_exe, ...) and do its own solve; d10dd78
            # split that out into _solve_field_wcs and this caller was not
            # updated, so every call raised TypeError into the except below for
            # 67 commits — silently costing the transient search its Gaia and
            # SIMBAD identification, and with it the only condition that
            # suppresses a Pushover alert for a known variable star.
            #
            # Field stars come from the science image, not from `objs`: those
            # are difference residuals, and a solver needs the actual star
            # field. They feed the Gaia point-match fallback, which is the path
            # that runs whenever ASTAP is not configured.
            star_objs, _star_rms = _sep_detect(science_img, detect_sigma)
            if star_objs is None or len(star_objs) == 0:
                raise ValueError("no field stars detected for the astrometric solve")
            positions = np.column_stack([
                np.asarray(star_objs["x"], dtype=np.float64),
                np.asarray(star_objs["y"], dtype=np.float64)])
            star_flux = np.asarray(star_objs["flux"], dtype=np.float64)
            wcs = _solve_field_wcs(science_fits, astap_exe, positions,
                                   star_flux, progress_cb)
            if wcs is None:
                raise ValueError("astrometric solve failed (ASTAP and Gaia fallback)")
            _identify_candidates(wcs, top_for_id,
                                 match_radius_arcsec=gaia_match_radius,
                                 progress_cb=progress_cb)
            _apply_gaia_rejection(top_for_id, cfg)
        except TypeError:
            # Separated on purpose. Identification is allowed to fail softly
            # for real-world reasons (no solver, no network, no match), and
            # that tolerance is what hid a call-signature mismatch for 67
            # commits behind the same bland "Gaia identification failed". A
            # TypeError here is never a runtime condition — it is always this
            # caller disagreeing with transit.py — so it gets its own message.
            _logger.exception(
                "Gaia identification BROKEN: bad call signature into "
                "transit_search.transit — this is a code bug, not a failed solve")
        except Exception:
            _logger.exception("Gaia identification failed")
        try:
            _resolve_simbad(top_for_id, cfg, progress_cb)
        except Exception:
            _logger.exception("SIMBAD identification failed")
        cands = top_for_id + cands[top_n:]
        cands.sort(key=lambda c: c["score"], reverse=True)

    top = cands[:top_n]
    nucleus = _find_nucleus(science_img, coverage)
    for c in top:
        c["offset_from_nucleus_px"] = round(
            math.hypot(c["x"] - nucleus[0], c["y"] - nucleus[1]), 1)

    _notify(f"rendering {len(top)} candidate(s)…")
    _triptych(template_img, science_img, diff, top[:triptych_max_rows],
              output_plot_path, stamp_half)

    entry = {
        "dso": dso_dir.name,
        "filter": filter_name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_template_frames": len(tmpl_reg),
        "n_science_frames": len(sci_reg),
        "n_template_nights": len(template_nights),
        "baseline_days": round(baseline_days, 2),
        "n_candidates": len(cands),
        "nucleus_xy": [round(nucleus[0], 1), round(nucleus[1], 1)],
        "candidates": top,
    }
    save_transients(dso_dir.name, filter_name, entry)

    if top and top[0]["snr"] >= alert_snr and not top[0].get("gaia_rejected"):
        _notify_push(dso_dir.name, filter_name, top[0], output_plot_path)

    return entry


def _cli() -> None:
    """Run a transient search from the command line.

        .venv/Scripts/python transient_search/difference.py ngc5907 R
        .venv/Scripts/python -m transient_search.difference m101 r --no-identify

    Prints progress and the ranked candidates; writes the triptych + debug FITS
    to the scratch dir (or --out).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Transient / supernova image-differencing search")
    ap.add_argument("dso", help="target name, e.g. ngc5907")
    ap.add_argument("filter", help="filter name, e.g. R")
    ap.add_argument("--out", default=None, help="output triptych JPEG path")
    ap.add_argument("--top-n", type=int, default=None, help="max candidates to report")
    ap.add_argument("--no-identify", action="store_true",
                    help="skip the ASTAP plate-solve + Gaia lookup (much faster)")
    args = ap.parse_args()

    cfg = _config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    scratch = root / cfg["scratch"]["directory"]
    scratch.mkdir(parents=True, exist_ok=True)
    out = (Path(args.out) if args.out
           else scratch / f"transient_{args.dso.replace(' ', '_')}_{args.filter}.jpg")

    def _pg(msg: str) -> None:
        print(f"  > {msg}", flush=True)

    entry = run_transient_search(
        args.dso, args.filter, image_dir, out,
        progress_cb=_pg, top_n=args.top_n,
        identify=False if args.no_identify else None,
    )

    print("\n=== result ===")
    print(f"{entry['dso']} [{entry['filter']}] — "
          f"{entry['n_template_frames']} template + {entry['n_science_frames']} science "
          f"frames over {entry['baseline_days']:.1f}d ({entry['n_template_nights']} template nights)")
    cands = entry.get("candidates", [])
    survivors = [c for c in cands if not c.get("gaia_rejected")]
    print(f"{len(survivors)} new-source candidate(s) "
          f"({len(cands) - len(survivors)} matched known Gaia stars):")
    for i, c in enumerate(cands, 1):
        loc = (f"RA {c['ra_deg']:.4f} Dec {c['dec_deg']:+.4f}"
               if c.get("ra_deg") is not None else f"({c['x']:.0f},{c['y']:.0f})")
        tag = " [Gaia star]" if c.get("gaia_rejected") else ""
        off = c.get("offset_from_nucleus_px")
        off_str = f" {off:.0f}px from nucleus" if off is not None else ""
        print(f"  {i}. {loc} SNR={c['snr']:.1f} elong={c['elongation']:.2f}{off_str}{tag}")
    print(f"\ntriptych: {out}" if out.exists() else "\n(no triptych written)")


if __name__ == "__main__":
    _cli()
