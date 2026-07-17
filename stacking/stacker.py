"""FITS image stacker with optional calibration frame support and FWHM-based weighting.

Stacking methods
----------------
MEAN             – arithmetic mean (fast, no rejection)
MEDIAN           – pixel-wise median (robust against cosmics, lower SNR than mean)
SIGMA_CLIP       – sigma-clipped mean (rejects outliers per pixel)
FWHM_WEIGHTED    – weighted mean where weight = 1 / FWHM²  (rewards sharper frames)
SIGMA_CLIP_FWHM  – sigma-clip outliers per pixel, then weighted mean of the
                   survivors by 1 / FWHM²; best general-purpose result on
                   dithered data because it rejects hot pixels/cosmics AND
                   rewards better-seeing frames.

Calibration pipeline (applied to each light frame before stacking)
------------------------------------------------------------------
  corrected = (raw − master_bias − master_dark) / master_flat

Each master is built as the median of its calibration frames.

Filter handling
---------------
Frames are grouped by the FITS FILTER header keyword before stacking.
stack_directory() produces one output file per filter, named e.g. stack_Ha.fits.
LiveStacker maintains a separate frame list per filter and only restacks the
filter group that received a new frame. Flats are filter-specific; pass
flat_dirs={filter: Path} to supply per-filter flat directories. Darks and
biases are filter-agnostic.
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_logger = logging.getLogger(__name__)

from utils.cancellation import Cancelled


def _ckpt(cancel_cb: Optional[Callable[[], bool]]) -> None:
    """Raise :class:`Cancelled` if ``cancel_cb`` reports a cancellation."""
    if cancel_cb is not None and cancel_cb():
        raise Cancelled()


try:
    from fits_processing.fitsfwhm import calculate_fwhm as _calculate_fwhm
    _FWHM_AVAILABLE = True
except Exception:
    _FWHM_AVAILABLE = False
    _logger.warning("photutils not available — FWHM measurement disabled; FWHM_WEIGHTED will use equal weights")

try:
    import astroalign as _astroalign
    _REGISTER_AVAILABLE = True
except Exception:
    _REGISTER_AVAILABLE = False
    _logger.warning("astroalign not available — frame registration disabled")

# sep's default pixel stack (300k pixels above threshold) is sized for sparse
# fields and overflows on dense ones: a globular like m13/m92 puts millions of
# pixels over the threshold on a 61 MP frame, sep raises "internal pixel buffer
# full", astroalign re-raises it as the misleading "Input type for source not
# supported", and every frame fails to register. _count_sources swallows the
# same error and returns 0, silently breaking the reference pick too. Raise the
# limits once at import; the cost is a larger transient buffer inside sep.
try:
    import sep as _sep
    _sep.set_extract_pixstack(5_000_000)
    _sep.set_sub_object_limit(4096)
except Exception:
    _logger.warning("sep not available — extraction limits left at defaults")


# ---------------------------------------------------------------------------
# Stacking method
# ---------------------------------------------------------------------------

class StackMethod(Enum):
    MEAN = auto()
    MEDIAN = auto()
    SIGMA_CLIP = auto()
    FWHM_WEIGHTED = auto()
    SIGMA_CLIP_FWHM = auto()


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _load_fits_2d(path: Path) -> np.ndarray:
    """Load a single FITS file, squeeze to 2-D, return float32."""
    with fits.open(path) as hdul:
        data = np.squeeze(hdul[0].data).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D frame after squeeze, got shape {data.shape} in {path}")
    return data


def _load_cube(paths: list[Path]) -> np.ndarray:
    """Load multiple FITS files into a (N, H, W) float32 array."""
    if not paths:
        raise ValueError("No paths provided to _load_cube")
    frames = []
    expected_shape: Optional[tuple] = None
    for p in paths:
        frame = _load_fits_2d(p)
        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Frame shape mismatch: {frame.shape} != {expected_shape} in {p}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _build_master(paths: list[Path]) -> Optional[np.ndarray]:
    """Median-combine a list of FITS frames into a master frame."""
    if not paths:
        return None
    cube = _load_cube(paths)
    return np.median(cube, axis=0).astype(np.float32)


def build_master_bias(bias_paths: list[Path]) -> Optional[np.ndarray]:
    """Build a master bias from a list of bias frames."""
    return _build_master(bias_paths)


def build_master_dark(
    dark_paths: list[Path],
    master_bias: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build a master dark (bias-subtracted) from a list of dark frames."""
    master = _build_master(dark_paths)
    if master is not None and master_bias is not None:
        master = master - master_bias
    return master


def build_master_flat(
    flat_paths: list[Path],
    master_bias: Optional[np.ndarray] = None,
    master_dark: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build a normalised master flat from a list of flat frames."""
    master = _build_master(flat_paths)
    if master is None:
        return None
    if master_bias is not None:
        master = master - master_bias
    if master_dark is not None:
        master = master - master_dark
    norm = float(np.mean(master))
    if norm == 0.0:
        raise ValueError("Master flat normalisation factor is zero")
    return (master / norm).astype(np.float32)


def _calibrate(
    frame: np.ndarray,
    master_bias: Optional[np.ndarray],
    master_dark: Optional[np.ndarray],
    master_flat: Optional[np.ndarray],
) -> np.ndarray:
    result = frame.copy()
    if master_bias is not None:
        result -= master_bias
    if master_dark is not None:
        result -= master_dark
    if master_flat is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(master_flat != 0, result / master_flat, result)
    return result


# ---------------------------------------------------------------------------
# FWHM measurement
# ---------------------------------------------------------------------------

def _get_arcsec_per_pixel() -> float:
    try:
        from configs import config
        return float(config.data()["nina"]["arc_sec_per_pixel"])
    except Exception:
        return 1.0


def _count_sources(frame: np.ndarray) -> int:
    """Count detected stars in a frame using sep (astroalign dependency).

    Counts on the despiked frame: with uncalibrated lights, hot pixels dwarf
    the real star count (observed: 74,805 "sources" on an sh2-92 Ha sub whose
    true star count was ~600) and would steer the reference pick.

    Detection is deliberately coarse (thresh 8σ, deblending off): this is only
    a *relative* frame-quality proxy for the reference pick, not a science
    catalogue. On a dense field (globular, rich Milky Way) the default 3σ +
    deblend spends ~60 s/frame carving 35 k blended sources out of the cluster
    and nebulosity — pure waste for a comparison, and dominated by noise so the
    ranking is worse anyway. 8σ + no deblend gives the same *ordering* of frame
    quality in ~1.5 s/frame (40× faster on m92).
    """
    try:
        import sep
        data = np.ascontiguousarray(_despike(frame.astype(float)))
        bkg = sep.Background(data)
        sources = sep.extract(data - bkg.back(), thresh=8.0, err=bkg.rms(),
                              deblend_cont=1.0)
        return len(sources)
    except Exception:
        return 0


def _despike(frame: np.ndarray) -> np.ndarray:
    """3×3 median filter: annihilates single-pixel spikes, barely touches stars.

    Runs on the GPU when torch+CUDA is present (stacking/gpu_accel.py,
    ~4 s → ~0.1 s on 61 MP frames); scipy otherwise.

    Used on the DETECTION side of registration only (find_transform control
    points, reference source counts) — transforms are always applied to the
    original data. Without darks (calibration dirs are unset in config), hot
    pixels stay in the lights and outshine the stars on narrowband channels;
    astroalign then matches frames on the hot-pixel pattern, which is fixed to
    the sensor, and returns identity transforms with perfect residuals while
    the real sky drifts. Observed on sh2-92 Ha (2026-07-11): 31/31 frames
    "aligned" at identity, sky drifted 63 px over the night, every star smeared
    away — the stack's sharpest objects were the hot pixels themselves.
    """
    try:
        from stacking import gpu_accel
        out = gpu_accel.median3(frame)
        if out is not None:
            return out
    except Exception:
        pass
    from scipy.ndimage import median_filter
    return median_filter(frame, size=3)


def _reference_index_by_fwhm(fwhm_scores: list[float]) -> Optional[int]:
    """Return the index of the frame with the sharpest stars (lowest positive
    FWHM), or None if no frame has a measured FWHM.

    This is the preferred reference picker: the sharpest frame gives astroalign
    the cleanest control points and aligns the rest most reliably. The old
    "most detected sources" heuristic (_best_reference_idx) reliably picked the
    *worst* frame on low-SNR channels — the blurriest/noisiest sub also has the
    most detections, and astroalign could not match its bloated/noise centroids.
    Observed on ngc5907 B: the most-sources frame (also the worst FWHM, 13.8 px)
    aligned only 1/17 others, while every other candidate aligned 16-17/17.

    Guards against mis-measured FWHM: on a noise-dominated frame the per-star
    Gaussian fits collapse onto hot pixels, so the mean FWHM reads sub-pixel
    (~0.7 px). Picking that frame is just as bad as the old heuristic (observed
    on ngc5907 R: a 0.74 px "sharpest" reference aligned only 1/34). A real star
    cannot be much sharper than the session median, so we discard any frame whose
    FWHM is below 0.5x the median of measured frames before taking the minimum.
    """
    positive = [(f, i) for i, f in enumerate(fwhm_scores) if f and f > 0.0]
    if not positive:
        return None
    median_fwhm = float(np.median([f for f, _ in positive]))
    floor = 0.5 * median_fwhm
    plausible = [(f, i) for f, i in positive if f >= floor]
    if not plausible:
        # Every measured frame is implausibly sharp (degenerate fits across the
        # board) — fall back to the source-count picker rather than trust them.
        _logger.warning(
            "FWHM reference picker: all %d measured frames below floor %.2f px "
            "(median %.2f px) — falling back to source count",
            len(positive), floor, median_fwhm,
        )
        return None
    n_excluded = len(positive) - len(plausible)
    best_fwhm, best_idx = min(plausible)
    _logger.info(
        "Registration reference = frame %d (sharpest plausible, FWHM %.2f px; "
        "median %.2f px, excluded %d mis-measured)",
        best_idx, best_fwhm, median_fwhm, n_excluded,
    )
    return best_idx


def _best_reference_idx(frames: list[np.ndarray]) -> int:
    """
    Sample up to 10 evenly-spaced frames and return the index of the one
    with the most detected sources. Fallback reference picker used only when no
    FWHM is available — prefer _reference_index_by_fwhm (see its docstring for
    why source count is a poor criterion on noisy channels).
    """
    step = max(1, len(frames) // 10)
    candidates = list(range(0, len(frames), step))[:10]
    best_idx = candidates[0]
    best_count = -1
    for i in candidates:
        count = _count_sources(frames[i])
        _logger.debug("Reference candidate frame %d: %d sources", i, count)
        if count > best_count:
            best_count = count
            best_idx = i
    _logger.info("Selected frame %d as registration reference (%d sources)", best_idx, best_count)
    return best_idx


_REG_MIN_MATCHED_STARS = 10
_REG_MAX_MEDIAN_RESIDUAL_PX = 2.0

# astroalign builds its matching triangles from only the N brightest detected
# sources (default 50). When two frames have very different noise floors — e.g.
# subs taken on nights with different sky background — their brightest-50 sets
# don't overlap, and registration fails with "triangles exhausted" even though
# the field is identical. Raising this so enough real stars enter both sets
# bridges those frames. Measured on ngc5907 R (background 600 vs 250 across
# nights): 50 -> 8/35 frames registered, 200 -> 35/35. 200 is also faster than
# 100 here, because failed matches are what burn time (they exhaust the full
# triangle list before giving up).
_REG_MAX_CONTROL_POINTS = 200


def _register_frames(
    frames: list[np.ndarray],
    fwhm_scores: Optional[list[float]] = None,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Register all frames to the best reference frame using an affine transform.
    The reference is chosen as the frame with the most detected stars.

    Returns (registered_frames, surviving_indices) where surviving_indices[i]
    is the index in the *input* list that produced registered_frames[i].  The
    caller must use these indices to keep any parallel per-frame data (e.g.
    FWHM weights) in sync with the returned frame list.

    Frames are dropped when:
      - astroalign cannot find a transform at all,
      - the transform was fit from fewer than _REG_MIN_MATCHED_STARS pairs, or
      - the median residual of the matched pairs exceeds _REG_MAX_MEDIAN_RESIDUAL_PX
        (a successful affine fit on garbage matches will still return — this
        catches that case).
    """
    if not _REGISTER_AVAILABLE or len(frames) < 2:
        return frames, list(range(len(frames)))

    ref_idx = _reference_index_by_fwhm(fwhm_scores) if fwhm_scores is not None else None
    if ref_idx is None:
        ref_idx = _best_reference_idx(frames)
    reference = frames[ref_idx]
    # Transforms are found on despiked copies (hot pixels are fixed to the
    # sensor and would vote for the identity transform — see _despike) but
    # applied to the original frames so no real signal is filtered.
    reference_det = _despike(reference)
    registered = [reference]
    surviving_indices = [ref_idx]
    failed = 0
    poor_qa = 0
    for i, frame in enumerate(frames):
        if i == ref_idx:
            continue
        try:
            transform, (src_pts, dst_pts) = _astroalign.find_transform(
                _despike(frame), reference_det,
                max_control_points=_REG_MAX_CONTROL_POINTS,
            )
        except Exception as exc:
            failed += 1
            _logger.warning("Registration failed for frame %d: %s", i, exc)
            continue

        n_matched = len(src_pts) if src_pts is not None else 0
        if n_matched < _REG_MIN_MATCHED_STARS:
            poor_qa += 1
            _logger.warning(
                "Frame %d dropped — only %d matched stars (< %d)",
                i, n_matched, _REG_MIN_MATCHED_STARS,
            )
            continue

        transformed = transform(np.asarray(src_pts))
        residuals = np.sqrt(np.sum((transformed - np.asarray(dst_pts)) ** 2, axis=1))
        median_residual = float(np.median(residuals))
        if median_residual > _REG_MAX_MEDIAN_RESIDUAL_PX:
            poor_qa += 1
            _logger.warning(
                "Frame %d dropped — registration residual %.2f px > %.2f",
                i, median_residual, _REG_MAX_MEDIAN_RESIDUAL_PX,
            )
            continue

        try:
            aligned = footprint = None
            try:
                from stacking import gpu_accel
                res = gpu_accel.apply_affine(
                    np.asarray(transform.params), frame, reference.shape)
                if res is not None:
                    aligned, footprint = res
            except Exception:
                _logger.exception("GPU warp failed — using CPU apply_transform")
            if aligned is None:
                aligned, footprint = _astroalign.apply_transform(transform, frame, reference)
        except Exception as exc:
            failed += 1
            _logger.warning("apply_transform failed for frame %d: %s", i, exc)
            continue

        aligned = aligned.astype(np.float32)
        # astroalign's footprint is True for pixels that fell outside the
        # source frame after the transform — those carry no real signal, so
        # mark them NaN and let the combine step ignore them.
        aligned[footprint.astype(bool)] = np.nan
        registered.append(aligned)
        surviving_indices.append(i)

    _logger.info(
        "Registration: %d/%d frames aligned, %d failed, %d rejected by QA",
        len(registered), len(frames), failed, poor_qa,
    )
    return registered, surviving_indices


def _measure_fwhm(path: Path) -> float:
    """Return mean FWHM in pixels, or 0.0 if measurement fails / no stars found."""
    return _measure_fwhm_and_stars(path)[0]


def _measure_fwhm_and_stars(path: Path) -> tuple[float, int]:
    """Return (mean FWHM px, detected star count).

    Both come from the same source-detection pass in _calculate_fwhm, so this
    costs no more than measuring FWHM alone. Returns (0.0, 0) if detection
    fails entirely; returns (0.0, count) if stars were found but the FWHM fit
    was degenerate — callers treat FWHM 0.0 as "unmeasured".
    """
    if not _FWHM_AVAILABLE:
        return 0.0, 0
    try:
        fwhm_px, _, count, _ = _calculate_fwhm(path, arcsec_per_pixel=_get_arcsec_per_pixel())
        count = int(count)
        if count > 0 and fwhm_px > 0:
            return float(fwhm_px), count
        return 0.0, count
    except Exception as exc:
        _logger.warning("FWHM/star measurement failed for %s: %s", path.name, exc)
    return 0.0, 0


def _split_cached(
    paths: list[Path],
    precomputed: "Optional[dict[Path, tuple[float, int]]]",
) -> "tuple[dict[Path, tuple[float, int]], list[Path]]":
    """Split *paths* into cache hits and misses against *precomputed*.

    *precomputed* maps a frame path to its already-measured (FWHM px, star
    count) — typically loaded from a `stats`/`bad` frame_stats.json cache so the
    stacker doesn't redo the detection pass. Returns ({path: (fwhm_px, count)}
    for hits, [paths still needing measurement]). A None/empty map means every
    path is a miss (unchanged behaviour).
    """
    if not precomputed:
        return {}, list(paths)
    hits = {p: precomputed[p] for p in paths if p in precomputed}
    misses = [p for p in paths if p not in precomputed]
    return hits, misses


def _combine_tile(
    cube_tile: np.ndarray,
    method: "StackMethod",
    weights: Optional[np.ndarray],
    sigma: float,
) -> np.ndarray:
    """Combine an (N, h, w) cube into an (h, w) 2-D tile.

    NaN entries are treated as non-coverage. `weights` is only used by the
    FWHM-weighted methods and must already be normalised to sum to 1.
    """
    if method == StackMethod.MEAN:
        return np.nanmean(cube_tile, axis=0)

    if method == StackMethod.MEDIAN:
        return np.nanmedian(cube_tile, axis=0)

    if method == StackMethod.SIGMA_CLIP:
        clipped = sigma_clip(cube_tile, sigma=sigma, axis=0, masked=True, stdfunc="mad_std")
        fallback = np.nanmean(cube_tile, axis=0)
        return np.where(clipped.mask.all(axis=0), fallback, np.ma.mean(clipped, axis=0).data)

    if method == StackMethod.FWHM_WEIGHTED:
        nan_mask = np.isnan(cube_tile)
        w = weights[:, None, None].astype(np.float32)
        contrib = (~nan_mask).astype(np.float32) * w
        numer = np.nansum(cube_tile * w, axis=0)
        denom = contrib.sum(axis=0)
        return np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)

    if method == StackMethod.SIGMA_CLIP_FWHM:
        clipped = sigma_clip(cube_tile, sigma=sigma, axis=0, masked=True, stdfunc="mad_std")
        valid = (~clipped.mask).astype(np.float32)
        w = weights[:, None, None].astype(np.float32) * valid
        safe = np.where(np.isnan(cube_tile), 0.0, cube_tile)
        numer = np.sum(safe * weights[:, None, None].astype(np.float32) * valid, axis=0)
        denom = w.sum(axis=0)
        fallback = np.nanmean(cube_tile, axis=0)
        return np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), fallback)

    raise ValueError(f"Unknown stack method: {method}")


def _fwhm_weights(fwhm_values: dict, accepted: list[Path]) -> np.ndarray:
    """Build per-frame 1/FWHM² weights, normalised to sum to 1.

    Frames with no FWHM measurement get the median weight of measured frames.
    """
    fwhms = np.array([fwhm_values.get(p, 0.0) for p in accepted], dtype=np.float64)
    zero_mask = fwhms == 0.0
    if zero_mask.any():
        measured = fwhms[~zero_mask]
        fallback_fwhm = float(np.median(measured)) if measured.size > 0 else 1.0
        fwhms[zero_mask] = fallback_fwhm
        _logger.warning(
            "%d frame(s) had no FWHM measurement; assigned median FWHM %.2f px",
            int(zero_mask.sum()), fallback_fwhm,
        )
    weights = 1.0 / (fwhms ** 2)
    weights /= weights.sum()
    return weights


def _select_by_quality(
    paths: list[Path],
    fwhm_values: dict[Path, float],
    star_counts: dict[Path, int],
    max_fwhm: Optional[float] = None,
    max_fwhm_multiplier: Optional[float] = None,
    min_star_fraction: Optional[float] = None,
) -> tuple[list[Path], list[Path], dict]:
    """The single frame-quality gate shared by every stacking path.

    A frame is rejected when, relative to the session medians of the *measured*
    frames, it is:
      * too blurry — FWHM > max_fwhm (or > max_fwhm_multiplier × median FWHM), or
      * too sparse — star count < min_star_fraction × median star count, or
      * unmeasurable — FWHM 0.0 while its peers measured fine (noise-dominated).

    Self-tunes to the night's seeing because thresholds are relative to the
    medians. It is a no-op (keeps every frame) when no frame has a measurable
    FWHM, so it can never reject everything on a host without photutils, and
    when no criterion is active (all thresholds None).

    Returns (accepted, rejected, summary). summary carries the resolved
    thresholds/medians for logging and progress messages.
    """
    measured_fwhm = np.array([v for v in fwhm_values.values() if v > 0.0])
    if measured_fwhm.size == 0:
        return list(paths), [], {"gated": False}

    median_fwhm = float(np.median(measured_fwhm))
    if max_fwhm is None and max_fwhm_multiplier:
        max_fwhm = median_fwhm * max_fwhm_multiplier

    measured_stars = np.array([s for s in star_counts.values() if s > 0], dtype=float)
    median_stars = float(np.median(measured_stars)) if measured_stars.size else None
    min_stars = (
        median_stars * min_star_fraction
        if (median_stars is not None and min_star_fraction) else None
    )

    arcsec = _get_arcsec_per_pixel()
    summary = {
        "gated": max_fwhm is not None or min_stars is not None,
        "median_fwhm": median_fwhm,
        "max_fwhm": max_fwhm,
        "median_stars": median_stars,
        "min_stars": min_stars,
        "arcsec": arcsec,
    }
    crit = []
    if max_fwhm is not None:
        crit.append(f'FWHM <= {max_fwhm * arcsec:.2f}"')
    if min_stars is not None:
        crit.append(f"stars >= {min_stars:.0f}")
    summary["crit_str"] = " & ".join(crit) if crit else "no cut"

    if not summary["gated"]:
        return list(paths), [], summary

    accepted: list[Path] = []
    rejected: list[Path] = []
    for p in paths:
        f = fwhm_values.get(p, 0.0)
        s = star_counts.get(p, 0)
        if f <= 0.0:
            rejected.append(p)
        elif max_fwhm is not None and f > max_fwhm:
            rejected.append(p)
        elif min_stars is not None and s < min_stars:
            rejected.append(p)
        else:
            accepted.append(p)
    return accepted, rejected, summary


# ---------------------------------------------------------------------------
# Core stacking
# ---------------------------------------------------------------------------

def stack(
    light_paths: list[Path],
    method: StackMethod = StackMethod.SIGMA_CLIP,
    bias_paths: Optional[list[Path]] = None,
    dark_paths: Optional[list[Path]] = None,
    flat_paths: Optional[list[Path]] = None,
    sigma: float = 3.0,
    max_fwhm: Optional[float] = None,
    max_fwhm_multiplier: Optional[float] = 1.25,
    min_star_fraction: Optional[float] = 0.5,
    register: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    precomputed_fwhm_stars: Optional[dict[Path, tuple[float, int]]] = None,
) -> tuple[np.ndarray, dict]:
    """
    Stack a list of FITS light frames with optional calibration and FWHM weighting.

    Args:
        light_paths:  Ordered list of light-frame FITS paths.
        method:       Stacking algorithm to use (default SIGMA_CLIP).
        bias_paths:   Bias calibration frames.
        dark_paths:   Dark calibration frames.
        flat_paths:   Flat calibration frames.
        sigma:        Rejection sigma for SIGMA_CLIP (default 3.0).
        max_fwhm:     Reject frames whose FWHM (pixels) exceeds this absolute
                      value. Overrides max_fwhm_multiplier when set.
        max_fwhm_multiplier:
                      Derive max_fwhm as multiplier × median(measured FWHM) when
                      max_fwhm is None (default 1.25). Lets the caller say "reject
                      frames worse than 1.25× the typical sub" without knowing the
                      seeing in advance. Pass None to disable the FWHM cut.
        min_star_fraction:
                      Reject frames with fewer than this × median(star count)
                      detected stars (default 0.5). Pass None to disable.
        precomputed_fwhm_stars:
                      Optional {path: (fwhm_px, star_count)} of already-measured
                      frames (e.g. from a `stats`/`bad` frame_stats.json cache),
                      so the detection pass is skipped for cached frames. Frames
                      absent from the map are measured normally. Defaults to None
                      (measure everything, unchanged behaviour).

    Frame selection goes through the shared _select_by_quality gate, the same
    one the convergence/snr analysis uses, so "good frames" means the same thing
    everywhere. Frames whose FWHM cannot be measured (while peers measured fine)
    are rejected as noise-dominated.

    Returns:
        (stacked_array, info_dict)

        info_dict keys:
            n_frames     – number of frames that entered the stack
            rejected     – list of Path objects rejected by the quality gate
            method       – name of the method used
            fwhm_values  – {str(path): fwhm_px} for every measured frame
    """
    if not light_paths:
        raise ValueError("No light frames provided")

    # Build calibration masters
    _logger.info("Building calibration masters…")
    master_bias = build_master_bias(bias_paths or [])
    master_dark = build_master_dark(dark_paths or [], master_bias)
    master_flat = build_master_flat(flat_paths or [], master_bias, master_dark)

    # Measure FWHM + star count (one detection pass) whenever a method needs
    # FWHM weights or any quality criterion is active. Both come from the same
    # _measure_fwhm_and_stars call, so star counts are free.
    need_fwhm = (
        method == StackMethod.FWHM_WEIGHTED
        or method == StackMethod.SIGMA_CLIP_FWHM
        or max_fwhm is not None
        or max_fwhm_multiplier is not None
        or min_star_fraction is not None
    )
    fwhm_values: dict[Path, float] = {}
    star_counts: dict[Path, int] = {}
    if need_fwhm:
        hits, misses = _split_cached(light_paths, precomputed_fwhm_stars)
        for p, (fw, sc) in hits.items():
            fwhm_values[p], star_counts[p] = fw, sc
        if hits:
            _logger.info("Reused %d/%d cached FWHM/star measurements",
                         len(hits), len(light_paths))
        if misses:
            _logger.info("Measuring FWHM + star counts for %d frames…", len(misses))
            if progress_cb:
                progress_cb(f"measuring FWHM + star counts for {len(misses)} frames…")
            for p in misses:
                _ckpt(cancel_cb)
                fwhm_values[p], star_counts[p] = _measure_fwhm_and_stars(p)
                _logger.debug("  %s → FWHM %.2f px, %d stars", p.name, fwhm_values[p], star_counts[p])

    # Shared quality gate — identical logic to the convergence/snr analysis.
    accepted, rejected, q = _select_by_quality(
        light_paths, fwhm_values, star_counts,
        max_fwhm=max_fwhm,
        max_fwhm_multiplier=max_fwhm_multiplier,
        min_star_fraction=min_star_fraction,
    )

    if not accepted:
        raise ValueError(
            f"All {len(light_paths)} frames were rejected by the quality gate "
            f"({q.get('crit_str', 'n/a')})"
        )

    if rejected:
        _logger.info(
            "Quality gate: rejected %d/%d frames (keep %s; median FWHM %.2f px, "
            "median stars %s)",
            len(rejected), len(light_paths), q["crit_str"], q["median_fwhm"],
            f'{q["median_stars"]:.0f}' if q.get("median_stars") is not None else "n/a",
        )
    if need_fwhm and progress_cb:
        if q.get("median_fwhm") is not None:
            progress_cb(
                f'quality gate — median {q["median_fwhm"] * q["arcsec"]:.2f}", '
                f"keeping {len(accepted)}/{len(light_paths)} frames"
                + (f", rejected {len(rejected)}" if rejected else "")
            )
        else:
            progress_cb(
                f"FWHM unmeasurable — keeping {len(accepted)}/{len(light_paths)} frames"
            )

    _logger.info(
        "Loading and calibrating %d frames (rejected %d)…", len(accepted), len(rejected)
    )
    _ckpt(cancel_cb)
    calibrated = [_calibrate(_load_fits_2d(p), master_bias, master_dark, master_flat)
                  for p in accepted]

    if register:
        _logger.info("Registering %d frames to reference…", len(calibrated))
        if progress_cb:
            progress_cb(f"registering {len(calibrated)} frames…")
        ref_fwhm = [fwhm_values.get(p, 0.0) for p in accepted] if fwhm_values else None
        calibrated, surviving_indices = _register_frames(calibrated, fwhm_scores=ref_fwhm)
        accepted = [accepted[i] for i in surviving_indices]
        _logger.info("%d frames remain after registration", len(calibrated))
        if progress_cb:
            n_dropped = len(light_paths) - len(accepted) - len(rejected)
            progress_cb(
                f"registration done — {len(calibrated)} frames aligned"
                + (f", {n_dropped} dropped" if n_dropped > 0 else "")
                + ", combining…"
            )

    n_frames = len(calibrated)
    if n_frames == 0:
        raise ValueError("All frames were dropped during registration")
    H, W = calibrated[0].shape

    # Stream each registered frame to a temp memmap on disk and free the
    # in-memory array, so we never hold all frames in RAM at once. A full
    # cube of e.g. 60 × 6388 × 9576 float32 is ~14 GiB and OOMs the typical
    # observatory machine.
    import shutil
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="srt_stack_"))

    weights = None
    if method in (StackMethod.FWHM_WEIGHTED, StackMethod.SIGMA_CLIP_FWHM):
        weights = _fwhm_weights(fwhm_values, accepted)
        _logger.info("FWHM weights: min=%.4f max=%.4f", weights.min(), weights.max())

    try:
        mmap_paths: list[Path] = []
        for i, frame in enumerate(calibrated):
            p = tmp_dir / f"f{i:04d}.npy"
            np.save(p, frame.astype(np.float32, copy=False))
            mmap_paths.append(p)
        calibrated = None  # release the in-memory list

        mmaps = [np.load(p, mmap_mode="r") for p in mmap_paths]

        result = np.empty((H, W), dtype=np.float32)
        coverage = np.zeros((H, W), dtype=np.int32)

        TILE = 512  # 512×512×N×4B ≈ 1 MiB per slice → ~60 MiB per tile cube at N=60
        n_ty = (H + TILE - 1) // TILE
        n_tx = (W + TILE - 1) // TILE
        n_tiles_total = n_ty * n_tx
        _logger.info(
            "Tiled combine: %d frames, %d×%d tiles of %d px, method %s",
            n_frames, n_ty, n_tx, TILE, method.name,
        )

        _progress_pcts_fired: set[int] = set()
        tile_idx = 0
        for y0 in range(0, H, TILE):
            y1 = min(y0 + TILE, H)
            for x0 in range(0, W, TILE):
                _ckpt(cancel_cb)
                x1 = min(x0 + TILE, W)
                cube_tile = np.stack(
                    [np.asarray(m[y0:y1, x0:x1]) for m in mmaps], axis=0,
                )
                coverage[y0:y1, x0:x1] = np.sum(~np.isnan(cube_tile), axis=0).astype(np.int32)
                result[y0:y1, x0:x1] = _combine_tile(cube_tile, method, weights, sigma)
                tile_idx += 1
                if progress_cb:
                    pct = tile_idx * 100 // n_tiles_total
                    milestone = (pct // 25) * 25
                    if milestone > 0 and milestone not in _progress_pcts_fired:
                        _progress_pcts_fired.add(milestone)
                        progress_cb(f"combining {milestone}%…")
    finally:
        # Drop memmaps before deleting the backing files (matters on Windows).
        mmaps = None
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Coverage-based crop: keep the bounding box where at least 80% of frames
    # contributed a real (non-NaN) pixel.
    min_cov = max(1, int(round(0.8 * n_frames)))
    well_covered = coverage >= min_cov
    if well_covered.any():
        rows = np.where(well_covered.any(axis=1))[0]
        cols = np.where(well_covered.any(axis=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        result = result[y0:y1, x0:x1]
        _logger.info(
            "Cropped stack to high-coverage region: %dx%d → %dx%d (>=%d/%d frames)",
            W, H, x1 - x0, y1 - y0, min_cov, n_frames,
        )

    # Any residual NaN in the cropped image is replaced with the global
    # sky median so DAOStarFinder and writers don't choke.
    nan_residual = np.isnan(result)
    if nan_residual.any():
        finite = result[~nan_residual]
        sky = float(np.median(finite)) if finite.size > 0 else 0.0
        result[nan_residual] = sky

    info = {
        "n_frames": len(accepted),
        "rejected": rejected,
        "method": method.name,
        "fwhm_values": {str(p): v for p, v in fwhm_values.items()},
    }
    return result.astype(np.float32), info


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

_UNKNOWN_FILTER = "UNKNOWN"


def _collect_fits(directory: Optional[Path]) -> list[Path]:
    if directory is None:
        return []
    return sorted(directory.rglob("*.fits")) + sorted(directory.rglob("*.fit"))


def read_filter(path: Path) -> str:
    """Return the FILTER header value from a FITS file, or 'UNKNOWN' if absent."""
    try:
        with fits.open(path) as hdul:
            return str(hdul[0].header.get("FILTER", _UNKNOWN_FILTER)).strip()
    except Exception:
        return _UNKNOWN_FILTER


def group_by_filter(paths: list[Path]) -> dict[str, list[Path]]:
    """
    Partition *paths* by their FITS FILTER header value.

    Returns a dict mapping filter name → list of paths, preserving the
    original order within each group.
    """
    groups: dict[str, list[Path]] = {}
    for p in paths:
        f = read_filter(p)
        groups.setdefault(f, []).append(p)
    return groups


def _filter_output_path(output_path: Path, filter_name: str) -> Path:
    """Insert the filter name before the file suffix: stack.fits → stack_Ha.fits."""
    safe = filter_name.replace(" ", "_")
    return output_path.with_name(f"{output_path.stem}_{safe}{output_path.suffix}")


def flat_dirs_from_root(root: Optional[Path]) -> tuple[Optional[Path], Optional[dict[str, Path]]]:
    """Map a single flats root directory onto (flat_dir, flat_dirs).

    Layout convention: ``root/<FILTER>/*.fits`` where each subdirectory is
    named exactly like the FITS ``FILTER`` header value (e.g. ``Ha``, ``OIII``,
    ``R``). With subdirectories present, a filter without its own subdirectory
    gets NO flats (the fallback dir is collected recursively, so offering the
    root would blend every filter's flats into one master). A root with no
    subdirectories is treated as one shared flat directory for all filters.
    This backs the ``calibration.flat_root`` config key.
    """
    if root is None or not root.is_dir():
        return None, None
    subdirs = {d.name: d for d in root.iterdir() if d.is_dir()}
    if subdirs:
        return None, subdirs
    return root, None


def _resolve_flat_paths(
    filter_name: str,
    flat_dir: Optional[Path],
    flat_dirs: Optional[dict[str, Path]],
) -> list[Path]:
    """Return flat paths for *filter_name*, preferring flat_dirs over flat_dir."""
    if flat_dirs and filter_name in flat_dirs:
        return _collect_fits(flat_dirs[filter_name])
    return _collect_fits(flat_dir)


def _write_stack(
    result: np.ndarray,
    header,
    output_path: Path,
    info: dict,
    max_fwhm: Optional[float],
) -> None:
    header["STACKMTH"] = (info["method"], "Stacking method")
    header["NFRAMES"] = (info["n_frames"], "Number of frames stacked")
    header.add_history(f"Stacked {info['n_frames']} frames with method {info['method']}")
    if info["rejected"]:
        header.add_history(f"Rejected {len(info['rejected'])} frame(s) by quality gate (FWHM/star count)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(str(output_path), result, header, overwrite=True)


def _save_jpg(data: np.ndarray, output_path: Path, title: str = "") -> Path:
    """Save a ZScale-stretched JPEG preview alongside the stacked FITS."""
    from astropy.visualization import ZScaleInterval
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    jpg_path = output_path.with_suffix(".jpg")
    vmin, vmax = ZScaleInterval().get_limits(data)
    fig = Figure(figsize=(10, 10))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.5)
    fig.savefig(jpg_path, format="jpeg", dpi=150, bbox_inches="tight")
    return jpg_path


# ---------------------------------------------------------------------------
# Convergence curve
# ---------------------------------------------------------------------------

def _fib_counts(max_n: int) -> list[int]:
    """Fibonacci-spaced frame counts from 1 up to max_n, always including max_n."""
    a, b = 1, 2
    counts = [1]
    while b < max_n:
        counts.append(b)
        a, b = b, a + b
    counts.append(max_n)
    return counts


def _prepare_for_convergence(
    paths: list[Path],
    max_fwhm_multiplier: float = 1.25,
    min_star_fraction: float = 0.5,
    register: bool = True,
    downscale_to: Optional[int] = 512,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    precomputed_fwhm_stars: Optional[dict[Path, tuple[float, int]]] = None,
) -> tuple[list[np.ndarray], list[Path], dict[Path, float]]:
    """
    Load, FWHM-filter, register, and downscale a set of FITS paths for convergence
    analysis.

    Uses *streaming* registration so the full-resolution cube never lives in RAM:
    the reference and one source frame are held at full resolution at a time;
    each aligned frame is immediately downscaled to ~downscale_to px before the
    next frame is loaded.  Peak memory is roughly two full-res frames plus the
    entire downscaled cube (a few hundred MB for a full-frame sensor).

    Pass downscale_to=None to keep frames at full resolution (e.g. for per-star
    aperture photometry).

    Returns (frames, accepted_paths, fwhm_values).
    fwhm_values maps every measured path; pass it with accepted_paths to
    _fwhm_weights() to get per-frame weights in registration-output order.
    """
    if not paths:
        raise ValueError("No paths provided")

    hits, misses = _split_cached(paths, precomputed_fwhm_stars)
    fwhm_values: dict[Path, float] = {}
    star_counts: dict[Path, int] = {}
    for p, (fw, sc) in hits.items():
        fwhm_values[p], star_counts[p] = fw, sc
    if hits:
        _logger.info("Reused %d/%d cached FWHM/star measurements", len(hits), len(paths))
    if misses:
        if progress_cb:
            progress_cb(f"measuring FWHM + star counts for {len(misses)} frames…")
        tick = max(1, len(misses) // 5)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_measure_fwhm_and_stars, p): p for p in misses}
            for i, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                fwhm_values[p], star_counts[p] = fut.result()
                if progress_cb and i % tick == 0:
                    progress_cb(f"FWHM + stars: {i}/{len(misses)} frames…")
    _ckpt(cancel_cb)

    # Shared quality gate (see _select_by_quality): reject blurry / sparse /
    # unmeasurable subs relative to the session median. Same logic the real
    # stack() uses, so the convergence analysis and the stacked output agree on
    # which frames are good.
    accepted, rejected, q = _select_by_quality(
        paths, fwhm_values, star_counts,
        max_fwhm_multiplier=max_fwhm_multiplier,
        min_star_fraction=min_star_fraction,
    )
    if not q.get("gated"):
        if progress_cb:
            progress_cb(f"FWHM unmeasurable — using all {len(accepted)} frames")
    elif rejected:
        _logger.info(
            "Convergence prep: rejected %d/%d frames (keep %s; median FWHM "
            "%.2f px, median stars %s)",
            len(rejected), len(paths), q["crit_str"], q["median_fwhm"],
            f'{q["median_stars"]:.0f}' if q["median_stars"] is not None else "n/a",
        )
        if progress_cb:
            progress_cb(
                f"quality cut — rejected {len(rejected)}/{len(paths)} "
                f"(keep {q['crit_str']}), {len(accepted)} remain"
            )
    elif progress_cb:
        progress_cb(
            f'quality cut — median {q["median_fwhm"] * q["arcsec"]:.2f}", '
            f"all {len(accepted)} frames kept"
        )

    if not accepted:
        raise ValueError("All frames rejected by FWHM/star-count threshold")

    if not register or not _REGISTER_AVAILABLE or len(accepted) < 2:
        if progress_cb:
            progress_cb(f"loading {len(accepted)} frames (no registration)…")
        frame0 = _load_fits_2d(accepted[0])
        scale = 1 if downscale_to is None else max(1, min(frame0.shape) // downscale_to)
        frames_out = [frame0[::scale, ::scale]]
        frame0 = None
        for p in accepted[1:]:
            f = _load_fits_2d(p)
            frames_out.append(f[::scale, ::scale])
            f = None
        return frames_out, accepted, fwhm_values

    # Pick the reference as the sharpest frame (lowest FWHM); FWHM is always
    # measured above. Falls back to source-count sampling only if no frame had a
    # measurable FWHM. (The old source-count-only pick selected the blurriest
    # frame on noisy channels and failed registration — see
    # _reference_index_by_fwhm.)
    actual_ref_idx = _reference_index_by_fwhm([fwhm_values.get(p, 0.0) for p in accepted])
    if actual_ref_idx is None:
        step = max(1, len(accepted) // 10)
        sample_indices = list(range(0, len(accepted), step))[:10]
        sample_frames = [_load_fits_2d(accepted[i]) for i in sample_indices]
        actual_ref_idx = sample_indices[_best_reference_idx(sample_frames)]
        sample_frames = None  # free full-res copies
    _logger.info("Convergence: reference = accepted[%d]", actual_ref_idx)

    if progress_cb:
        progress_cb(f"registering {len(accepted)} frames (streaming)…")

    reference = _load_fits_2d(accepted[actual_ref_idx])
    # Same despike-for-detection as _register_frames: hot pixels must not
    # provide the control points (they'd vote for the identity transform).
    reference_det = _despike(reference)
    scale = 1 if downscale_to is None else max(1, min(reference.shape) // downscale_to)

    result_frames: list[np.ndarray] = [reference[::scale, ::scale]]
    result_accepted: list[Path] = [accepted[actual_ref_idx]]
    failed, poor_qa = 0, 0

    for i, p in enumerate(accepted):
        if i == actual_ref_idx:
            continue
        _ckpt(cancel_cb)
        try:
            frame = _load_fits_2d(p)
            transform, (src_pts, dst_pts) = _astroalign.find_transform(
                _despike(frame), reference_det,
                max_control_points=_REG_MAX_CONTROL_POINTS,
            )
        except Exception as exc:
            failed += 1
            _logger.warning("Convergence registration failed for frame %d: %s", i, exc)
            continue

        n_matched = len(src_pts) if src_pts is not None else 0
        if n_matched < _REG_MIN_MATCHED_STARS:
            poor_qa += 1
            _logger.warning("Convergence: frame %d dropped — %d matched stars", i, n_matched)
            continue

        pts_tx = transform(np.asarray(src_pts))
        residuals = np.sqrt(np.sum((pts_tx - np.asarray(dst_pts)) ** 2, axis=1))
        if float(np.median(residuals)) > _REG_MAX_MEDIAN_RESIDUAL_PX:
            poor_qa += 1
            _logger.warning("Convergence: frame %d dropped — high residual", i)
            continue

        try:
            aligned, footprint = _astroalign.apply_transform(transform, frame, reference)
        except Exception as exc:
            failed += 1
            _logger.warning("Convergence: apply_transform failed for frame %d: %s", i, exc)
            continue

        frame = None  # free full-res copy
        aligned = aligned.astype(np.float32)
        aligned[footprint.astype(bool)] = np.nan
        result_frames.append(aligned[::scale, ::scale])
        result_accepted.append(p)

    reference = None  # free reference
    reference_det = None
    n_dropped = len(accepted) - len(result_frames)
    _logger.info(
        "Convergence prep: %d/%d frames registered, %d failed, %d dropped QA",
        len(result_frames), len(accepted), failed, poor_qa,
    )
    if progress_cb:
        progress_cb(
            f"registration done — {len(result_frames)} frames aligned"
            + (f", {n_dropped} dropped" if n_dropped > 0 else "")
            + ", building golden…"
        )

    return result_frames, result_accepted, fwhm_values


def convergence_curve(
    paths: list[Path],
    filter_name: str = "",
    output_path: Optional[Path] = None,
    golden_output_path: Optional[Path] = None,
    n_trials: int = 20,
    max_fwhm_multiplier: float = 1.25,
    min_star_fraction: float = 0.5,
    register: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    precomputed_fwhm_stars: Optional[dict[Path, tuple[float, int]]] = None,
) -> tuple[list[int], list[float], float]:
    """
    Measure how quickly stacking converges to the golden (all-frames) stack.

    Prepares frames via the same FWHM-filter + registration pipeline as stack(),
    then downscales to ~512 px for speed.  For each Fibonacci-spaced count k it
    draws n_trials random FWHM-weighted subsets and measures RMSE against the
    SIGMA_CLIP_FWHM golden stack.  RMSE is normalised by the golden stack's
    sigma-clipped median so the y-axis is dimensionless (0 = identical to golden).

    Args:
        paths:               LIGHT-frame FITS paths (caller already split by filter).
        filter_name:         Label used in the plot title.
        output_path:         If given, save the plot as a JPEG to this path.
        n_trials:            Random subsets to average per frame count (default 20).
        max_fwhm_multiplier: Reject frames whose FWHM exceeds this × median FWHM
                             (default 1.25). Unmeasured frames are also rejected.
        min_star_fraction:   Reject frames with fewer than this × median star
                             count (default 0.5).
        register:            Register frames before stacking (default True).
        progress_cb:         Called with a status string at each major milestone.

    Returns:
        (counts, mean_residuals, slope_pct, final_rmse_pct) — Fibonacci frame counts,
        mean normalised RMSE per point, tail slope in %/frame (negative = improving),
        and the RMSE at the last (all-frames) point as a percentage.
    """
    from astropy.stats import sigma_clipped_stats
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import FuncFormatter

    rng = np.random.default_rng()

    frames, accepted, fwhm_values = _prepare_for_convergence(
        paths,
        max_fwhm_multiplier=max_fwhm_multiplier,
        min_star_fraction=min_star_fraction,
        register=register,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        precomputed_fwhm_stars=precomputed_fwhm_stars,
    )

    n = len(frames)
    arr = np.stack(frames, axis=0)  # (N, H//scale, W//scale)
    weights = _fwhm_weights(fwhm_values, accepted)

    # Golden: SIGMA_CLIP_FWHM stack of all registered frames — same method as stack().
    golden = _combine_tile(arr, StackMethod.SIGMA_CLIP_FWHM, weights, sigma=3.0)
    nan_px = np.isnan(golden)
    if nan_px.any():
        golden[nan_px] = float(np.nanmedian(golden))

    _, golden_median, _ = sigma_clipped_stats(golden, sigma=3.0)
    if golden_median <= 0:
        golden_median = 1.0

    if golden_output_path is not None:
        golden_output_path.parent.mkdir(parents=True, exist_ok=True)
        _label = f"{filter_name} golden  {n} frames" if filter_name else f"golden  {n} frames"
        _save_jpg(golden, golden_output_path, title=_label)

    _ckpt(cancel_cb)
    if progress_cb:
        progress_cb(f"sampling convergence ({n} frames, {len(_fib_counts(n))} points)…")

    counts = _fib_counts(n)

    # Each Fibonacci count is fully independent — run them all concurrently.
    # Give each worker its own seeded RNG so there is no shared mutable state.
    seeds = rng.integers(0, 2**31, size=len(counts))

    def _trials_for_k(args: tuple[int, int]) -> tuple[float, float]:
        k, seed = args
        local_rng = np.random.default_rng(seed)
        actual_trials = n_trials if k < n else 1
        trial_rmses: list[float] = []
        for _ in range(actual_trials):
            idx = local_rng.choice(n, size=k, replace=False)
            sub_w = weights[idx]
            sub_w = sub_w / sub_w.sum()
            nan_mask = np.isnan(arr[idx])
            safe = np.where(nan_mask, 0.0, arr[idx])
            contrib = (~nan_mask).astype(np.float32) * sub_w[:, None, None]
            numer = np.sum(safe * sub_w[:, None, None], axis=0)
            denom = contrib.sum(axis=0)
            subset_stack = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), golden)
            trial_rmses.append(
                float(np.sqrt(np.nanmean((subset_stack - golden) ** 2))) / golden_median
            )
        return float(np.mean(trial_rmses)), float(np.std(trial_rmses))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_trials_for_k, zip(counts, seeds.tolist())))

    mean_residuals = [r[0] for r in results]
    std_residuals = [r[1] for r in results]

    xs = np.array(counts)
    ys = np.array(mean_residuals)
    errs = np.array(std_residuals)

    # Fit a line to the tail, excluding the final k=n point.  The last point is
    # computed with only 1 trial and uses a plain weighted mean against a
    # sigma-clipped golden, so it systematically undershoots and distorts the slope.
    tail_n = max(4, round(len(xs) * 0.4))
    xs_tail = xs[-tail_n:-1]
    ys_tail = ys[-tail_n:-1]
    tail_slope, tail_intercept = np.polyfit(xs_tail, ys_tail, 1)
    tail_fit = np.polyval([tail_slope, tail_intercept], xs_tail)
    slope_pct = tail_slope * 100  # convert fraction/frame → %/frame

    fig = Figure(figsize=(10, 5))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#0d0d1a")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.plot(xs, ys, "o-", color="#5fa8d3", label="Mean residual")
    ax.fill_between(xs, np.maximum(ys - errs, 0), ys + errs,
                    alpha=0.25, color="#5fa8d3", label="±1 σ  (trial spread)")
    ax.plot(xs_tail, tail_fit, "--", color="#ff6b5b", linewidth=1.5,
            label=f"Tail slope: {slope_pct:+.4f}% / frame")
    ax.set_xlabel("Frames stacked")
    ax.set_ylabel("Normalised RMSE  (vs golden stack)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs], rotation=45 if len(xs) > 8 else 0, ha="right")
    title = "Stack convergence vs golden"
    if filter_name:
        title += f"  [{filter_name}]"
    ax.set_title(title)
    legend = ax.legend()
    legend.get_frame().set_facecolor("#1a1a2e")
    legend.get_frame().set_edgecolor("#444466")
    for text in legend.get_texts():
        text.set_color("white")
    ax.grid(True, alpha=0.3, color="#444466")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    else:
        plt.show()

    # Use second-to-last point: "how much worse with ~40% fewer frames?"
    # This is a better flatness signal than [-1] which just measures method gap (weighted mean vs sigma clip).
    penult = mean_residuals[-2] if len(mean_residuals) >= 2 else mean_residuals[-1]
    final_rmse_pct = float(penult) * 100
    return counts, mean_residuals, slope_pct, final_rmse_pct


# ---------------------------------------------------------------------------
# Convenience: stack a directory
# ---------------------------------------------------------------------------

def stack_directory(
    directory: Path,
    output_path: Path,
    method: StackMethod = StackMethod.SIGMA_CLIP,
    bias_dir: Optional[Path] = None,
    dark_dir: Optional[Path] = None,
    flat_dir: Optional[Path] = None,
    flat_dirs: Optional[dict[str, Path]] = None,
    sigma: float = 3.0,
    max_fwhm: Optional[float] = None,
    register: bool = True,
    filters: Optional[list[str]] = None,
    snr_demo: bool = False,
) -> dict[str, tuple[Path, dict]]:
    """
    Stack all .fits/.fit files in *directory*, grouped by FILTER header.

    One output file is written per filter, named by inserting the filter before
    the suffix of *output_path* (e.g. stack.fits → stack_Ha.fits, stack_R.fits).
    If all frames share the same filter, output_path is used as-is.

    Args:
        directory:   Directory containing light frames.
        output_path: Output path template; filter name is inserted before the suffix.
        method:      Stacking algorithm (default SIGMA_CLIP).
        bias_dir:    Directory of bias frames (filter-agnostic, optional).
        dark_dir:    Directory of dark frames (filter-agnostic, optional).
        flat_dir:    Fallback flat directory used when flat_dirs has no entry for a filter.
        flat_dirs:   Per-filter flat directories, e.g. {"Ha": Path("flats/Ha"), "R": ...}.
        sigma:       Sigma rejection threshold for SIGMA_CLIP.
        max_fwhm:    Drop frames whose measured FWHM exceeds this value (pixels).

    Returns:
        dict mapping filter_name → (output_path, info_dict)
    """
    lights = _collect_fits(directory)
    if not lights:
        raise FileNotFoundError(f"No .fits/.fit files found in {directory}")

    groups = group_by_filter(lights)
    _logger.info(
        "Found %d light frames in %s across %d filter(s): %s",
        len(lights), directory, len(groups), list(groups.keys()),
    )

    bias_paths = _collect_fits(bias_dir)
    dark_paths = _collect_fits(dark_dir)
    results: dict[str, tuple[Path, dict]] = {}

    active_groups = {
        f: p for f, p in groups.items()
        if filters is None or f in filters
    }
    if not active_groups:
        raise ValueError(f"No frames match the requested filter(s): {filters}")

    for filter_name, group_paths in active_groups.items():
        flat_paths = _resolve_flat_paths(filter_name, flat_dir, flat_dirs)
        _logger.info("Stacking filter %s: %d frames", filter_name, len(group_paths))

        result, info = stack(
            light_paths=group_paths,
            method=method,
            bias_paths=bias_paths,
            dark_paths=dark_paths,
            flat_paths=flat_paths,
            sigma=sigma,
            max_fwhm=max_fwhm,
            register=register,
        )

        with fits.open(group_paths[0]) as hdul:
            header = hdul[0].header.copy()

        out = output_path if len(active_groups) == 1 else _filter_output_path(output_path, filter_name)
        _write_stack(result, header, out, info, max_fwhm)
        jpg = _save_jpg(result, out, title=f"{filter_name}  {info['n_frames']} frames  ({info['method']})")
        _logger.info("Wrote %s stack → %s  preview → %s", filter_name, out, jpg)
        results[filter_name] = (out, info)

        if snr_demo:
            conv_jpg = out.with_name(out.stem + "_convergence.jpg")
            convergence_curve(
                group_paths,
                filter_name=filter_name,
                output_path=conv_jpg,
                register=register,
            )
            _logger.info("Convergence curve → %s", conv_jpg)

    return results


# ---------------------------------------------------------------------------
# Live stacker
# ---------------------------------------------------------------------------

class LiveStacker:
    """
    Watches *watch_dir* for new .fits/.fit files and re-stacks whenever a new
    frame arrives, keeping each filter's stack separate.

    Frames are grouped by their FITS FILTER header. When a new frame arrives
    for filter "Ha", only the Ha stack is recomputed — other filters are left
    untouched. Output files are named by inserting the filter before the suffix
    of *output_path* (e.g. live.fits → live_Ha.fits, live_R.fits). If only one
    filter is ever seen, *output_path* is used as-is.

    Polling is used (default every 5 s) so no extra dependencies are required.

    Usage::

        stacker = LiveStacker(
            watch_dir=Path("/data/lights"),
            output_path=Path("/data/live.fits"),
            method=StackMethod.FWHM_WEIGHTED,
            flat_dirs={"Ha": Path("flats/Ha"), "R": Path("flats/R")},
            on_stack=lambda path, info: print(f"{info['filter']} stack: {info['n_frames']} frames → {path}"),
        )
        stacker.start()
        # … later …
        stacker.stop()
    """

    _POLL_INTERVAL = 5.0  # seconds

    def __init__(
        self,
        watch_dir: Path,
        output_path: Path,
        method: StackMethod = StackMethod.SIGMA_CLIP,
        bias_dir: Optional[Path] = None,
        dark_dir: Optional[Path] = None,
        flat_dir: Optional[Path] = None,
        flat_dirs: Optional[dict[str, Path]] = None,
        sigma: float = 3.0,
        max_fwhm: Optional[float] = None,
        on_stack: Optional[Callable[[Path, dict], None]] = None,
        poll_interval: float = _POLL_INTERVAL,
    ):
        self.watch_dir = watch_dir
        self.output_path = output_path
        self.method = method
        self.bias_dir = bias_dir
        self.dark_dir = dark_dir
        self.flat_dir = flat_dir
        self.flat_dirs = flat_dirs
        self.sigma = sigma
        self.max_fwhm = max_fwhm
        self.on_stack = on_stack
        self._poll_interval = poll_interval

        # filter_name → set of known paths for that filter
        self._known_by_filter: dict[str, set[Path]] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="LiveStacker"
        )
        self._thread.start()
        _logger.info("LiveStacker started, watching %s", self.watch_dir)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._poll_interval * 2)
        _logger.info("LiveStacker stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _scan_by_filter(self) -> dict[str, set[Path]]:
        """Return {filter_name: {path, ...}} for all FITS in watch_dir."""
        result: dict[str, set[Path]] = {}
        for p in _collect_fits(self.watch_dir):
            f = read_filter(p)
            result.setdefault(f, set()).add(p)
        return result

    def _poll_loop(self) -> None:
        self._known_by_filter = self._scan_by_filter()
        if self._known_by_filter:
            total = sum(len(v) for v in self._known_by_filter.values())
            _logger.info(
                "LiveStacker: %d existing frame(s) across filter(s) %s — initial stack",
                total, list(self._known_by_filter.keys()),
            )
            for filter_name in self._known_by_filter:
                self._do_stack(filter_name)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._poll_interval)
            current_by_filter = self._scan_by_filter()

            # Find which filters have new frames
            dirty: set[str] = set()
            for filter_name, current_paths in current_by_filter.items():
                known_paths = self._known_by_filter.get(filter_name, set())
                new_paths = current_paths - known_paths
                if new_paths:
                    _logger.info(
                        "LiveStacker: %d new frame(s) for filter %s: %s",
                        len(new_paths), filter_name,
                        [p.name for p in sorted(new_paths)],
                    )
                    dirty.add(filter_name)

            if dirty:
                self._known_by_filter = current_by_filter
                for filter_name in dirty:
                    self._do_stack(filter_name)

    def _do_stack(self, filter_name: str) -> None:
        lights = sorted(self._known_by_filter.get(filter_name, set()))
        if not lights:
            return

        # Use output_path directly if this is the only known filter, else suffix it
        if len(self._known_by_filter) == 1:
            out = self.output_path
        else:
            out = _filter_output_path(self.output_path, filter_name)

        try:
            flat_paths = _resolve_flat_paths(filter_name, self.flat_dir, self.flat_dirs)
            result, info = stack(
                light_paths=lights,
                method=self.method,
                bias_paths=_collect_fits(self.bias_dir),
                dark_paths=_collect_fits(self.dark_dir),
                flat_paths=flat_paths,
                sigma=self.sigma,
                max_fwhm=self.max_fwhm,
            )
            info["filter"] = filter_name

            with fits.open(lights[0]) as hdul:
                header = hdul[0].header.copy()
            header.add_history(f"Live-stacked {info['n_frames']} frames")
            _write_stack(result, header, out, info, self.max_fwhm)
            _save_jpg(result, out, title=f"{filter_name}  {info['n_frames']} frames  ({info['method']})")

            _logger.info(
                "LiveStacker [%s]: wrote %s (%d frames, %d rejected)",
                filter_name, out, info["n_frames"], len(info["rejected"]),
            )
            if self.on_stack:
                self.on_stack(out, info)

        except Exception:
            _logger.exception("LiveStacker [%s]: stack failed", filter_name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Stack FITS light frames")
    parser.add_argument("directory", type=Path, help="Directory of light frames")
    parser.add_argument("output", type=Path, help="Output FITS path")
    parser.add_argument(
        "--method",
        choices=[m.name for m in StackMethod],
        default="SIGMA_CLIP",
        help="Stacking method (default: SIGMA_CLIP)",
    )
    parser.add_argument("--bias-dir",  type=Path, default=None, help="Bias frame directory (filter-agnostic)")
    parser.add_argument("--dark-dir",  type=Path, default=None, help="Dark frame directory (filter-agnostic)")
    parser.add_argument("--flat-dir",  type=Path, default=None, help="Flat directory fallback (used when --flat-dirs has no entry for a filter)")
    parser.add_argument(
        "--flat-dirs", nargs="+", metavar="FILTER=DIR", default=[],
        help="Per-filter flat directories, e.g. Ha=/flats/Ha R=/flats/R",
    )
    parser.add_argument("--sigma",       type=float, default=3.0,  help="Sigma for SIGMA_CLIP")
    parser.add_argument("--max-fwhm",   type=float, default=None, help="Reject frames above this FWHM (pixels)")
    parser.add_argument("--no-register", action="store_true",     help="Disable star-based frame registration")
    parser.add_argument("--filters",    nargs="+",  default=None, help="Only process these filter(s), e.g. --filters Ha R")
    parser.add_argument("--snr-demo",   action="store_true",      help="Plot SNR vs frame count for each filter")
    parser.add_argument(
        "--live", action="store_true",
        help="Watch directory for new frames and restack on each arrival",
    )
    args = parser.parse_args()

    method = StackMethod[args.method]

    flat_dirs: Optional[dict[str, Path]] = None
    if args.flat_dirs:
        flat_dirs = {}
        for entry in args.flat_dirs:
            if "=" not in entry:
                parser.error(f"--flat-dirs entries must be FILTER=DIR, got: {entry!r}")
            filt, _, d = entry.partition("=")
            flat_dirs[filt.strip()] = Path(d.strip())

    if args.live:
        def _on_stack(path: Path, info: dict) -> None:
            filt = info.get("filter", "?")
            print(
                f"[live/{filt}] {info['n_frames']} frames → {path}"
                + (f"  (rejected {len(info['rejected'])})" if info["rejected"] else "")
            )

        ls = LiveStacker(
            watch_dir=args.directory,
            output_path=args.output,
            method=method,
            bias_dir=args.bias_dir,
            dark_dir=args.dark_dir,
            flat_dir=args.flat_dir,
            flat_dirs=flat_dirs,
            sigma=args.sigma,
            max_fwhm=args.max_fwhm,
            on_stack=_on_stack,
        )
        ls.start()
        print(f"Watching {args.directory} — press Ctrl-C to stop")
        try:
            while ls.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            ls.stop()
    else:
        results = stack_directory(
            directory=args.directory,
            output_path=args.output,
            method=method,
            bias_dir=args.bias_dir,
            dark_dir=args.dark_dir,
            flat_dir=args.flat_dir,
            flat_dirs=flat_dirs,
            sigma=args.sigma,
            max_fwhm=args.max_fwhm,
            register=not args.no_register,
            filters=args.filters,
            snr_demo=args.snr_demo,
        )
        for filt, (out, info) in results.items():
            print(f"[{filt}] Stacked {info['n_frames']} frames → {out}")
            if info["rejected"]:
                print(f"  Rejected {len(info['rejected'])} frame(s):")
                for p in info["rejected"]:
                    print(f"    {p.name}")
            if info["fwhm_values"]:
                vals = [v for v in info["fwhm_values"].values() if v > 0]
                if vals:
                    app = _get_arcsec_per_pixel()
                    print(f'  FWHM range: {min(vals)*app:.2f} – {max(vals)*app:.2f}"  (mean {np.mean(vals)*app:.2f}")')
