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
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from typing import Callable, NamedTuple, Optional

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


# Rows per band when median-combining calibration frames. 50 bias frames at
# 61 MP is 12 GB as one cube; 400 rows at a time is under 1 GB and gives a
# bit-identical median.
_MASTER_BAND_ROWS = 400


def _build_master(paths: list[Path]) -> Optional[np.ndarray]:
    """Median-combine a list of FITS frames into a master frame.

    Reads a horizontal band from every frame at a time rather than loading the
    whole cube — a full set of 61 MP calibration frames does not fit in RAM.
    """
    if not paths:
        return None
    with fits.open(paths[0]) as hdul:
        shape = np.squeeze(hdul[0].data).shape
    if len(shape) != 2:
        raise ValueError(f"Expected 2-D calibration frame, got {shape} in {paths[0]}")

    # Read each band straight off disk rather than through hdul.data, which
    # materialises the whole frame every time it is touched: banding 100 flats
    # that way costs 16 x 100 full-frame reads (~195 GB of I/O) instead of one
    # pass. These files carry BZERO/BSCALE, which rules out astropy's memmap
    # unless scaling is deferred — so defer it and apply the scale by hand.
    master = np.empty(shape, dtype=np.float32)
    for y0 in range(0, shape[0], _MASTER_BAND_ROWS):
        y1 = min(y0 + _MASTER_BAND_ROWS, shape[0])
        band = np.empty((len(paths), y1 - y0, shape[1]), dtype=np.float32)
        for i, p in enumerate(paths):
            with fits.open(p, memmap=True, do_not_scale_image_data=True) as hdul:
                hdu = hdul[0]
                raw = np.squeeze(hdu.data[..., y0:y1, :])
                if raw.shape != (y1 - y0, shape[1]):
                    raise ValueError(
                        f"Calibration frame shape mismatch in {p}: got {raw.shape}")
                band[i] = (np.asarray(raw, dtype=np.float32)
                           * float(hdu.header.get("BSCALE", 1))
                           + float(hdu.header.get("BZERO", 0)))
        master[y0:y1] = np.median(band, axis=0)
        band = None
    return master


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


class CalibrationSet(NamedTuple):
    """Masters on disk as .npy, plus the exposure the dark was taken at.

    Paths rather than arrays: the registration workers are separate processes,
    and a 61 MP master is 245 MB. Pickling bias+dark to each worker would cost
    1.5 GB of IPC per filter and hold a private copy in every process, where
    np.load(mmap_mode="r") lets them all share one set of OS page-cache pages.
    """
    bias_npy: Optional[Path]
    dark_npy: Optional[Path]
    dark_exptime: Optional[float]
    # Flats are per-filter, so this is filled in per channel by the caller
    # (see build_master_flat_npy) rather than by load_calibration_set.
    flat_npy: Optional[Path] = None


def _master_cache_path(paths: list[Path], tag: str) -> Path:
    """Scratch path keyed by the exact input frames and their mtimes."""
    import hashlib
    key = "|".join(f"{p}:{p.stat().st_mtime_ns}" for p in sorted(paths))
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from configs import config
        scratch = root / config.data()["scratch"]["directory"]
    except Exception:
        scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch / f"master_{tag}_{digest}.npy"


def _cached_master(paths: list[Path], tag: str, subtract: Optional[np.ndarray] = None) -> Path:
    """Build (or reuse) a master for *paths* and return its .npy path.

    Building costs ~1 s/frame, so the result is cached against the input files'
    mtimes: the masters only change when you shoot new calibration frames.
    """
    out = _master_cache_path(paths, tag)
    if out.exists():
        return out
    _logger.info("Building master %s from %d frames…", tag, len(paths))
    master = _build_master(paths)
    if master is None:
        raise ValueError(f"No frames to build master {tag}")
    if subtract is not None:
        master = master - subtract
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, master.astype(np.float32))
    tmp.replace(out)
    return out


def load_calibration_set(
    bias_paths: list[Path],
    dark_paths: list[Path],
) -> Optional[CalibrationSet]:
    """Build/reuse master bias and (bias-subtracted) master dark on disk.

    Darks are grouped by exposure and the largest group wins, so a directory
    holding a stray 60 s dark among 300 s ones does not poison the master.
    Returns None when there is nothing to calibrate with.
    """
    if not bias_paths and not dark_paths:
        return None

    bias_npy = _cached_master(bias_paths, "bias") if bias_paths else None
    bias_arr = np.load(bias_npy, mmap_mode="r") if bias_npy else None

    dark_npy = dark_exptime = None
    if dark_paths:
        by_exp: dict[float, list[Path]] = {}
        for p in dark_paths:
            try:
                exp = round(float(fits.getheader(p).get("EXPTIME", 0.0)), 2)
            except Exception:
                continue
            by_exp.setdefault(exp, []).append(p)
        if by_exp:
            dark_exptime, chosen = max(by_exp.items(), key=lambda kv: len(kv[1]))
            if len(by_exp) > 1:
                _logger.info(
                    "Master dark: using %d frames at %.1fs (ignoring %s)",
                    len(chosen), dark_exptime,
                    ", ".join(f"{len(v)}x{k:.0f}s" for k, v in by_exp.items()
                              if k != dark_exptime),
                )
            dark_npy = _cached_master(chosen, f"dark{dark_exptime:g}", subtract=bias_arr)
    bias_arr = None
    return CalibrationSet(bias_npy, dark_npy, dark_exptime)


def results_dir(dso_name: str) -> Path:
    """Where finished products go: ``<image_dir>/Iris/<dso>/``.

    Beside the lights and calibration frames rather than in scratch, so a
    target's stacks and colour renders travel with its data instead of living in
    a directory that gets cleared. Nothing under Iris/ can be mistaken for a
    light frame — every path that collects lights requires the parent directory
    to be named LIGHT — so writing FITS here cannot contaminate a frame scan.
    """
    from configs import config
    out = Path(config.data()["nina"]["image_dir"]) / "Iris" / dso_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def calibration_paths_from_config(
    filter_name: Optional[str] = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """(bias, dark, flat) frame paths from the ``calibration`` config block.

    Flats are only returned when *filter_name* is given, since they are
    per-filter, and they are matched by FITS header rather than by directory
    layout (see flats_by_filter).
    """
    from configs import config
    cal = config.data().get("calibration", {}) or {}
    bias = _collect_fits(Path(cal["bias_dir"])) if cal.get("bias_dir") else []
    dark = _collect_fits(Path(cal["dark_dir"])) if cal.get("dark_dir") else []
    flat: list[Path] = []
    if filter_name and cal.get("flat_root"):
        flat = flats_by_filter(Path(cal["flat_root"])).get(filter_name, [])
    return bias, dark, flat


def calibration_from_config(filter_name: Optional[str] = None) -> Optional[CalibrationSet]:
    """Build the masters named by ``calibration`` config, or None.

    Pass *filter_name* to get that filter's flat attached as well; without it
    the set is bias+dark only, because a flat is meaningless across filters.
    """
    try:
        bias, dark, flat = calibration_paths_from_config(filter_name)
        if not bias and not dark:
            return None
        cal = load_calibration_set(bias, dark)
        if cal is not None and flat:
            npy = build_master_flat_npy(flat, cal, (filter_name or "").replace("-", ""))
            if npy is not None:
                cal = cal._replace(flat_npy=npy)
        return cal
    except Exception:
        _logger.exception("Could not build calibration masters — using raw frames")
        return None


def _dark_exptime(dark_paths) -> Optional[float]:
    """Exposure of the dominant exposure group in *dark_paths*."""
    groups: dict[float, int] = {}
    for p in dark_paths or []:
        try:
            e = round(float(fits.getheader(p).get("EXPTIME", 0.0)), 2)
        except Exception:
            continue
        groups[e] = groups.get(e, 0) + 1
    return max(groups, key=groups.get) if groups else None


def build_master_flat_npy(flat_paths: list[Path], cal: CalibrationSet,
                          tag: str) -> Optional[Path]:
    """Bias/dark-corrected, mean-normalised master flat, cached as .npy.

    Flats are exposed long enough for dark current to matter (47.5 s on Ha here,
    against 300 s darks), so the master dark is scaled by the exposure ratio
    rather than ignored. The result is floored at 0.05 so dividing by it can
    never blow a pixel up by more than 20x, and so _apply_calibration needs no
    zero-guard on the hot path.
    """
    if not flat_paths:
        return None
    out = _master_cache_path(flat_paths, f"flat_{tag}")
    if out.exists():
        return out
    _logger.info("Building master flat %s from %d frames…", tag, len(flat_paths))
    master = _build_master(flat_paths)
    if master is None:
        return None
    try:
        exptime = float(fits.getheader(flat_paths[0]).get("EXPTIME", 0.0))
    except Exception:
        exptime = 0.0
    if cal.bias_npy is not None:
        master = master - np.asarray(np.load(cal.bias_npy, mmap_mode="r"), np.float32)
    if cal.dark_npy is not None and cal.dark_exptime and exptime:
        master = master - (np.asarray(np.load(cal.dark_npy, mmap_mode="r"), np.float32)
                           * (exptime / cal.dark_exptime))
    mean = float(np.mean(master))
    if not np.isfinite(mean) or mean == 0:
        return None
    master = np.clip(master / mean, 0.05, None).astype(np.float32)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, master)
    tmp.replace(out)
    return out


def _apply_calibration(frame: np.ndarray, cal: Optional[CalibrationSet],
                       exptime: Optional[float]) -> np.ndarray:
    """Subtract master bias and exposure-scaled master dark, in place."""
    if cal is None:
        return frame
    if cal.bias_npy is not None:
        frame -= np.load(cal.bias_npy, mmap_mode="r")
    if cal.dark_npy is not None:
        dark = np.load(cal.dark_npy, mmap_mode="r")
        # Dark current is linear in exposure, and the master is bias-subtracted,
        # so a light shot at a different length scales cleanly.
        if exptime and cal.dark_exptime and abs(exptime - cal.dark_exptime) > 1e-6:
            frame -= np.asarray(dark, dtype=np.float32) * (exptime / cal.dark_exptime)
        else:
            frame -= dark
    if cal.flat_npy is not None:
        # (raw - bias - dark) / flat, in that order. The master is normalised to
        # mean 1 and floored away from zero at build time, so this needs no
        # guard and no temporary the size of the frame.
        frame /= np.load(cal.flat_npy, mmap_mode="r")
    return frame


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
    ~4 s → ~0.1 s on 61 MP frames), then OpenCV, then scipy.

    The OpenCV path matters: this is called once per frame inside the
    registration loop, and scipy's generic n-d median_filter takes ~5.8 s on a
    61 MP (6388×9576) frame — about 40% of the whole per-frame registration
    cost. cv2.medianBlur is the same 3×3 median, specialised for 2-D, and runs
    it in ~0.08 s — 70x faster and bit-identical (verified on real sh2-92 Ha
    subs: np.array_equal(scipy, cv2) is True). It only accepts uint8/uint16/
    float32 at ksize=3, so anything else falls through to scipy.

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
    if frame.ndim == 2 and frame.dtype in (np.uint8, np.uint16, np.float32):
        try:
            import cv2
            return cv2.medianBlur(np.ascontiguousarray(frame), 3)
        except Exception:
            _logger.debug("cv2 despike unavailable, falling back to scipy", exc_info=True)
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


def _reference_control_points(reference_det: np.ndarray) -> Optional[np.ndarray]:
    """Return the reference's (N, 2) control points, or None if unavailable.

    ``astroalign.find_transform(source, target)`` runs its own source extraction
    on *both* arguments every call, so registering N frames against one reference
    re-detects the reference's stars N times — 1.5 s of the 3.8 s per-frame
    find_transform on a 61 MP frame, repeated for nothing. find_transform also
    accepts an (N, 2) array of source positions in place of an image, so extract
    the reference's points once and pass those instead. Verified on sh2-92 Ha
    subs: the returned transform matrix is bit-identical either way.

    Returns None if astroalign's private extractor moves or finds too few
    sources; callers then pass the reference image itself, as before — same
    result, just slower.
    """
    try:
        pts = np.asarray(_astroalign._find_sources(reference_det))[:_REG_MAX_CONTROL_POINTS]
        if len(pts) >= 3:
            return pts
    except Exception:
        _logger.debug("Reference source pre-extraction unavailable", exc_info=True)
    return None


def _apply_affine(transform, frame: np.ndarray, ref_shape: tuple[int, int]):
    """Warp *frame* onto a *ref_shape* grid. Returns (aligned float32, footprint).

    Does what ``astroalign.apply_transform`` does, minus its two avoidable costs.
    On a 61 MP frame that call takes 4.1 s — about half the per-frame
    registration budget — and only 2.4 s of it is the warp anyone wanted:

      * 1.15 s warping a second, all-zero image just to learn which output
        pixels fell outside the source. That is pure geometry, and cv2 gets the
        same mask in 0.09 s (1758 differing pixels out of 61 M, all in a 1-px
        edge sliver — cv2 and skimage are bit-identical on integer shifts).
      * 0.52 s taking the median of the whole frame for the out-of-bounds fill
        value. Every pixel it fills is about to be masked anyway; a ::8
        subsample gives the same number to 1 ADU in 0.01 s.

    The data warp itself is left exactly as astroalign does it — same order-3
    spline, same clip and preserve_range — because the convergence RMSE this
    feeds is sensitive to the resampling kernel. Swapping in cv2's bicubic
    (sharpening) moved the whole curve +8%, and its bilinear (smoothing) -13%,
    which would quietly shift every stored slope in convergence.json and the
    `done` decisions made from them. 1.6x is worth more than 34x here.
    """
    try:
        from stacking import gpu_accel
        res = gpu_accel.apply_affine(np.asarray(transform.params), frame, ref_shape)
        if res is not None:
            return res
    except Exception:
        _logger.exception("GPU warp failed — falling back to CPU")

    try:
        from skimage.transform import warp

        aligned = warp(
            frame, inverse_map=transform.inverse, output_shape=ref_shape, order=3,
            mode="constant", cval=float(np.median(frame[::8, ::8])),
            clip=True, preserve_range=True,
        )
        footprint = _warp_footprint(transform, frame.shape, ref_shape)
        return aligned, footprint
    except Exception:
        _logger.debug("Fast warp unavailable, falling back to astroalign", exc_info=True)

    # apply_transform only reads the target's shape, so an unwritten array is
    # enough and costs nothing.
    return _astroalign.apply_transform(transform, frame, np.empty(ref_shape, dtype=np.float32))


def _warp_footprint(transform, src_shape: tuple[int, int], ref_shape: tuple[int, int]):
    """True where the warped output pulled from outside the source frame.

    Matches astroalign's ``warp(zeros, cval=1.0) > 0.4`` test, computed on a
    uint8 mask with cv2 when available (0.09 s vs 1.15 s).
    """
    h, w = ref_shape
    try:
        import cv2
        covered = cv2.warpAffine(
            np.ones(src_shape, dtype=np.uint8),
            np.asarray(transform.params, dtype=np.float64)[:2, :], (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return covered < 0.6
    except Exception:
        _logger.debug("cv2 footprint unavailable, using skimage", exc_info=True)
    from skimage.transform import warp
    return warp(np.zeros(src_shape, dtype="float32"), inverse_map=transform.inverse,
                output_shape=ref_shape, cval=1.0) > 0.4


def _load_calibrated(path: Path, cal: Optional["CalibrationSet"]) -> np.ndarray:
    """Load a light frame and remove its bias + dark pedestal."""
    frame = _load_fits_2d(path)
    if cal is None:
        return frame
    try:
        exptime = float(fits.getheader(path).get("EXPTIME", 0.0)) or None
    except Exception:
        exptime = None
    return _apply_calibration(frame, cal, exptime)


def _register_one(
    path: Path,
    det_target: np.ndarray,
    ref_shape: tuple[int, int],
    scale: int,
    cal: Optional["CalibrationSet"] = None,
) -> tuple[Optional[np.ndarray], str]:
    """Register one frame against *det_target*, downscaled by *scale*.

    Returns (downscaled aligned frame, "") on success, or (None, reason) — the
    reason is prefixed "QA:" when the frame was rejected rather than errored.
    Runs in a worker process (see _registration_pool), so it takes a path rather
    than an array, reports outcomes by return value rather than by logging, and
    never touches shared state.

    *det_target* is the reference's control points (see
    _reference_control_points) or, in the serial fallback, the reference image.
    """
    frame = None
    try:
        frame = _load_calibrated(path, cal)
    except Exception as exc:
        return None, f"load failed: {exc}"
    aligned, reason = _register_one_array(frame, det_target, ref_shape)
    frame = None  # free full-res copy
    if aligned is None:
        return None, reason
    return _downsample_mean(aligned, scale), ""


def _register_one_array(frame: np.ndarray, det_target: np.ndarray,
                        ref_shape: tuple[int, int]) -> tuple[Optional[np.ndarray], str]:
    """Register one already-loaded, already-calibrated frame onto *ref_shape*.

    The QA rules live here so the streaming path in stack() and the pooled path
    in _prepare_for_convergence cannot drift apart on what counts as a good
    registration.
    """
    try:
        transform, (src_pts, dst_pts) = _astroalign.find_transform(
            _despike(frame), det_target,
            max_control_points=_REG_MAX_CONTROL_POINTS,
        )
    except Exception as exc:
        return None, f"registration failed: {exc}"

    n_matched = len(src_pts) if src_pts is not None else 0
    if n_matched < _REG_MIN_MATCHED_STARS:
        return None, f"QA: dropped — {n_matched} matched stars"

    pts_tx = transform(np.asarray(src_pts))
    residuals = np.sqrt(np.sum((pts_tx - np.asarray(dst_pts)) ** 2, axis=1))
    if float(np.median(residuals)) > _REG_MAX_MEDIAN_RESIDUAL_PX:
        return None, "QA: dropped — high residual"

    try:
        aligned, footprint = _apply_affine(transform, frame, ref_shape)
    except Exception as exc:
        return None, f"apply_transform failed: {exc}"

    aligned = aligned.astype(np.float32)
    aligned[footprint.astype(bool)] = np.nan
    return aligned, ""


# Workers per registration pool. `snr` already runs its filters in parallel
# threads, so the real budget is workers x filters, and RAM binds before cores
# do: a worker peaks near 1 GB, because skimage's warp upcasts the 61 MP frame
# to float64. Measured peak RSS on the usual two-filter night was 9.7 GB at 4
# workers each; 3 holds it to 8.0 GB, which leaves room for N.I.N.A on this
# 29 GB box and costs only 9% wall-clock (347 s -> 378 s).
_REG_POOL_WORKERS = 3


def _reg_worker_init() -> None:
    """Detach a spawned worker from the app's log file.

    Windows spawn makes each worker re-import the parent's __main__ — in
    production that is start_srt.py, which pulls in the whole app and with it a
    FileHandler on iris.log. Three workers appending to the one log file is
    exactly what the logging docs warn against, and they have nothing to say
    anyway: _register_one reports outcomes through its return value.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.NullHandler())


@contextmanager
def _registration_pool(enabled: bool):
    """Yield a process pool for one prep call's registration, or None.

    Created and torn down per call so the worker processes' full-resolution
    frame buffers are returned to the OS between runs rather than sitting on an
    idle long-lived pool — this module lives inside an always-on server.
    """
    if not enabled:
        yield None
        return
    try:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=_REG_POOL_WORKERS,
                                   initializer=_reg_worker_init)
    except Exception:
        _logger.exception("Could not start registration pool — registering serially")
        yield None
        return
    try:
        yield pool
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


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
    ref_pts = _reference_control_points(reference_det)
    if ref_pts is not None:
        reference_det = ref_pts
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
            aligned, footprint = _apply_affine(transform, frame, reference.shape)
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

def cleanup_stale_spill_dirs(min_age_hours: float = 2.0) -> int:
    """Delete leftover srt_stack_* scratch directories. Returns bytes freed.

    A stack spills one calibrated float32 frame per sub to temp — ~245 MB each,
    tens of GB per run — and cleans up in a finally. A killed worker never
    reaches that finally, so the scratch survives (26.9 GB from one such run).
    Called at server start, where anything older than a couple of hours cannot
    belong to a live stack.
    """
    import shutil
    import tempfile
    import time as _time
    cutoff = _time.time() - min_age_hours * 3600
    freed = 0
    try:
        root = Path(tempfile.gettempdir())
        for d in root.glob("srt_stack_*"):
            try:
                if not d.is_dir() or d.stat().st_mtime > cutoff:
                    continue
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
                if not d.exists():
                    freed += size
            except OSError:
                # Usually delete-pending: a killed worker's rmtree ran while its
                # memmaps were still open, so the (now empty) directory entry
                # lingers and even listdir returns access-denied. Nothing to
                # reclaim there — skip it and leave it to the OS.
                continue
    except Exception:
        _logger.exception("stale spill cleanup failed")
    if freed:
        _logger.info("Reclaimed %.1f GB of stale stack scratch", freed / 1e9)
    return freed


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
    shared_reference: Optional[tuple[np.ndarray, tuple[int, int]]] = None,
) -> tuple[np.ndarray, dict]:
    """
    Stack a list of FITS light frames with optional calibration and FWHM weighting.

    shared_reference is (control_points, shape) from
    _reference_control_points. Pass it to register these frames onto someone
    else's grid instead of picking a reference from this set — which is how a
    multi-filter colour process keeps its channels on the same pixels. Without
    it each filter would align to its own reference and the channels would sit
    apart by however far the mount drifted between them.

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
    # Masters go through the scratch .npy cache keyed on the calibration frames'
    # mtimes. Rebuilding them from the raw frames every run costs ~170 frame
    # reads here (50 bias + 20 dark + 100 flats) — minutes of identical work for
    # files that only change when new calibration frames are shot.
    _logger.info("Building calibration masters…")
    if progress_cb:
        progress_cb("calibration masters…")
    master_bias = master_dark = master_flat = None
    if bias_paths:
        master_bias = np.load(_cached_master(list(bias_paths), "bias"), mmap_mode="r")
    if dark_paths:
        master_dark = np.load(
            _cached_master(list(dark_paths), "dark", subtract=master_bias), mmap_mode="r")
    if flat_paths:
        _cal = CalibrationSet(
            _master_cache_path(list(bias_paths), "bias") if bias_paths else None,
            _master_cache_path(list(dark_paths), "dark") if dark_paths else None,
            _dark_exptime(dark_paths) if dark_paths else None,
        )
        _flat_npy = build_master_flat_npy(list(flat_paths), _cal, "stack")
        if _flat_npy is not None:
            master_flat = np.load(_flat_npy, mmap_mode="r")

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

    # Load, calibrate, register and spill each frame one at a time. Building the
    # calibrated list first — as this used to — needs every frame in RAM at once:
    # 118 sh2-92 subs at 61 MP float32 is ~29 GB, so full resolution was not
    # actually reachable on this box. Peak is now two frames plus the tile cube.
    import shutil
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="srt_stack_"))
    mmap_paths: list[Path] = []
    survivors: list[Path] = []
    failed = poor_qa = 0

    # Sky levels vary hugely between frames — moon, transparency, altitude. On
    # ngc5907 the G subs ran 149 to 442 ADU across nights. Combining them at
    # their native levels turns that spread into per-pixel NOISE: sigma-clip
    # rejects a different subset of frames at every pixel, and because those
    # frames sit at different levels the surviving mean jumps pixel to pixel.
    # Measured there: 8.04x the photon limit at 39 frames, and worse than a
    # single frame. It gets worse with MORE frames, because more frames means
    # more nights and a wider spread, which is why it hides in small tests.
    #
    # So each frame is levelled to a common sky before combining, and the level
    # is restored afterwards so the output keeps a physical ADU scale.
    frame_levels: list[float] = []

    def _spill(frame: np.ndarray, path: Path) -> Path:
        level = float(np.nanmedian(frame))
        if np.isfinite(level):
            frame = frame - level
            frame_levels.append(level)
        else:
            frame_levels.append(0.0)
        p = tmp_dir / f"f{len(mmap_paths):04d}.npy"
        np.save(p, frame.astype(np.float32, copy=False))
        mmap_paths.append(p)
        survivors.append(path)
        return p

    try:
        do_register = register and _REGISTER_AVAILABLE and (
            len(accepted) >= 2 or shared_reference is not None)
        if not do_register:
            if progress_cb:
                progress_cb(f"loading {len(accepted)} frames (no registration)…")
            for p in accepted:
                _ckpt(cancel_cb)
                _spill(_calibrate(_load_fits_2d(p), master_bias, master_dark,
                                  master_flat), p)
        else:
            if shared_reference is not None:
                det_target, ref_shape = shared_reference
                ref_idx = -1          # the reference is external; stack every frame
                _logger.info("Registering %d frames to a shared reference…", len(accepted))
            else:
                ref_idx = _reference_index_by_fwhm(
                    [fwhm_values.get(p, 0.0) for p in accepted])
                if ref_idx is None:
                    ref_idx = 0
                _logger.info("Registering %d frames to accepted[%d]…", len(accepted), ref_idx)
                reference = _calibrate(_load_fits_2d(accepted[ref_idx]), master_bias,
                                       master_dark, master_flat)
                ref_shape = reference.shape
                det_target = _reference_control_points(_despike(reference))
                if det_target is None:
                    det_target = _despike(reference)
                _spill(reference, accepted[ref_idx])
                reference = None
            if progress_cb:
                progress_cb(f"registering {len(accepted)} frames (streaming)…")

            tick = max(1, len(accepted) // 10)
            for i, p in enumerate(accepted):
                if i == ref_idx:
                    continue
                _ckpt(cancel_cb)
                frame = _calibrate(_load_fits_2d(p), master_bias, master_dark, master_flat)
                aligned, reason = _register_one_array(frame, det_target, ref_shape)
                frame = None
                if aligned is None:
                    if reason.startswith("QA:"):
                        poor_qa += 1
                    else:
                        failed += 1
                    _logger.warning("Registration: frame %d — %s", i, reason)
                    continue
                _spill(aligned, p)
                aligned = None
                if progress_cb and i % tick == 0:
                    progress_cb(f"registered {len(mmap_paths)}/{len(accepted)}…")

            _logger.info("Registration: %d/%d aligned, %d failed, %d dropped QA",
                         len(mmap_paths), len(accepted), failed, poor_qa)
            if progress_cb:
                progress_cb(f"registration done — {len(mmap_paths)} frames aligned"
                            + (f", {failed + poor_qa} dropped" if failed + poor_qa else "")
                            + ", combining…")

        accepted = survivors
        n_frames = len(mmap_paths)
        if n_frames == 0:
            raise ValueError("All frames were dropped during registration")

        weights = None
        if method in (StackMethod.FWHM_WEIGHTED, StackMethod.SIGMA_CLIP_FWHM):
            weights = _fwhm_weights(fwhm_values, accepted)
            _logger.info("FWHM weights: min=%.4f max=%.4f", weights.min(), weights.max())

        mmaps = [np.load(p, mmap_mode="r") for p in mmap_paths]
        H, W = mmaps[0].shape

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

    # Put the sky back: the frames were levelled before combining (see _spill),
    # so add the weighted mean of what was removed. Without this the stack would
    # sit near zero and every downstream sky/ADU measurement would be wrong.
    if frame_levels:
        lv = np.asarray(frame_levels, dtype=np.float64)
        restored = float(np.average(lv, weights=weights) if weights is not None
                         else lv.mean())
        result = result + restored
        _logger.info("Frame levelling: sky spread %.1f..%.1f ADU, restored %.1f",
                     lv.min(), lv.max(), restored)

    # Coverage-based crop: keep the bounding box where at least 80% of frames
    # contributed a real (non-NaN) pixel.
    #
    # Skipped entirely when the caller supplied a shared reference. That box
    # depends on this filter's own dither pattern and frame count, so cropping
    # per filter leaves each channel starting at a different place on the sky —
    # which is exactly what a shared reference exists to prevent. On abell2151
    # it split every galaxy into separate red, green and blue blobs. The caller
    # aligning several stacks is responsible for its own common crop.
    if shared_reference is not None:
        _logger.info("Shared reference: skipping per-filter coverage crop")
        min_cov = 0
    else:
        min_cov = max(1, int(round(0.8 * n_frames)))
    well_covered = coverage >= min_cov
    if min_cov and well_covered.any():
        rows = np.where(well_covered.any(axis=1))[0]
        cols = np.where(well_covered.any(axis=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        result = result[y0:y1, x0:x1]
        _logger.info(
            # ASCII arrow: the console handler on this box is cp1252 and raises
            # UnicodeEncodeError on U+2192, which prints a logging traceback.
            "Cropped stack to high-coverage region: %dx%d -> %dx%d (>=%d/%d frames)",
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


_FLATS_BY_FILTER_CACHE: dict[str, dict[str, list[Path]]] = {}


def flats_by_filter(root: Optional[Path]) -> dict[str, list[Path]]:
    """Collect every flat under *root* and group by its FITS FILTER header.

    Cached per root for the life of the process: this opens a header for every
    flat on disk (~1000 of them here), and it is called once per filter by
    calibration_from_config. New flats appear between runs, not during one.

    flat_dirs_from_root assumes root/<FILTER>/*.fits, but N.I.N.A writes one FLAT
    directory per session with all filters mixed together
    (cdk17/2026-07-25/FLAT/...). Reading the header instead works for both
    layouts and picks up flats spread across many nights, which is what this
    observatory actually has.
    """
    out: dict[str, list[Path]] = {}
    if root is None or not root.is_dir():
        return out
    key = str(root)
    if key in _FLATS_BY_FILTER_CACHE:
        return _FLATS_BY_FILTER_CACHE[key]
    for f in sorted(root.rglob("*.fit*")):
        if f.parent.name.upper() != "FLAT":
            continue
        try:
            filt = str(fits.getheader(f).get("FILTER", "")).strip()
        except Exception:
            continue
        if filt:
            out.setdefault(filt, []).append(f)
    _FLATS_BY_FILTER_CACHE[key] = out
    return out


def _resolve_flat_paths(
    filter_name: str,
    flat_dir: Optional[Path],
    flat_dirs: Optional[dict[str, Path]],
    by_header: Optional[dict[str, list[Path]]] = None,
) -> list[Path]:
    """Return flat paths for *filter_name*, preferring flat_dirs over flat_dir.

    *by_header* (from flats_by_filter) wins when it has this filter — it is the
    only one of the three that copes with per-session FLAT directories.
    """
    if by_header and by_header.get(filter_name):
        return by_header[filter_name]
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


def save_plain_jpg(data: np.ndarray, output_path: Path) -> Path:
    """Save a ZScale-stretched JPEG with no title, axes, border or resampling.

    One output pixel per data pixel — unlike _save_jpg, which draws through a
    fixed-size matplotlib figure and so downsamples a 61 MP stack to whatever
    fits in 10 inches at 150 dpi. Written with PIL rather than matplotlib
    precisely so nothing can add furniture to the image.

    Row order matches the FITS convention used everywhere else here (origin
    lower), so the result is oriented like the annotated preview.
    """
    from astropy.visualization import ZScaleInterval
    from PIL import Image

    vmin, vmax = ZScaleInterval().get_limits(data)
    span = float(vmax - vmin) or 1.0
    scaled = np.clip((np.asarray(data, dtype=np.float32) - vmin) / span, 0.0, 1.0)
    img = (scaled * 255.0).astype(np.uint8)[::-1]        # flip to origin="lower"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path, quality=92, optimize=True)
    return output_path


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


def _downsample_mean(frame: np.ndarray, scale: int) -> np.ndarray:
    """Block-average ``frame`` by an integer ``scale`` (NaN-aware).

    Replaces stride decimation (``frame[::scale, ::scale]``). Nearest-neighbour
    striding aliases the sensor's fixed-pattern texture (hot pixels, amp glow,
    column structure) into visible moiré in the downscaled golden/convergence
    preview; block-averaging anti-aliases and beats the noise down instead. Edge
    pixels that don't fill a whole block are cropped; all-NaN blocks (the
    registration footprint border) stay NaN.
    """
    if scale <= 1:
        return frame
    h, w = frame.shape
    h2, w2 = h - h % scale, w - w % scale
    sub = np.ascontiguousarray(frame[:h2, :w2], dtype=np.float32)
    blocks = sub.reshape(h2 // scale, scale, w2 // scale, scale)
    finite = np.isfinite(blocks)
    summed = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    count = finite.sum(axis=(1, 3))
    return np.divide(
        summed, count,
        out=np.full(summed.shape, np.nan, dtype=np.float32),
        where=count > 0,
    )


def _prepare_for_convergence(
    paths: list[Path],
    max_fwhm_multiplier: float = 1.25,
    min_star_fraction: float = 0.5,
    register: bool = True,
    downscale_to: Optional[int] = 512,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    precomputed_fwhm_stars: Optional[dict[Path, tuple[float, int]]] = None,
    calibration: Optional[CalibrationSet] = None,
) -> tuple[list[np.ndarray], list[Path], dict[Path, float]]:
    """
    Load, FWHM-filter, register, and downscale a set of FITS paths for convergence
    analysis.

    With *calibration*, every frame has its master bias and exposure-scaled
    master dark subtracted before registration, so what comes out is sky signal
    rather than sky-on-a-pedestal. That matters more than it sounds: on this
    camera the bias sits at 151 ADU and an Ha sub's sky is ~1.5 ADU, so
    uncalibrated frames are 99% pedestal.

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
        frame0 = _load_calibrated(accepted[0], calibration)
        scale = 1 if downscale_to is None else max(1, min(frame0.shape) // downscale_to)
        frames_out = [_downsample_mean(frame0, scale)]
        frame0 = None
        for p in accepted[1:]:
            f = _load_calibrated(p, calibration)
            frames_out.append(_downsample_mean(f, scale))
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
        sample_frames = [_load_calibrated(accepted[i], calibration) for i in sample_indices]
        actual_ref_idx = sample_indices[_best_reference_idx(sample_frames)]
        sample_frames = None  # free full-res copies
    _logger.info("Convergence: reference = accepted[%d]", actual_ref_idx)

    if progress_cb:
        progress_cb(f"registering {len(accepted)} frames (streaming)…")

    reference = _load_calibrated(accepted[actual_ref_idx], calibration)
    # Same despike-for-detection as _register_frames: hot pixels must not
    # provide the control points (they'd vote for the identity transform).
    # Extracted to a point list once so find_transform doesn't redo it per frame.
    reference_det = _despike(reference)
    ref_pts = _reference_control_points(reference_det)
    scale = 1 if downscale_to is None else max(1, min(reference.shape) // downscale_to)
    ref_shape = reference.shape

    result_frames: list[np.ndarray] = [_downsample_mean(reference, scale)]
    result_accepted: list[Path] = [accepted[actual_ref_idx]]
    failed, poor_qa = 0, 0

    todo = [(i, p) for i, p in enumerate(accepted) if i != actual_ref_idx]

    # Registering one frame is ~3 s of pure CPU (source detection, triangle
    # matching, warp) and holds the GIL throughout — sep and skimage are C, but
    # neither releases it, so threads give ~1.15x on 4 workers. Processes do
    # scale: each worker loads its own frame from disk and sends back only the
    # downscaled result (~2 MB), so there is nothing large to pickle. The full-
    # resolution path (transit photometry, downscale_to=None) stays serial —
    # there the results are 245 MB each and pickling them would cost more than
    # the registration saves.
    use_pool = ref_pts is not None and downscale_to is not None and len(todo) > 2
    det_target = reference_det if ref_pts is None else ref_pts

    def _serial_results():
        for i, p in todo:
            yield i, _register_one(p, det_target, ref_shape, scale, calibration)

    def _pooled_results(pool):
        # Read back in submission order so result_frames keeps the same ordering
        # as the serial loop; anything raised here cancels the rest.
        futures = [(i, pool.submit(_register_one, p, ref_pts, ref_shape, scale, calibration))
                   for i, p in todo]
        try:
            for i, fut in futures:
                yield i, fut.result()
        except BaseException:
            for _, fut in futures:
                fut.cancel()
            raise

    def _consume(results):
        frames, paths, failed, poor_qa = [], [], 0, 0
        for i, (small, reason) in results:
            _ckpt(cancel_cb)
            if small is None:
                if reason.startswith("QA:"):
                    poor_qa += 1
                else:
                    failed += 1
                _logger.warning("Convergence: frame %d — %s", i, reason)
                continue
            frames.append(small)
            paths.append(accepted[i])
        return frames, paths, failed, poor_qa

    reference = None  # free full-res reference; only its shape is needed now
    reference_det = None if ref_pts is not None else reference_det

    from concurrent.futures.process import BrokenProcessPool

    with _registration_pool(use_pool) as pool:
        if pool is None:
            registered, paths, failed, poor_qa = _consume(_serial_results())
        else:
            try:
                registered, paths, failed, poor_qa = _consume(_pooled_results(pool))
            except Cancelled:
                raise
            except BrokenProcessPool:
                # A worker died (OOM, or a bad FITS taking the interpreter down).
                # Don't lose the night's analysis over it — redo it serially.
                _logger.exception("Registration pool broke — falling back to serial")
                registered, paths, failed, poor_qa = _consume(_serial_results())

    result_frames.extend(registered)
    result_accepted.extend(paths)

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
    calibration: Optional[CalibrationSet] = None,
) -> tuple[list[int], list[float], float]:
    """
    Measure how quickly stacking converges to the golden (all-frames) stack.

    Prepares frames via the same FWHM-filter + registration pipeline as stack(),
    then downscales to ~512 px for speed.  For each Fibonacci-spaced count k it
    draws n_trials random FWHM-weighted subsets and measures RMSE against the
    SIGMA_CLIP_FWHM golden stack.  RMSE is normalised by the golden stack's
    sigma-clipped median so the y-axis is dimensionless (0 = identical to golden).

    That normalisation is only meaningful with *calibration*. Uncalibrated, the
    golden's median is the bias pedestal (151 ADU) plus the sky (~1.5 ADU on Ha),
    so the y-axis reads as a percentage of a number that is 99% pedestal and the
    curve is compressed by ~100x. Calibrated, it is a percentage of sky signal,
    and the thresholds in the convergence config are set against that scale.

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
        calibration=calibration,
    )

    n = len(frames)
    arr = np.stack(frames, axis=0)  # (N, H//scale, W//scale)
    weights = _fwhm_weights(fwhm_values, accepted)

    # Golden: SIGMA_CLIP_FWHM stack of all registered frames — same method as stack().
    golden = _combine_tile(arr, StackMethod.SIGMA_CLIP_FWHM, weights, sigma=3.0)
    nan_px = np.isnan(golden)
    if nan_px.any():
        golden[nan_px] = float(np.nanmedian(golden))

    # Denominator = sky signal once calibration has removed the pedestal. It can
    # legitimately land near zero on a dark narrowband night, and dividing by
    # that would report a meaningless three-digit RMSE, so fall back to the
    # stack's own noise — the curve stays comparable within the run, and the
    # warning says the absolute scale is not.
    _, golden_median, golden_std = sigma_clipped_stats(golden, sigma=3.0)
    if golden_median <= 0 or (golden_std > 0 and golden_median < golden_std):
        _logger.warning(
            "Convergence [%s]: sky level %.3f ADU is at or below the stack noise "
            "%.3f ADU — normalising by noise instead; RMSE%% is not comparable "
            "to other targets", filter_name or "?", golden_median, golden_std,
        )
        golden_median = float(golden_std) if golden_std > 0 else 1.0
    _logger.info("Convergence [%s]: sky %.3f ADU, stack noise %.3f ADU (%d frames)",
                 filter_name or "?", golden_median, golden_std, n)

    if golden_output_path is not None:
        golden_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Say both counts on the image itself: n is what went into this stack,
        # len(paths) is what was on disk before the quality cut and registration.
        _count = f"{n} of {len(paths)} frames" if n != len(paths) else f"{n} frames"
        _label = f"{filter_name} golden  {_count}" if filter_name else f"golden  {_count}"
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
            # Combine subsets exactly as the golden is combined. Using a plain
            # weighted mean here instead put a constant floor under the whole
            # curve — the k=n point compares all the frames against all the same
            # frames, so anything it reports is pure method difference, and that
            # was 16.5% of sky on Ha and 21.2% on O-III. The floor flattened the
            # tail and swamped the slope: O-III's slope varied 2.1x between runs
            # on identical data. Same combine both sides means k=n is 0 by
            # construction and the curve measures frame count alone.
            subset_stack = _combine_tile(arr[idx], StackMethod.SIGMA_CLIP_FWHM,
                                         sub_w, sigma=3.0)
            subset_stack = np.where(np.isnan(subset_stack), golden, subset_stack)
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

    # Fit a line to the tail, excluding the final k=n point — that point compares
    # every frame against every frame by the same method, so it is identically
    # zero and carries no information about frame count.
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
    # [-1] is zero by construction (see above), so it says nothing.
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
