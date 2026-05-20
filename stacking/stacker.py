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
    """Count detected stars in a frame using sep (astroalign dependency)."""
    try:
        import sep
        data = frame.astype(float)
        bkg = sep.Background(data)
        sources = sep.extract(data - bkg.back(), thresh=3.0, err=bkg.rms())
        return len(sources)
    except Exception:
        return 0


def _best_reference_idx(frames: list[np.ndarray]) -> int:
    """
    Sample up to 10 evenly-spaced frames and return the index of the one
    with the most detected sources — this gives astroalign the best chance
    of successfully aligning all other frames.
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


def _register_frames(
    frames: list[np.ndarray],
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

    ref_idx = _best_reference_idx(frames)
    reference = frames[ref_idx]
    registered = [reference]
    surviving_indices = [ref_idx]
    failed = 0
    poor_qa = 0
    for i, frame in enumerate(frames):
        if i == ref_idx:
            continue
        try:
            transform, (src_pts, dst_pts) = _astroalign.find_transform(frame, reference)
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
    if not _FWHM_AVAILABLE:
        return 0.0
    try:
        fwhm_px, _, count, _ = _calculate_fwhm(path, arcsec_per_pixel=_get_arcsec_per_pixel())
        if count > 0 and fwhm_px > 0:
            return float(fwhm_px)
    except Exception as exc:
        _logger.warning("FWHM measurement failed for %s: %s", path.name, exc)
    return 0.0


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
    max_fwhm_multiplier: Optional[float] = None,
    register: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
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
        max_fwhm:     Reject frames whose FWHM (pixels) exceeds this value.
                      Frames where FWHM cannot be measured are kept.
        max_fwhm_multiplier:
                      If set and max_fwhm is None, derive max_fwhm as
                      multiplier × median(measured FWHM). Lets the caller say
                      "reject frames worse than 1.5× the typical sub" without
                      knowing the seeing in advance.

    Returns:
        (stacked_array, info_dict)

        info_dict keys:
            n_frames     – number of frames that entered the stack
            rejected     – list of Path objects that were rejected by max_fwhm
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

    # Measure FWHM when needed
    need_fwhm = (
        method == StackMethod.FWHM_WEIGHTED
        or method == StackMethod.SIGMA_CLIP_FWHM
        or max_fwhm is not None
        or max_fwhm_multiplier is not None
    )
    fwhm_values: dict[Path, float] = {}
    if need_fwhm:
        _logger.info("Measuring FWHM for %d frames…", len(light_paths))
        if progress_cb:
            progress_cb(f"measuring FWHM for {len(light_paths)} frames…")
        for p in light_paths:
            fwhm_values[p] = _measure_fwhm(p)
            _logger.debug("  %s → FWHM %.2f px", p.name, fwhm_values[p])

    if max_fwhm is None and max_fwhm_multiplier is not None:
        measured = np.array([v for v in fwhm_values.values() if v > 0.0])
        if measured.size > 0:
            median_fwhm = float(np.median(measured))
            max_fwhm = median_fwhm * max_fwhm_multiplier
            _logger.info(
                "max_fwhm auto-derived: median %.2f px × %.2f = %.2f px",
                median_fwhm, max_fwhm_multiplier, max_fwhm,
            )

    # Reject frames that exceed max_fwhm (unmeasured frames are kept)
    rejected: list[Path] = []
    accepted: list[Path] = []
    for p in light_paths:
        fwhm = fwhm_values.get(p, 0.0)
        if max_fwhm is not None and fwhm > 0.0 and fwhm > max_fwhm:
            rejected.append(p)
            _logger.info("Rejected %s (FWHM %.2f px > max %.2f px)", p.name, fwhm, max_fwhm)
        else:
            accepted.append(p)

    if not accepted:
        raise ValueError(
            f"All {len(light_paths)} frames were rejected by max_fwhm={max_fwhm}"
        )

    if need_fwhm and progress_cb:
        measured_vals = [v for v in fwhm_values.values() if v > 0.0]
        arcsec = _get_arcsec_per_pixel()
        if measured_vals:
            median_px = float(np.median(measured_vals))
            progress_cb(
                f"FWHM done — median {median_px * arcsec:.2f}″, "
                f"keeping {len(accepted)}/{len(light_paths)} frames"
                + (f", rejected {len(rejected)} blurry" if rejected else "")
            )
        else:
            progress_cb(
                f"FWHM unmeasurable — keeping {len(accepted)}/{len(light_paths)} frames"
            )

    _logger.info(
        "Loading and calibrating %d frames (rejected %d)…", len(accepted), len(rejected)
    )
    calibrated = [_calibrate(_load_fits_2d(p), master_bias, master_dark, master_flat)
                  for p in accepted]

    if register:
        _logger.info("Registering %d frames to reference…", len(calibrated))
        if progress_cb:
            progress_cb(f"registering {len(calibrated)} frames…")
        calibrated, surviving_indices = _register_frames(calibrated)
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
        header.add_history(f"Rejected {len(info['rejected'])} frame(s) via max_fwhm={max_fwhm}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(str(output_path), result, header, overwrite=True)


def _save_jpg(data: np.ndarray, output_path: Path, title: str = "") -> Path:
    """Save a ZScale-stretched JPEG preview alongside the stacked FITS."""
    from astropy.visualization import ZScaleInterval
    import matplotlib.pyplot as plt

    jpg_path = output_path.with_suffix(".jpg")
    vmin, vmax = ZScaleInterval().get_limits(data)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.5)
    fig.savefig(jpg_path, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)
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


def convergence_curve(
    frames: list[np.ndarray],
    filter_name: str = "",
    output_path: Optional[Path] = None,
    n_trials: int = 20,
) -> tuple[list[int], list[float], float]:
    """
    Measure how quickly stacking converges to the golden (all-frames) stack.

    For each Fibonacci-spaced count k, draws n_trials random subsets of k frames,
    mean-stacks each, and computes RMSE against the golden stack.  RMSE is
    normalised by the golden stack's sigma-clipped median so the y-axis is a
    dimensionless fraction (0 = identical to golden).

    Args:
        frames:      Calibrated 2-D arrays (any order — subsets are drawn randomly).
        filter_name: Label used in the plot title.
        output_path: If given, save the plot as a JPEG to this path.
        n_trials:    Random subsets to average per frame count (default 20).

    Returns:
        (counts, mean_residuals, slope_pct) — Fibonacci frame counts, mean normalised RMSE,
        and tail slope in %/frame (negative means improving).
    """
    from astropy.stats import sigma_clipped_stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()
    n = len(frames)
    h, w = frames[0].shape[:2]
    scale = max(1, min(h, w) // 512)
    if scale > 1:
        frames = [f[::scale, ::scale] for f in frames]
    arr = np.stack(frames, axis=0)  # (N, H//scale, W//scale)

    golden = arr.mean(axis=0)
    _, golden_median, _ = sigma_clipped_stats(golden, sigma=3.0)
    if golden_median <= 0:
        golden_median = 1.0

    counts = _fib_counts(n)
    mean_residuals: list[float] = []
    std_residuals: list[float] = []

    for k in counts:
        actual_trials = n_trials if k < n else 1
        trial_rmses: list[float] = []
        for _ in range(actual_trials):
            idx = rng.choice(n, size=k, replace=False)
            subset_stack = arr[idx].mean(axis=0)
            rmse = float(np.sqrt(np.mean((subset_stack - golden) ** 2)))
            trial_rmses.append(rmse / golden_median)
        mean_residuals.append(float(np.mean(trial_rmses)))
        std_residuals.append(float(np.std(trial_rmses)))

    xs = np.array(counts)
    ys = np.array(mean_residuals)
    errs = np.array(std_residuals)

    # Fit a line to the tail (last ~40% of points, min 3) to quantify the linear decline.
    tail_n = max(3, round(len(xs) * 0.4))
    xs_tail = xs[-tail_n:]
    ys_tail = ys[-tail_n:]
    tail_slope, tail_intercept = np.polyfit(xs_tail, ys_tail, 1)
    tail_fit = np.polyval([tail_slope, tail_intercept], xs_tail)
    slope_pct = tail_slope * 100  # convert fraction/frame → %/frame

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d0d1a")
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
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
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
        plt.close(fig)
    else:
        plt.show()

    return counts, mean_residuals, slope_pct


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
            master_bias = build_master_bias(bias_paths)
            master_dark = build_master_dark(dark_paths, master_bias)
            master_flat = build_master_flat(flat_paths, master_bias, master_dark)
            calibrated = [_calibrate(_load_fits_2d(p), master_bias, master_dark, master_flat)
                          for p in group_paths]
            if register:
                calibrated, _ = _register_frames(calibrated)
            convergence_curve(calibrated, filter_name=filter_name, output_path=conv_jpg)
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
