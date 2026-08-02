"""Convergence persistence — compute, store, and query per-DSO tail slopes.

JSON structure (local/convergence.json):
{
  "m31": {
    "Ha": {"tail_slope_pct": 0.23, "frame_count": 45, "total_frames": 52,
           "updated": "2026-05-04"},
    "R":  {"tail_slope_pct": 0.89, "frame_count": 12, "total_frames": 12,
           "updated": "2026-05-04"}
  }
}

frame_count is the number of frames actually stacked into the golden (post
quality cut and registration) — the N the slope describes; total_frames is every
LIGHT frame on disk for that filter. Entries written before 2026-08-01 have no
total_frames, and their frame_count is the total rather than the stacked count.

tail_slope_pct is negative (RMSE is falling), so "done" means abs(slope) < threshold.
"""

import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config as _config

_logger = logging.getLogger(__name__)


def _conv_path() -> Path:
    cfg = _config.data()
    rel = cfg.get("convergence", {}).get("file", "local/convergence.json")
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return root / rel


def _threshold() -> float:
    cfg = _config.data()
    return float(cfg.get("convergence", {}).get("tail_slope_threshold", 0.40))


def _min_frames() -> int:
    cfg = _config.data()
    return int(cfg.get("convergence", {}).get("min_frames_per_filter", 30))


def _rmse_threshold() -> float:
    cfg = _config.data()
    return float(cfg.get("convergence", {}).get("rmse_done_threshold", 5.0))


def load_convergence() -> dict:
    path = _conv_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        _logger.exception("Failed to load convergence.json")
        return {}


def save_convergence(dso_name: str, results: dict) -> None:
    """Merge filter results for dso_name into convergence.json (atomic write)."""
    path = _conv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_convergence()
    key = dso_name.lower().replace(" ", "")
    existing = data.get(key, {})
    existing.update(results)
    data[key] = existing
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except Exception:
        _logger.exception("Failed to save convergence.json")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def compute_dso_convergence(dso_name: str, image_dir: Path) -> dict[str, dict]:
    """Return {filter: {tail_slope_pct, frame_count, total_frames, updated}} per filter.

    Reuses the same FITS-loading / filter-grouping logic as _snr_run().
    Returns an empty dict if no LIGHT frames are found.
    """
    from stacking import stacker

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir = _find_dso_dir_by_name(dso_name)
    if dso_dir is None:
        _logger.warning("compute_dso_convergence: no directory for '%s'", dso_name)
        return {}

    fits_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if _is_light(f)),
        key=lambda f: f.stat().st_mtime,
    )
    if not fits_files:
        return {}

    by_filter = stacker.group_by_filter(fits_files)

    results: dict[str, dict] = {}
    today = date.today().isoformat()
    # Resolved per filter inside the loop so each gets its own flat.
    for fname, paths in by_filter.items():
        try:
            calibration = stacker.calibration_from_config(fname)
            counts, _, slope_pct, final_rmse_pct = stacker.convergence_curve(
                paths, filter_name=fname, calibration=calibration)
            # counts[-1] is the all-frames point of the curve: the frames that
            # survived the quality cut and registered, i.e. the ones the slope
            # was actually measured on. len(paths) is every LIGHT frame on disk.
            results[fname] = {
                "tail_slope_pct": round(slope_pct, 6),
                "final_rmse_pct": round(final_rmse_pct, 4),
                "frame_count": counts[-1],
                "total_frames": len(paths),
                "calibrated": calibration is not None,
                "updated": today,
            }
        except Exception:
            _logger.exception("compute_dso_convergence: failed for filter '%s'", fname)
    return results


def decay_ratio(counts: list[int], residuals: list[float]) -> Optional[float]:
    """How far the curve's tail sits above what independent noise predicts.

    Stacking k of n frames and comparing against all n leaves a residual of
    A*sqrt(1/k - 1/n) if every frame's noise is independent. Anchor A on the
    k=1 point and the ratio at the tail says whether this filter is actually
    averaging down: ~1.0 means textbook, and well above 1 means a correlated
    term — sky gradients, no flats, thermal residual — that more frames will
    not remove, so the 1/sqrt(N) promise does not apply to it.

    Measured 2026-08-01 on sh2-92: Ha 1.04 (textbook), O-III 2.44.
    """
    if len(counts) < 3 or len(residuals) != len(counts) or residuals[0] <= 0:
        return None
    n = counts[-1]
    k = counts[-2]
    if n <= 1 or k <= 0 or k >= n:
        return None
    A = residuals[0] / ((1 - 1 / n) ** 0.5)
    model = A * max(1 / k - 1 / n, 0) ** 0.5
    return residuals[-2] / model if model > 0 else None


def progress_summary(
    filter_name: str,
    counts: list[int],
    residuals: list[float],
    slope_pct: float,
    final_rmse_pct: float,
    total_frames: int,
) -> str:
    """A few sentences on where this filter stands and what more frames would buy.

    Answers the question the curve is actually for — keep shooting this filter or
    move on — rather than leaving a slope to be eyeballed. Stacking noise falls
    as 1/sqrt(N), so N more frames cut it by 1 - sqrt(n/(n+N)); that is what sets
    the "worth it" numbers here, and it is why the honest answer past a few
    hundred frames is almost always "move on".
    """
    n = counts[-1] if counts else 0
    if n <= 0:
        return f"**{filter_name}** — no frames measured."
    threshold = _threshold()
    min_frames = _min_frames()
    slope_abs = abs(slope_pct)
    gain = lambda extra: (1 - (n / (n + extra)) ** 0.5) * 100

    head = f"**{filter_name}** — {n} of {total_frames} frames stacked"
    if len(counts) >= 2:
        head += (f"; tail slope {slope_pct:+.3f}%/frame against a {threshold:g} "
                 f"threshold, and dropping to {counts[-2]} frames costs "
                 f"{final_rmse_pct:.1f}% of sky.")
    else:
        head += "."
    parts = [head]

    # Below min_frames the tail is a fit through almost nothing, so a flat slope
    # there is noise rather than convergence. Say so instead of reading the
    # threshold, which would contradict is_dso_done — it gates on the same count.
    if n < min_frames:
        parts.append(
            f"Too early to judge: the tail fit needs at least {min_frames} frames "
            f"and this has {n}, so the slope above is not yet meaningful. Another "
            f"50 frames would cut stack noise by {gain(50):.0f}%."
        )
        parts.append("Recommendation: keep shooting — there is not enough here to call it.")
        return " ".join(parts)

    if slope_abs <= threshold:
        need = 0
        parts.append(
            f"That is converged: the curve has flattened, so more frames are "
            f"polish rather than progress. Another 50 would cut stack noise by "
            f"{gain(50):.0f}% and 100 by {gain(100):.0f}%, which is less than a "
            f"night of better seeing would give you."
        )
        parts.append("Recommendation: move to another target or filter.")
    else:
        need = int(n * (slope_abs / threshold - 1) * (total_frames / max(n, 1)))
        parts.append(
            f"Still improving. At this rate it needs roughly {need} more frames "
            f"to flatten out; 50 more would cut stack noise by {gain(50):.0f}% "
            f"and 100 by {gain(100):.0f}%."
        )
        parts.append(
            "Recommendation: keep shooting this filter."
            if need <= 3 * n else
            "Recommendation: keep shooting, but that is several more nights — "
            "worth deciding whether this target deserves them."
        )

    ratio = decay_ratio(counts, residuals)
    if ratio is not None and ratio >= 1.5:
        parts.append(
            f"Caveat: this filter's residual is falling {ratio:.1f}x slower than "
            f"independent noise would, so it carries a correlated component — "
            f"gradients, or the missing flats — that averaging cannot remove. "
            f"The frame-count numbers above are an upper bound on what more "
            f"frames will actually buy here."
        )
    return " ".join(parts)


def is_dso_done(dso_name: str) -> bool:
    """True if every imaged filter has a flat tail AND low absolute RMSE AND min frames.

    Entries without ``calibrated: true`` were measured on the old uncalibrated
    scale, where the RMSE was a fraction of the bias pedestal and so ~100x
    smaller than the same data reads today. Judging them against the current
    thresholds would call almost anything done, so they count as not-yet-known
    and the target gets re-measured. That errs toward imaging a target twice
    rather than abandoning one early.
    """
    data = load_convergence()
    key = dso_name.lower().replace(" ", "")
    filters = data.get(key, {})
    if not filters:
        return False
    slope_threshold = _threshold()
    rmse_threshold = _rmse_threshold()
    min_frames = _min_frames()
    for info in filters.values():
        if not info.get("calibrated"):
            return False
        if info.get("frame_count", 0) < min_frames:
            return False
        if abs(info.get("tail_slope_pct", 999.0)) > slope_threshold:
            return False
        rmse = info.get("final_rmse_pct")
        if rmse is not None and rmse > rmse_threshold:
            return False
    return True


def frames_needed_estimate(dso_name: str) -> Optional[int]:
    """Worst-case frames needed across all filters.

    In the tail, slope ≈ k/N, so N_target = N * (|slope| / threshold),
    meaning frames_needed = N * (|slope| / threshold - 1).
    Returns None if no convergence data exists.

    N is ``frame_count``, the frames that survived the quality cut and got
    stacked — that is the N the slope was measured against. The answer, though,
    is frames to *shoot*, and some fraction of those get rejected, so it is
    scaled back up by the keep rate (``frame_count / total_frames``). Entries
    written before total_frames existed keep the old unscaled behaviour.
    """
    data = load_convergence()
    key = dso_name.lower().replace(" ", "")
    filters = data.get(key, {})
    if not filters:
        return None
    threshold = _threshold()
    worst: Optional[int] = None
    for info in filters.values():
        if not info.get("calibrated"):
            continue        # old pedestal-scale slope; see is_dso_done
        n = info.get("frame_count", 0)
        slope_abs = abs(info.get("tail_slope_pct", 0.0))
        if threshold <= 0 or n <= 0:
            continue
        needed = n * (slope_abs / threshold - 1)
        total = info.get("total_frames")
        if total and n:
            needed *= total / n          # kept frames -> frames to shoot
        needed = max(0, int(needed))
        if worst is None or needed > worst:
            worst = needed
    return worst
