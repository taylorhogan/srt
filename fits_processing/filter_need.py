"""Need-weighted filter shares — §2c of docs/ARCHITECTURE_PLAN.md.

Answers one question for the night planner: given what the convergence record
already knows about a DSO, how should tonight's usable seconds split across
the object type's filter set? The answer is a weight per filter, where a
weight is an estimate of frames still worth shooting:

  * a filter at its convergence gate gets ZERO share — more frames there are
    polish, and every second spent on it is stolen from a filter that needs it;
  * a filter in the type's set with NO usable measurement gets the LARGEST
    share — this rule alone would have caught the zero-S-II archive, where a
    never-imaged filter stayed invisible precisely because it had no record to
    look converged or unconverged;
  * otherwise the share is proportional to the frames still needed, the same
    tail-slope arithmetic as convergence.frames_needed_estimate, per filter.

The core (`filter_need`) is a pure function of the DSO's convergence entries
and the thresholds, so the whole policy is testable from a dict fixture with
no config, no astropy, and no hardware — and CI (which installs only pytest)
can run those tests. Disk and config access live only in `filter_need_for_dso`.

Weights are frame counts, not percentages: callers turn them into tonight's
exposure counts with `counts_from_weights`, and hand those to the existing
explicit-plan machinery (`_apply_filter_plan(..., explicit=...)`). Need-
weighting deliberately reduces to computing counts for a path that already
exists — see the plan's §2c.
"""
from typing import Optional

# Which filters an object type wants — mirrors the split hard-coded in
# nina_sequence_gen._apply_filter_plan (nebula -> narrowband, else broadband).
# Order matters: it is the order counts are assigned to template blocks, so
# the first filter gets the night's earliest (darkest) hours.
TYPE_FILTER_SETS = {
    "nebula": ("Ha", "O-III", "S-II"),
    "galaxy": ("L", "R", "G", "B"),
    "unknown": ("L", "R", "G", "B"),
}

# Fallback thresholds, mirroring configs.config_public["convergence"] defaults.
# filter_need_for_dso overrides them from live config; tests pass their own.
DEFAULT_SLOPE_THRESHOLD = 0.40
DEFAULT_RMSE_THRESHOLD = 5.0
DEFAULT_MIN_FRAMES = 30


def _entry_for(entries: dict, filter_name: str) -> Optional[dict]:
    """Case-insensitive lookup: convergence keys come from FITS headers."""
    want = filter_name.lower()
    for name, info in entries.items():
        if str(name).lower() == want:
            return info
    return None


def _is_converged(info: dict, slope_threshold: float, rmse_threshold: float,
                  min_frames: int) -> bool:
    """The same gate as convergence.is_dso_done, for one filter."""
    if info.get("frame_count", 0) < min_frames:
        return False
    if abs(info.get("tail_slope_pct", 999.0)) > slope_threshold:
        return False
    rmse = info.get("final_rmse_pct")
    if rmse is not None and rmse > rmse_threshold:
        return False
    return True


def _frames_needed(info: dict, slope_threshold: float, min_frames: int) -> float:
    """Frames still worth SHOOTING for one unconverged filter.

    Tail slope ~ k/N, so flattening to the threshold needs
    N * (|slope|/threshold - 1) more *stacked* frames; scaled up by the keep
    rate because some shot frames fail the quality cut (same arithmetic as
    convergence.frames_needed_estimate). Below min_frames the slope is a fit
    through almost nothing, so the need is floored at what it takes to reach a
    judgeable count.
    """
    n = info.get("frame_count", 0)
    slope_abs = abs(info.get("tail_slope_pct", 0.0))
    needed = 0.0
    if n > 0 and slope_threshold > 0:
        needed = n * (slope_abs / slope_threshold - 1)
        total = info.get("total_frames")
        if total and n:
            needed *= total / n
    needed = max(needed, float(min_frames - n), 1.0)
    return needed


def filter_need(entries: Optional[dict], obj_type: str,
                slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
                rmse_threshold: float = DEFAULT_RMSE_THRESHOLD,
                min_frames: int = DEFAULT_MIN_FRAMES) -> Optional[dict]:
    """{filter: weight} for a DSO's convergence entries, or None.

    entries: this DSO's slice of convergence.json ({filter: info}); None or {}
        means the DSO has no imaging history at all, and the answer is None —
        the caller should fall back to the object-type default split, because
        with no measurements every weight would be an invention.
    obj_type: 'nebula' / 'galaxy' / 'unknown' (anything else reads as unknown).

    Returns {} when every filter in the set is converged (the planner should
    not have scheduled this target; callers treat it like None but may log it).

    An entry without ``calibrated: true`` was measured on the old pedestal
    scale (~100x smaller numbers — see convergence.is_dso_done); its need is
    unknowable, so it is treated exactly like a missing filter: largest share.
    """
    if not entries:
        return None
    filter_set = TYPE_FILTER_SETS.get(obj_type, TYPE_FILTER_SETS["unknown"])

    needs: dict[str, Optional[float]] = {}
    for fname in filter_set:
        info = _entry_for(entries, fname)
        if info is None or not info.get("calibrated"):
            needs[fname] = None          # no usable measurement
        elif _is_converged(info, slope_threshold, rmse_threshold, min_frames):
            needs[fname] = 0.0
        else:
            needs[fname] = _frames_needed(info, slope_threshold, min_frames)

    known = [v for v in needs.values() if v]
    largest = max(known + [float(min_frames)])
    weights = {f: (largest if v is None else v) for f, v in needs.items()}
    if not any(weights.values()):
        return {}
    return weights


def counts_from_weights(weights: dict, usable_seconds: float,
                        exposure_seconds: float) -> dict:
    """Split tonight's seconds by weight into whole exposure counts.

    Zero-weight filters keep an explicit 0 in the result: the explicit-plan
    applicator writes that 0 into the template block, which is what stops a
    converged filter's built-in template count from running by accident.
    """
    total = sum(weights.values())
    if total <= 0 or usable_seconds <= 0 or exposure_seconds <= 0:
        return {}
    return {f: int(usable_seconds * w / total // exposure_seconds)
            for f, w in weights.items()}


def filter_need_for_dso(dso_name: str, obj_type: str) -> Optional[dict]:
    """The live wrapper: convergence.json + config thresholds. Never raises."""
    try:
        from fits_processing import convergence as conv
        data = conv.load_convergence()
        entries = data.get(dso_name.lower().replace(" ", ""))
        return filter_need(entries, obj_type,
                           slope_threshold=conv._threshold(),
                           rmse_threshold=conv._rmse_threshold(),
                           min_frames=conv._min_frames())
    except Exception:
        return None
