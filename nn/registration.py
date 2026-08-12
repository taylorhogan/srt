"""Frame registration for N2N training.

All frames within a same-DSO group are registered to the group's first frame,
by phase cross-correlation on the background-subtracted frame, and the shift is
applied with scipy.ndimage.shift.

Until 2026-08-12 this used star centroid matching, and the docstring claimed it
was "reliable on sparse-star, noise-dominated frames where phase cross-
correlation fails". Measured against phase correlation as ground truth, the
opposite held, badly: on frames that arrive aligned to 0.0-0.1 px it invented
shifts of 3-14 px.

    m92       frame1  0.10 px -> 2.84    frame3  0.10 px -> 5.44
    abell2151 frame1  0.00 px -> 5.58    frame4  0.00 px -> 14.02

At FWHM ~6 px that is 1-2.3 FWHM, so a training pair's stars did not overlap at
all. N2N needs two noisy views of the *same* scene; given pairs that far apart
the only learnable answer is the local mean, and the net duly collapsed to a
constant uncorrelated with its input, erasing every star. This module, not the
loss and not the normalisation, is why.

The failure mode: _find_stars decimates by 2, so centroids carry +-1-2 px, and
_match_translation nearest-neighbour matched two 50-star lists with a 200 px
tolerance. In a dense field (m92 is a globular, abell2151 a galaxy cluster)
that pairs *different* stars, and the median of those mismatched deltas is a
spurious few-pixel shift — which was then applied unconditionally, with nothing
checking whether it improved anything.

Phase correlation recovered the true offset in all 12 test cases, including the
one frame with a real 8.6 px dither. The star-matching estimator is kept below
as a fallback for when phase correlation cannot lock on.
"""

from collections.abc import Callable

import numpy as np


def _background_subtracted(frame: np.ndarray) -> np.ndarray:
    """Frame minus its smooth sky, so phase correlation sees sources not gradient."""
    import sep

    f64 = frame.astype(np.float64)
    try:
        return (f64 - np.array(sep.Background(f64))).astype(np.float32)
    except Exception:
        return frame.astype(np.float32)


def _find_stars(frame: np.ndarray, scale: int = 2, n_stars: int = 50) -> np.ndarray:
    """Return (N, 2) array of (y, x) centroids for the N brightest sources."""
    import sep

    small = frame[::scale, ::scale].astype(np.float64)
    bkg = sep.Background(small)
    data_sub = (small - bkg).astype(np.float64)
    try:
        sources = sep.extract(data_sub, thresh=3.0, err=bkg.globalrms)
    except Exception:
        return np.empty((0, 2))
    if len(sources) == 0:
        return np.empty((0, 2))
    # Sort ascending by flux, keep the brightest n_stars
    order = np.argsort(sources["flux"])
    sources = sources[order[-n_stars:]]
    yx = np.column_stack([sources["y"], sources["x"]]) * scale
    return yx


def _match_translation(
    ref_stars: np.ndarray,
    src_stars: np.ndarray,
    max_dist: float = 200.0,
) -> np.ndarray | None:
    """Median (dy, dx) from nearest-neighbour centroid matches."""
    from scipy.spatial import cKDTree

    tree = cKDTree(ref_stars)
    dists, idxs = tree.query(src_stars, k=1)
    mask = dists < max_dist
    if mask.sum() < 3:
        return None
    deltas = ref_stars[idxs[mask]] - src_stars[mask]
    return np.median(deltas, axis=0)


def _phase_translation(
    reference: np.ndarray,
    source: np.ndarray,
    max_shift: float = 200.0,
) -> "np.ndarray | None":
    """(dy, dx) to bring *source* onto *reference*, by phase cross-correlation.

    Runs on 2x-decimated, background-subtracted data for speed and scales the
    result back up; upsample_factor=10 keeps that to ~0.2 px precision, well
    inside the sub-pixel accuracy the frames already arrive with.

    Returns None rather than a guess when the fit is implausible, so an
    unlockable frame is left alone instead of being shifted by garbage — the
    exact failure that made the previous estimator destructive.
    """
    from skimage.registration import phase_cross_correlation

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            sh, _, _ = phase_cross_correlation(
                reference[::2, ::2].astype(np.float32),
                source[::2, ::2].astype(np.float32),
                upsample_factor=10,
                normalization=None,
            )
    except Exception:
        return None
    shift = np.asarray(sh, dtype=np.float64) * 2.0
    if not np.all(np.isfinite(shift)) or np.hypot(*shift) > max_shift:
        return None
    return shift


def register_frames(
    frames: list[np.ndarray],
    group_ids: list[str],
    progress_cb: Callable[[str], None] = print,
    min_shift: float = 0.5,
) -> list[np.ndarray]:
    """Register all frames within each group to the first frame of that group.

    Frames below *min_shift* px are left untouched rather than resampled. This
    matters more than it looks: these frames arrive aligned to ~0.1 px, so
    nearly every shift is a no-op, and nd_shift's order-3 spline is not free —
    it low-pass filters the frame, which on an N2N pair correlates the very
    noise the method assumes is independent.

    Returns a new list; original frames are not modified.
    """
    from scipy.ndimage import shift as nd_shift

    groups: dict[str, list[int]] = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(idx)

    registered = [f.copy() for f in frames]

    for gid, idxs in groups.items():
        reference = frames[idxs[0]]
        n = len(idxs) - 1
        if n == 0:
            continue
        progress_cb(f"  [{gid}] registering {n} frames…")

        ref_sub = _background_subtracted(reference)
        ref_stars = None          # built lazily, only if phase correlation fails
        moved = skipped = failed = 0

        for idx in idxs[1:]:
            try:
                shift = _phase_translation(ref_sub, _background_subtracted(frames[idx]))
                if shift is None:
                    # Fallback to the old estimator rather than leaving the
                    # frame unregistered — but only when phase correlation
                    # could not lock on at all.
                    if ref_stars is None:
                        ref_stars = _find_stars(reference)
                    src_stars = _find_stars(frames[idx])
                    if len(ref_stars) < 3 or len(src_stars) < 3:
                        failed += 1
                        continue
                    shift = _match_translation(ref_stars, src_stars)
                    if shift is None:
                        failed += 1
                        continue
                if np.hypot(*shift) < min_shift:
                    skipped += 1          # already aligned; do not resample
                    continue
                fill = float(np.median(frames[idx]))
                registered[idx] = nd_shift(
                    frames[idx], shift, mode="constant", cval=fill
                ).astype(frames[idx].dtype)
                moved += 1
            except Exception as exc:
                failed += 1
                progress_cb(f"  Warning: registration failed for frame {idx}: {exc}")

        progress_cb(
            f"  [{gid}] {moved} shifted, {skipped} already aligned (<{min_shift} px), "
            f"{failed} unregistered"
        )

    return registered
