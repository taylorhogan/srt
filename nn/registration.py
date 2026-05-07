"""Frame registration for N2N training.

All frames within a same-DSO group are registered to the group's first frame
using star centroid matching (SEP / Source Extractor).  The N brightest stars
are detected in each frame, matched by nearest-neighbour, and the median
translation is applied with scipy.ndimage.shift.

This approach is reliable on sparse-star, noise-dominated frames where phase
cross-correlation fails.  Frames with too few detectable stars are left
unshifted with a warning.
"""

from collections.abc import Callable

import numpy as np


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


def register_frames(
    frames: list[np.ndarray],
    group_ids: list[str],
    progress_cb: Callable[[str], None] = print,
) -> list[np.ndarray]:
    """Register all frames within each group to the first frame of that group.

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

        ref_stars = _find_stars(reference)
        if len(ref_stars) < 3:
            progress_cb(f"  Warning: too few stars in reference frame for [{gid}], skipping group")
            continue

        for idx in idxs[1:]:
            try:
                src_stars = _find_stars(frames[idx])
                if len(src_stars) < 3:
                    progress_cb(f"  Warning: too few stars in frame {idx}, skipping")
                    continue
                shift = _match_translation(ref_stars, src_stars)
                if shift is None:
                    progress_cb(f"  Warning: no centroid matches for frame {idx}, skipping")
                    continue
                fill = float(np.median(frames[idx]))
                registered[idx] = nd_shift(
                    frames[idx], shift, mode="constant", cval=fill
                ).astype(frames[idx].dtype)
            except Exception as exc:
                progress_cb(f"  Warning: registration failed for frame {idx}: {exc}")

    return registered
