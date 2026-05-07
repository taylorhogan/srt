"""Frame registration for N2N training.

All frames within a same-DSO group are registered to the group's first
frame using phase cross-correlation (translation only).  This corrects
for NINA dithering so that paired patches from two different frames
actually show the same patch of sky.

Registration is translation-only which is appropriate for dithered
sequences from a well-guided mount.  Pixels that shift outside the
frame boundary are filled with the frame median.
"""

from collections.abc import Callable

import numpy as np


def register_frames(
    frames: list[np.ndarray],
    group_ids: list[str],
    progress_cb: Callable[[str], None] = print,
) -> list[np.ndarray]:
    """Register all frames within each group to the first frame of that group.

    Returns a new list; original frames are not modified.
    """
    from skimage.registration import phase_cross_correlation
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

        # Downsample for speed; still accurate because we scale shift back up
        scale = 4
        ref_small = reference[::scale, ::scale].astype(np.float32)

        for idx in idxs[1:]:
            try:
                src_small = frames[idx][::scale, ::scale].astype(np.float32)
                shift, _, _ = phase_cross_correlation(
                    ref_small, src_small, upsample_factor=10
                )
                shift_full = shift * scale
                fill = float(np.median(frames[idx]))
                registered[idx] = nd_shift(
                    frames[idx], shift_full, mode="constant", cval=fill
                ).astype(frames[idx].dtype)
            except Exception as exc:
                progress_cb(f"  Warning: registration failed for frame {idx}: {exc}")

    return registered
