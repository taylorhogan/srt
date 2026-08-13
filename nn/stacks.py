"""Split-half stacks: N2N training pairs made of stacks rather than subs.

The rest of the N2N chain denoises every sub and then stacks. This module backs
the alternative — denoise the *stacked* image — by building the training pairs
it needs.

For each DSO the subs are registered, partitioned into two disjoint halves, and
each half is combined. The two results are two noisy views of the same scene
with independent noise, which is exactly what N2N requires, and they carry a
stack's noise character rather than a sub's, so inference on the full stack is
in distribution.

Why this is better posed than denoising subs:

- Per-frame denoising puts every frame through the same network with the same
  learned prior, so whatever it gets wrong it gets wrong identically on all of
  them. Independent noise averages down as sqrt(N); a shared bias survives
  stacking untouched. Denoising once, after stacking, cannot launder a bias into
  an apparently-converging curve.
- Inference costs ~200x less: one stack per target rather than every sub.
- The stack is the artefact that gets published, and denoised output is a
  display product.

What it gives up is the "buy frames" claim — that denoising subs reaches a given
SNR from fewer of them. That claim requires the denoiser to run before stacking.

The partition is disjoint by construction. Overlapping halves would share noise
realisations, the pair would no longer be independent, and the network could
reduce its loss by reproducing the shared noise — the failure N2N exists to
avoid.
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from nn import registration


def _combine(frames: list[np.ndarray]) -> np.ndarray:
    """Mean-combine registered frames.

    Mean rather than median: with the frame counts here (5-28 per half) the
    median is materially noisier, and outlier rejection is not this module's
    job — the pair only has to be two honest noisy views of the same scene.
    """
    acc = np.zeros(frames[0].shape, dtype=np.float64)
    for f in frames:
        acc += f
    return (acc / len(frames)).astype(np.float32)


def build_half_stacks(
    frames: list[np.ndarray],
    dso_names: list[str],
    min_frames_per_dso: int = 6,
    seed: int = 0,
    progress_cb: Callable[[str], None] = print,
) -> tuple[list[np.ndarray], list[str]]:
    """Return (stacks, group_ids) — two disjoint half-stacks per usable DSO.

    Output is ordered so each DSO's two stacks are adjacent, and group_ids
    repeats the DSO name for both. Feed straight to N2NDataset, whose
    within-group pairing then yields exactly the half-stack pairs.

    A DSO needs min_frames_per_dso subs to contribute: below that each half is
    too thin for its stack to resemble the full stack the model will be applied
    to.
    """
    by_dso: dict[str, list[int]] = {}
    for i, d in enumerate(dso_names):
        by_dso.setdefault(d, []).append(i)

    rng = np.random.default_rng(seed)
    out_frames: list[np.ndarray] = []
    out_groups: list[str] = []

    for dso in sorted(by_dso):
        idxs = by_dso[dso]
        if len(idxs) < min_frames_per_dso:
            progress_cb(f"  [{dso}] {len(idxs)} frames — skipped (need {min_frames_per_dso})")
            continue

        group = [frames[i] for i in idxs]
        # Register within the DSO first: the two halves must land on the same
        # pixels or the pair is two views of *different* scenes, which is the
        # bug that collapsed the sub-based training (see the lab manual).
        group = registration.register_frames(
            group, [dso] * len(group), progress_cb=lambda m: None
        )

        order = rng.permutation(len(group))
        half = len(order) // 2
        a = [group[i] for i in order[:half]]
        b = [group[i] for i in order[half:2 * half]]   # drop the odd frame

        out_frames += [_combine(a), _combine(b)]
        out_groups += [dso, dso]
        progress_cb(f"  [{dso}] {len(idxs)} frames -> 2 half-stacks of {len(a)}")

    return out_frames, out_groups


def build_full_stack(
    frames: list[np.ndarray],
    dso: str,
    progress_cb: Callable[[str], None] = print,
) -> Optional[np.ndarray]:
    """Register and combine every frame — what inference is applied to."""
    if not frames:
        return None
    reg = registration.register_frames(frames, [dso] * len(frames),
                                       progress_cb=lambda m: None)
    progress_cb(f"  [{dso}] full stack of {len(reg)} frames")
    return _combine(reg)
