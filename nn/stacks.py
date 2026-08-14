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


def build_split_stacks(
    frames: list[np.ndarray],
    dso_names: list[str],
    min_frames_per_split: int = 6,
    # 2, not 4. Quartering gives 5x the training pairs and measured worse: the
    # response to an injected source fell from 1.07 to 0.56 at SNR 5.7, because
    # quarter-stacks are sqrt(2) noisier than the full stack inference runs on
    # and so teach over-aggressive shrinkage. Depth match beat pair count.
    max_splits: int = 2,
    seed: int = 0,
    progress_cb: Callable[[str], None] = print,
) -> tuple[list[np.ndarray], list[str]]:
    """Return (stacks, group_ids) — N disjoint sub-stacks per usable DSO.

    Output is ordered so a DSO's stacks are adjacent and group_ids repeats the
    DSO name across them. Feed straight to N2NDataset, whose within-group
    pairing then yields every pair among that DSO's splits — C(N,2) of them, so
    4 splits give 6 pairs where 2 give 1.

    **The split count is adaptive**, `frames // min_frames_per_split` clamped to
    [2, max_splits]. Data thinness is the binding constraint here — 4 training
    pairs, with validation bottoming out by epoch 7 — but a fixed split count
    trades that against the opposite problem: quarter-stacks of a shallow target
    are too noisy to resemble the full stack inference is applied to. Scaling
    per DSO takes the extra pairs from the deep targets without forcing the
    shallow ones below usable depth, and keeps every DSO in play, which matters
    because scene diversity is what the sub-based run lacked most.

    A DSO needs `2 * min_frames_per_split` subs to contribute at all. That floor
    also excludes the 9-frame abell2218, whose 4-frame halves sat far off the
    distribution the model is applied to.

    Splits are disjoint by construction. Overlapping ones would share noise
    realisations, the pair would no longer be independent, and the network could
    reduce its loss by reproducing the shared noise — the failure N2N exists to
    avoid.
    """
    by_dso: dict[str, list[int]] = {}
    for i, d in enumerate(dso_names):
        by_dso.setdefault(d, []).append(i)

    rng = np.random.default_rng(seed)
    out_frames: list[np.ndarray] = []
    out_groups: list[str] = []
    total_pairs = 0

    for dso in sorted(by_dso):
        idxs = by_dso[dso]
        n_splits = min(max_splits, len(idxs) // min_frames_per_split)
        if n_splits < 2:
            progress_cb(f"  [{dso}] {len(idxs)} frames — skipped "
                        f"(need {2 * min_frames_per_split})")
            continue

        group = [frames[i] for i in idxs]
        # Register within the DSO first: the splits must land on the same pixels
        # or a pair is two views of *different* scenes, which is the bug that
        # collapsed the sub-based training (see the lab manual).
        group = registration.register_frames(
            group, [dso] * len(group), progress_cb=lambda m: None
        )

        order = rng.permutation(len(group))
        per = len(order) // n_splits          # equal depth; remainder dropped
        for k in range(n_splits):
            chunk = [group[i] for i in order[k * per:(k + 1) * per]]
            out_frames.append(_combine(chunk))
            out_groups.append(dso)
        pairs = n_splits * (n_splits - 1) // 2
        total_pairs += pairs
        progress_cb(f"  [{dso}] {len(idxs)} frames -> {n_splits} stacks of {per} "
                    f"({pairs} pairs)")

    progress_cb(f"  total {len(out_frames)} stacks, {total_pairs} pairs")
    return out_frames, out_groups


# Kept so older callers and the lab manual's step 12 still resolve.
def build_half_stacks(frames, dso_names, min_frames_per_dso=6, seed=0,
                      progress_cb=print):
    """Two disjoint half-stacks per DSO — the original 2-way split."""
    return build_split_stacks(
        frames, dso_names,
        min_frames_per_split=max(1, min_frames_per_dso // 2),
        max_splits=2, seed=seed, progress_cb=progress_cb,
    )


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
