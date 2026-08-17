"""Split stacks: N2N training pairs made of stacks rather than subs.

For each (DSO, filter) the subs are partitioned into disjoint subsets and each
subset is stacked. Two such stacks are two noisy views of the same scene with
independent noise — what N2N requires — and they carry a stack's noise
character, so inference on the full stack is in distribution.

Stacking is delegated entirely to `stacking.stacker.stack`. This module used to
carry its own: register with `nn.registration`, then a plain mean. That was a
mistake and it produced visibly wrong stacks — no calibration, so hot pixels and
dark structure survived; no sigma clipping, so satellite trails and cosmic rays
went straight into the mean; no quality gate, so soft frames counted equally.
The real stacker does bias/dark/flat calibration, an FWHM and star-count quality
gate, astroalign registration (which fits rotation, not just the translation
`nn.registration` is limited to), sky levelling to a common background before
combining, and a tiled sigma-clip combine with FWHM weighting. There is no
reason for a second implementation to exist, and every reason for it not to.

Why stacks rather than subs:

- Per-frame denoising puts every frame through the same network with the same
  learned prior, so whatever it gets wrong it gets wrong identically on all of
  them. Independent noise averages down as sqrt(N); a shared bias survives
  stacking untouched. Denoising once, after stacking, cannot launder a bias into
  an apparently-converging curve.
- Inference costs ~200x less: one stack per target rather than every sub.
- The stack is the artefact that gets published, and denoised output is a
  display product.

The partition is disjoint by construction. Overlapping subsets would share noise
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


def index_frames(
    subs_dir: Path,
    filters: list[str],
    exptime_s: int,
) -> dict[tuple[str, str], list[Path]]:
    """Map (dso, filter) -> LIGHT frame paths, reading headers only.

    Pixel data is never touched, so this costs seconds where loading would cost
    ~200 GB across L+R+G+B.
    """
    from astropy.io import fits as _fits

    want = {f.strip() for f in filters}
    out: dict[tuple[str, str], list[Path]] = {}
    for fp in sorted(Path(subs_dir).rglob("*.fits")):
        if fp.parent.name.upper() != "LIGHT":
            continue
        try:
            hdr = _fits.getheader(fp)
            filt = str(hdr.get("FILTER", "")).strip()
            if filt not in want:
                continue
            if round(float(hdr.get("EXPTIME", 0))) != exptime_s:
                continue
        except Exception:
            continue
        try:
            dso = fp.relative_to(subs_dir).parts[0].lower().replace(" ", "")
        except ValueError:
            dso = fp.parent.parent.name.lower().replace(" ", "")
        out.setdefault((dso, filt), []).append(fp)
    return out


def stack_paths(
    paths: list[Path],
    filter_name: str,
    progress_cb: Callable[[str], None] = print,
) -> Optional[np.ndarray]:
    """Stack *paths* with the project stacker, calibrated where masters exist."""
    from stacking import stacker

    if len(paths) < 2:
        return None
    bias, dark, flat = stacker.calibration_paths_from_config(filter_name)
    if not bias and not dark:
        progress_cb("    no calibration masters in config — stacking uncalibrated")
    img, _meta = stacker.stack(
        list(paths),
        method=stacker.StackMethod.SIGMA_CLIP_FWHM,
        bias_paths=bias or None,
        dark_paths=dark or None,
        flat_paths=flat or None,
        register=True,
        # Pass the caller's callback through rather than swallowing it. The
        # stacker measures FWHM per frame, which dominates the wall time on a
        # large set, and with progress suppressed a multi-hour build is
        # indistinguishable from a hang.
        progress_cb=lambda m: progress_cb(f"    {m}"),
    )
    return img


def build_split_stacks(
    subs_dir: Path,
    filters: list[str],
    exptime_s: int,
    min_frames_per_split: int = 6,
    max_splits: int = 2,
    exclude_dsos: Optional[set[str]] = None,
    seed: int = 0,
    progress_cb: Callable[[str], None] = print,
) -> tuple[list[np.ndarray], list[str]]:
    """Return (stacks, group_ids) — N disjoint sub-stacks per (dso, filter).

    Output is ordered so a group's stacks are adjacent and group_ids repeats the
    key across them. Feed straight to N2NDataset, whose within-group pairing then
    yields every pair among that group's splits.

    **The group key is `dso|filter`, never bare dso.** Pairing an L stack with an
    R stack of the same target would pair two different scenes — a star's
    relative brightness changes with passband, nebulosity present in Ha is absent
    in R — which is the same class of error as the misregistration bug that
    collapsed the sub-based training.

    `max_splits` is 2, not 4. Quartering gives 5x the training pairs and measured
    worse: the response to an injected source fell from 1.07 to 0.56 at SNR 5.7,
    because quarter-stacks are sqrt(2) noisier than the full stack inference runs
    on and so teach over-aggressive shrinkage. Depth match beat pair count.

    `exclude_dsos` drops whole targets across every filter. It has to be
    DSO-level: a group-level holdout would hold out "m92|R" while training on
    "m92|L", the same field in a different passband.
    """
    idx = index_frames(subs_dir, filters, exptime_s)
    excl = {d.lower() for d in (exclude_dsos or set())}
    rng = np.random.default_rng(seed)
    out_frames: list[np.ndarray] = []
    out_groups: list[str] = []
    total_pairs = 0

    for (dso, filt) in sorted(idx):
        if dso in excl:
            progress_cb(f"  [{dso}|{filt}] excluded")
            continue
        paths = idx[(dso, filt)]
        n_splits = min(max_splits, len(paths) // min_frames_per_split)
        if n_splits < 2:
            progress_cb(f"  [{dso}|{filt}] {len(paths)} frames — skipped "
                        f"(need {2 * min_frames_per_split})")
            continue

        order = rng.permutation(len(paths))
        per = len(order) // n_splits
        made = []
        for k in range(n_splits):
            chunk = [paths[i] for i in order[k * per:(k + 1) * per]]
            img = stack_paths(chunk, filt, progress_cb=progress_cb)
            if img is None:
                break
            made.append(img)
        if len(made) != n_splits:
            progress_cb(f"  [{dso}|{filt}] stacking failed — skipped")
            continue

        out_frames += made
        out_groups += [f"{dso}|{filt}"] * n_splits
        pairs = n_splits * (n_splits - 1) // 2
        total_pairs += pairs
        progress_cb(f"  [{dso}|{filt}] {len(paths)} frames -> {n_splits} stacks "
                    f"of {per} ({pairs} pairs)")

    out_frames = crop_to_common(out_frames)
    progress_cb(f"  total {len(out_frames)} stacks, {total_pairs} pairs, "
                f"{len(set(out_groups))} groups"
                + (f", cropped to {out_frames[0].shape}" if out_frames else ""))
    return out_frames, out_groups


def crop_to_common(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Crop every stack to the smallest common shape.

    The stacker trims its output to the region all frames overlap after
    registration, so two stacks of the same target can differ by a few pixels —
    measured 6371x9570 against the sensor's 6388x9576. N2NDataset takes a pair
    and crops the *same* coordinates from both, so a mismatch produces patches
    of different shapes and the DataLoader dies in collate with "Trying to
    resize storage that is not resizable", which names nothing that would lead
    you here.

    Cropping from the origin is right because the stacker's own trim is already
    referenced to its registration origin; the discrepancy is a boundary, not
    an offset, so this does not shift one stack against another.
    """
    if not frames:
        return frames
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    return [f if f.shape == (h, w) else f[:h, :w] for f in frames]
