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

    from stacking.color_process import canonical_filter

    # Match through the alias table rather than on the literal FILTER card, and
    # key the result on the canonical name. Matching literally silently drops
    # any frame whose capture software spelled the filter differently — the
    # 2024 archive writes O-III as "O2", so a literal match discarded 122 O-III
    # frames of NGC 6888 while reporting nothing.
    want = {}
    for f in filters:
        f = f.strip()
        want[f] = canonical_filter(f) or f
    canon_want = {v: k for k, v in want.items()}
    out: dict[tuple[str, str], list[Path]] = {}
    for fp in sorted(Path(subs_dir).rglob("*.fits")):
        if fp.parent.name.upper() != "LIGHT":
            continue
        try:
            hdr = _fits.getheader(fp)
            raw_filt = str(hdr.get("FILTER", "")).strip()
            canon = canonical_filter(raw_filt) or raw_filt
            if canon not in canon_want:
                continue
            filt = canon_want[canon]
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


# How many frames the shared-reference picker will try, and how many control
# points make a frame good enough to stop looking. astroalign caps extraction
# at stacker._REG_MAX_CONTROL_POINTS (200); a clear 300 s L frame of a galaxy
# field hits that cap, a thin-cloud frame does not, and a clouded one gives
# none. Each candidate costs one calibrated load and extraction (~3 s at 61 MP).
REF_CANDIDATES = 9
REF_MIN_POINTS = 50


def shared_reference_for(
    paths: list[Path],
    filter_name: str,
    progress_cb: Callable[[str], None] = print,
) -> Optional[tuple]:
    """Control points + shape for ONE reference frame, shared across splits.

    Without this each `stack_paths` call picks its own reference and the splits
    of a training pair land on different pixel grids — measured 8 px and 36 px
    apart on two stacks of the same target. That breaks N2N's premise that input
    and target differ only in noise, and the loss-minimising response to it is to
    delete point sources: the target holds background where the input holds a
    star. Convolution cannot compensate, because astroalign fits rotation as well
    as translation and a translation-invariant operator cannot undo a
    position-dependent shift.

    Reference is the sharpest frame when a `frame_stats.json` cache is available
    (the same picker `color_process` uses), else the middle frame — and if that
    frame yields too few control points (clouds, a slew caught mid-exposure),
    up to REF_CANDIDATES frames fanned out from the middle are tried before
    giving up. Which one is chosen only affects how many frames align; that it
    is the *same* one for every split is what this function exists for.

    **On the Spark the cache never hits, so this is always the middle frame.**
    `frame_stats.json` is written by the observatory PC and its `path` keys are
    Windows absolute paths (`C:\\Users\\iriso\\Documents\\N.I.N.A\\Targets\\...`),
    while `load_precomputed_fwhm_stars` matches on `os.path.abspath` — which on
    Linux can never equal a `C:\\...` string. Every lookup misses, the `except`
    below is not reached (an empty dict is not an error), and no line is logged.
    Measured 2026-08-19: every one of the 9 targets has a cache, none of them
    match.

    Two consequences, neither of them visible in any output:

    1. The reference is the middle frame, not the sharpest, on every N2N run
       ever done on this machine.
    2. The stacker re-measures FWHM for all four `stack_paths` calls of a group.
       That is 65% of stack wall time (2.2 min of a 3.4 min 17-frame stack,
       measured from iris.log), so the cache would roughly halve it.

    Fixing it is **not** a pure speedup and must not be done casually: the cached
    values disagree with what the stacker measures itself — FWHM ~11% high and
    star counts ~1.5x low on abell2151 G, systematically — and the quality gate
    cuts on both, so the surviving frames would change. See the runbook.
    """
    from stacking import stacker

    if not paths:
        return None
    n = len(paths)
    mid = n // 2
    first = mid
    try:
        from fits_processing.frame_cache import load_precomputed_fwhm_stars
        arcsec = stacker._get_arcsec_per_pixel()
        dso_dir = Path(paths[0]).parents[3]
        pre = load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
        if pre:
            fwhm = {p: v[0] for p, v in pre.items()}
            idx = stacker._reference_index_by_fwhm([fwhm.get(p, 0.0) for p in paths])
            if idx is not None:
                first = idx
    except Exception:
        progress_cb("    no FWHM cache — using the middle frame as reference")

    # The first choice is tried blind, then neighbours fanned out from the
    # middle of the run. 2026-10-18: ngc6946's 41 L frames had a clouded
    # hour in the middle (frames 17-28 of the 10-15 session, 0-232 stars,
    # sky 2855 ADU against 647) and the middle frame was one of them: no
    # control points, and the whole target's render was refused on a night
    # that had 12 good frames either side. The quality gate would have
    # dropped those frames at stack time; the reference pick was the one
    # place a single bad frame could still stop everything.
    step = max(1, n // (REF_CANDIDATES - 1)) if n > 1 else 1
    order = [first]
    for k in range(1, REF_CANDIDATES):
        for i in (mid - k * step, mid + k * step):
            if 0 <= i < n and i not in order:
                order.append(i)
    order = order[:REF_CANDIDATES]

    cal = stacker.calibration_from_config(filter_name)
    best = None
    for i in order:
        ref_path = paths[i]
        try:
            reference = stacker._load_calibrated(ref_path, cal)
            pts = stacker._reference_control_points(stacker._despike(reference))
        except Exception:
            progress_cb(f"    {ref_path.name}: could not load as a reference — skipped")
            continue
        if pts is None:
            progress_cb(f"    {ref_path.name}: no control points — trying another")
            continue
        if len(pts) >= REF_MIN_POINTS:
            progress_cb(f"    shared reference: {ref_path.name} ({len(pts)} control points)")
            return (pts, reference.shape)
        progress_cb(f"    {ref_path.name}: only {len(pts)} control points — trying another")
        if best is None or len(pts) > len(best[0]):
            best = (pts, reference.shape, ref_path.name)
    if best is not None:
        progress_cb(f"    shared reference: {best[2]} ({len(best[0])} control points, "
                    f"best of {len(order)} candidates)")
        return (best[0], best[1])
    progress_cb(f"    reference yielded no control points in {len(order)} candidates — per-split refs")
    return None


def stack_paths(
    paths: list[Path],
    filter_name: str,
    progress_cb: Callable[[str], None] = print,
    shared_reference: Optional[tuple] = None,
    meta_out: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Stack *paths* with the project stacker, calibrated where masters exist.

    `meta_out`, if given, is updated with the stacker's own info dict — most
    usefully `n_frames`, the count that survived the quality gate. That is not
    `len(paths)`: the gate rejects on FWHM and star count, and it routinely
    drops a quarter of a split (measured on abell2151 G, 5 of 17 and 3 of 17 in
    the two halves of one training pair). Anything reasoning about stack depth —
    the depth-match argument of lab manual step 16 above all — needs the
    surviving count, not the nominal one.
    """
    from stacking import stacker

    if len(paths) < 2:
        return None
    bias, dark, flat = stacker.calibration_paths_from_config(filter_name)
    if not bias and not dark:
        progress_cb("    no calibration masters in config — stacking uncalibrated")
    img, _meta = stacker.stack(
        list(paths),
        method=stacker.StackMethod.SIGMA_CLIP_FWHM,
        shared_reference=shared_reference,
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
    if meta_out is not None:
        meta_out.update(_meta)
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
        # One reference for every split of this group, so the pair the network
        # trains on is the same scene on the same pixels.
        ref = shared_reference_for(paths, filt, progress_cb=progress_cb)
        made = []
        for k in range(n_splits):
            chunk = [paths[i] for i in order[k * per:(k + 1) * per]]
            img = stack_paths(chunk, filt, progress_cb=progress_cb,
                              shared_reference=ref)
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

    This makes the shapes match and **nothing else**. An earlier version of this
    docstring claimed "the discrepancy is a boundary, not an offset, so this does
    not shift one stack against another" — that was wrong. Two stacks built from
    their own references sit on different grids, measured 8 px and 36 px apart,
    and cropping to a common shape does not align them. Alignment is
    `shared_reference_for`'s job and has to happen at stack time; this function
    runs afterwards and cannot recover it.
    """
    if not frames:
        return frames
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    return [f if f.shape == (h, w) else f[:h, :w] for f in frames]
