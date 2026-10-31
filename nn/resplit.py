"""Re-split N2N pairs: a fresh random half-split of a group's frames for every patch.

The ladder (scripts/n2n_lrgb_ladder.py) cuts each (dso, filter) group into two
fixed half-stacks once, so 23 groups are 23 training pairs and an epoch of 2000
patch draws re-reads the same 23 noise realisations. The 2026-10-31 retrain
(23 groups against 12) tied the previous model on the held-out trunk, which
said the pool's breadth was not the limit. This module changes what a pair IS,
not which frames go in: the group's training frames are calibrated, registered
and sky-levelled ONCE by the project stacker (`stacker.stack(keep_frames_dir=)`,
so there is no second registration) and kept on disk; at training time every
patch draw permutes the frames, splits them into two disjoint halves, and
stacks the patch region of each half with the stacker's own combine
(`_combine_tile`, SIGMA_CLIP_FWHM, the frames' FWHM weights). A group of 30
frames has C(30,15)/2 ~ 7.7e7 distinct pairs where it had one.

What is held fixed, on purpose:

- **The frames.** Each group's training frames are the same seeded permutation
  slice the fixed halves were built from (`order[:2 * train_per]`), so a model
  trained this way differs from the fixed-pair model in the pairing alone.
- **The depth.** Each half holds floor(k / 2) frames, the depth the fixed halves
  had; lab manual step 16 measured shallower-than-inference stacks teaching
  over-aggressive shrinkage, and quartering (5x the pairs) losing to halving.
- **The validation stacks**, the fixed `val0/val1` files, so the val loss is
  comparable with every earlier ladder run.
- **The normalisation.** The background map is sep.Background of the group's
  full-depth stack (the half's own map differs from it by noise), every half
  is put back at the full stack's restored sky level rather than its own
  frames' (the frames were levelled to be interchangeable; which level goes
  back is a convention, and a half's own weighted level sat 9 ADU off the
  map on ic410 O-III, 1.8 asinh units at the sky), and the asinh
  scale is the one `denoiser.normalise` gives the group's fixed `train0` half:
  a half of the same depth, so the same sky sigma. Inference normalises the
  frame it is given; a half-stack of this depth is what it was trained on.

Patch origins come from N2NDataset's source-biased sampler run on the full
stack (same block CDF, same fill-map rule), clamped into the box where at
least 80 % of the frames have real pixels, because outside it a half holds
few frames and its noise no longer matches the depth the model is told.
"""
import json
import os
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

GROUP_FILE = "group.json"
FRAMES_SUBDIR = "frames"
ALL_STACK = "all.npy"
BACKGROUND = "bg.npy"
MIN_COVER = 0.8
SIGMA = 3.0


def build_group(paths: list, filter_name: str, out_dir: Path, scale_from: Path,
                shared_reference=None, progress_cb: Callable[[str], None] = print) -> Optional[dict]:
    """Stack *paths* once, keeping every registered frame under *out_dir*.

    Writes frames/f*.npy (the stacker's levelled, registered frames), all.npy
    (the full-depth stack, for the sampler and as a deep reference), bg.npy
    (sep.Background of that stack) and group.json. *scale_from* is a stack of
    the depth a half will have (the ladder's train0), whose sky sigma sets the
    asinh scale. Returns the group record, or None when the stack failed.
    """
    from nn import denoiser, stacks
    from stacking import stacker

    out_dir = Path(out_dir)
    fdir = out_dir / FRAMES_SUBDIR
    if fdir.exists():
        for f in fdir.glob("f*.npy"):
            f.unlink()
    meta: dict = {}
    img = stacks.stack_paths(list(paths), filter_name, progress_cb=progress_cb,
                             shared_reference=shared_reference, meta_out=meta,
                             keep_frames_dir=fdir)
    if img is None or not meta.get("frames"):
        return None
    fr = meta["frames"]
    n = len(fr["files"])
    if n < 4:
        progress_cb(f"    only {n} frames survived — too few to split")
        return None
    cov = np.zeros(img.shape, dtype=np.int32)
    for f in fr["files"]:
        cov += ~np.isnan(np.load(f, mmap_mode="r"))
    box = stacker.coverage_box(cov, max(1, int(round(MIN_COVER * n))))
    del cov
    sub, bg = denoiser.subtract_background(img)
    del sub
    np.save(out_dir / ALL_STACK, img.astype(np.float32))
    np.save(out_dir / BACKGROUND, bg.astype(np.float32))
    half = np.load(scale_from)
    hsub, _ = denoiser.subtract_background(half)
    _, scale = denoiser.normalise(hsub)
    del half, hsub
    rec = {
        "filter": filter_name, "n_frames": n,
        "files": [os.path.relpath(f, out_dir) for f in fr["files"]],
        "paths": fr["paths"], "levels": fr["levels"], "weights": fr["weights"],
        "restored": fr["restored"], "shape": fr["shape"],
        "coverage_box": list(box) if box is not None else None,
        "scale": float(scale), "scale_from": str(scale_from),
        "sky": float(np.median(img)),
    }
    (out_dir / GROUP_FILE).write_text(json.dumps(rec, indent=1))
    return rec


def split_halves(k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint index sets of floor(k/2) frames each; an odd frame sits out."""
    perm = rng.permutation(k)
    h = k // 2
    return perm[:h], perm[h:2 * h]


def combine_half(cube: np.ndarray, weights: np.ndarray, levels: np.ndarray,
                 restore: Optional[float] = None) -> np.ndarray:
    """The stacker's SIGMA_CLIP_FWHM combine of a (n, h, w) patch cube, sky put
    back: *restore* when given, else the weighted mean of *levels* (the
    stacker's own rule for a lone stack)."""
    from stacking import stacker
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = stacker._combine_tile(cube, stacker.StackMethod.SIGMA_CLIP_FWHM, w, SIGMA)
    if restore is None:
        restore = float(np.average(np.asarray(levels, dtype=np.float64), weights=w))
    return out + float(restore)


class ResplitDataset:
    """Map-style dataset (a __len__ and a __getitem__, all a DataLoader asks of
    one) of (input, target) patch pairs, each from a fresh half-split.

    *group_dirs*: directories written by build_group. Groups are drawn
    uniformly, as N2NDataset's one-pair-per-group pool was.
    """

    def __init__(self, group_dirs: list, patch_size: int = 512, pairs_per_epoch: int = 2000,
                 source_bias: float = 0.7, seed: Optional[int] = None):
        from nn.trainer import N2NDataset
        self.patch_size = int(patch_size)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._rng_pid: Optional[int] = None
        self.groups: list[dict] = []
        for d in group_dirs:
            d = Path(d)
            rec = json.loads((d / GROUP_FILE).read_text())
            full = np.load(d / ALL_STACK)
            # The sampler: N2NDataset's CDF and fill map on the full stack; its
            # copy of the normalised frame is dropped once they exist.
            sampler = N2NDataset([full], patch_size=self.patch_size, pairs_per_epoch=1,
                                 source_bias=source_bias, seed=seed)
            sampler.frames = []
            del full
            h, w = rec["shape"]
            box = rec.get("coverage_box") or [0, h, 0, w]
            self.groups.append({
                "dir": str(d), "key": d.name,
                "mm": [np.load(d / f, mmap_mode="r") for f in rec["files"]],
                "weights": np.asarray(rec["weights"], dtype=np.float64),
                "levels": np.asarray(rec["levels"], dtype=np.float64),
                "bg": np.load(d / BACKGROUND, mmap_mode="r"),
                "scale": float(rec["scale"]), "shape": (h, w), "box": box,
                "restored": float(rec["restored"]),
                "sampler": sampler,
            })
        if not self.groups:
            raise ValueError("no groups")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _worker_rng(self) -> None:
        pid = os.getpid()
        if self._rng_pid == pid:
            return
        import torch
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        self.rng = np.random.default_rng(None if self._seed is None else self._seed + 7919 * (wid + 1))
        for g in self.groups:
            g["sampler"].rng = np.random.default_rng(
                None if self._seed is None else self._seed + 104729 * (wid + 1) + zlib.crc32(g["key"].encode()) % 9973)
            g["sampler"]._rng_pid = pid
        self._rng_pid = pid

    def origin(self, g: dict) -> tuple[int, int]:
        h, w = g["shape"]
        ps = self.patch_size
        y0, x0 = g["sampler"]._sample_origin(0, h, w)
        by0, by1, bx0, bx1 = g["box"]
        y0 = int(min(max(y0, by0), max(by0, by1 - ps)))
        x0 = int(min(max(x0, bx0), max(bx0, bx1 - ps)))
        return y0, x0

    def pair(self, g: dict, y0: int, x0: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """Two normalised half-stack patches at (y0, x0) from one fresh split."""
        ps = self.patch_size
        a, b = split_halves(len(g["mm"]), rng)
        bg = np.asarray(g["bg"][y0:y0 + ps, x0:x0 + ps], dtype=np.float32)
        out = []
        for idx in (a, b):
            cube = np.stack([np.asarray(g["mm"][i][y0:y0 + ps, x0:x0 + ps], dtype=np.float32) for i in idx])
            half = combine_half(cube, g["weights"][idx], g["levels"][idx], g["restored"]).astype(np.float32)
            half = np.where(np.isfinite(half), half, 0.0)
            out.append(np.arcsinh((half - bg) / g["scale"]).astype(np.float32))
        return out[0], out[1]

    def __getitem__(self, _idx: int):
        import torch
        self._worker_rng()
        g = self.groups[int(self.rng.integers(len(self.groups)))]
        y0, x0 = self.origin(g)
        pa, pb = self.pair(g, y0, x0, self.rng)
        if self.rng.integers(2):
            pa, pb = np.fliplr(pa), np.fliplr(pb)
        if self.rng.integers(2):
            pa, pb = np.flipud(pa), np.flipud(pb)
        k = int(self.rng.integers(4))
        if k:
            pa, pb = np.rot90(pa, k), np.rot90(pb, k)
        return (torch.from_numpy(pa.copy()[None].astype(np.float32)),
                torch.from_numpy(pb.copy()[None].astype(np.float32)))
