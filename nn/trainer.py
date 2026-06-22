"""Noise2Noise training: dataset, training loop, checkpoint management.

Reference:
    Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data",
    ICML 2018. https://arxiv.org/abs/1803.04189
"""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nn.noise2noise_model import UNet


class N2NDataset(Dataset):
    """Random patch pairs from a pool of noisy frames.

    Each sample is (input_patch, target_patch) drawn from two distinct
    frames of the *same* DSO. N2N requires pairs to be two noisy views of
    the same scene — cross-DSO pairs teach the model to predict average sky.
    """

    def __init__(
        self,
        frames: list[np.ndarray],
        group_ids: Optional[list[str]] = None,
        patch_size: int = 256,
        pairs_per_epoch: int = 2000,
    ):
        self.frames = frames
        self.patch_size = patch_size
        self.pairs_per_epoch = pairs_per_epoch
        self.rng = np.random.default_rng()

        # Precompute per-frame stats so patch normalization matches inference
        # (inference normalizes each frame globally, so training must too)
        self.frame_stats: list[tuple[float, float]] = []
        for f in frames:
            p1  = float(np.percentile(f, 1))
            p99 = float(np.percentile(f, 99))
            self.frame_stats.append((p1, max(p99 - p1, 1.0)))

        # Build (i, j) pairs restricted to within the same DSO group
        if group_ids is not None:
            groups: dict[str, list[int]] = {}
            for idx, gid in enumerate(group_ids):
                groups.setdefault(gid, []).append(idx)
            self._valid_pairs = [
                (i, j)
                for idxs in groups.values()
                for ii, i in enumerate(idxs)
                for j in idxs[ii + 1:]
            ]
        else:
            n = len(frames)
            self._valid_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if not self._valid_pairs:
            # Fallback: pair everything (shouldn't happen with valid input)
            n = len(frames)
            self._valid_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx = self.rng.integers(len(self._valid_pairs))
        i, j = self._valid_pairs[pair_idx]
        fa, fb = self.frames[i], self.frames[j]
        h, w = fa.shape
        ps = self.patch_size
        if h <= ps or w <= ps:
            y0, x0 = 0, 0
        else:
            y0 = int(self.rng.integers(0, h - ps))
            x0 = int(self.rng.integers(0, w - ps))
        pa = fa[y0:y0 + ps, x0:x0 + ps]
        pb = fb[y0:y0 + ps, x0:x0 + ps]

        # Normalise each patch by its own frame's global stats — same scheme
        # as inference so training and inference see identical value ranges
        p1a, scale_a = self.frame_stats[i]
        p1b, scale_b = self.frame_stats[j]
        pa = (pa - p1a) / scale_a
        pb = (pb - p1b) / scale_b

        # Augment: same random flip/rotation applied to both patches so they
        # stay aligned — 8 possible orientations (flips × 90° rotations)
        if self.rng.integers(2):
            pa, pb = np.fliplr(pa), np.fliplr(pb)
        if self.rng.integers(2):
            pa, pb = np.flipud(pa), np.flipud(pb)
        k = int(self.rng.integers(4))
        if k:
            pa, pb = np.rot90(pa, k), np.rot90(pb, k)

        inp = torch.from_numpy(pa.copy()[None].astype(np.float32))
        tgt = torch.from_numpy(pb.copy()[None].astype(np.float32))
        return inp, tgt


def _holdout_frames(
    frames: list[np.ndarray],
    group_ids: Optional[list[str]],
    val_frac: float,
    progress_cb: Callable[[str], None],
    min_val_groups: int = 2,
) -> tuple[list, Optional[list], list, Optional[list]]:
    """Split frames into disjoint train/val pools for a real held-out metric.

    When DSO group_ids are available, whole groups are held out so the model
    never trains on a validation scene — val then measures generalization to
    novel scenes, which is both the real inference condition and what exposes
    scene memorization. Holds out at least ``min_val_groups`` DSOs (a single
    held-out scene makes the val curve noisy), continuing past that until
    ``val_frac`` of the frames are held out, while always leaving at least one
    pairable group for training. Needs at least two pairable groups (>=2
    frames each); otherwise falls back to a disjoint per-frame holdout.

    Returns (train_frames, train_groups, val_frames, val_groups); the *_groups
    entries are None when group_ids is None, and the val_* lists are empty when
    no valid holdout could be carved out.
    """
    n = len(frames)
    rng = np.random.default_rng(0)

    def pick(indices: list[int]):
        gids = [group_ids[i] for i in indices] if group_ids is not None else None
        return [frames[i] for i in indices], gids

    if group_ids is not None:
        groups: dict[str, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            groups.setdefault(gid, []).append(idx)
        # Only groups with >=2 frames can form within-DSO N2N pairs
        pairable = [g for g, idxs in groups.items() if len(idxs) >= 2]
        if len(pairable) >= 2:
            target = val_frac * n
            val_groups: list[str] = []
            acc = 0
            for g in rng.permutation(pairable):
                # Hold out at least min_val_groups DSOs, then keep going until
                # val_frac of frames is reached; always leave >=1 pairable
                # group for training
                if len(val_groups) >= len(pairable) - 1:
                    break
                if len(val_groups) >= min_val_groups and acc >= target:
                    break
                val_groups.append(str(g))
                acc += len(groups[g])
            val_set = set(val_groups)
            va = [i for g in val_groups for i in groups[g]]
            tr = [i for g, idxs in groups.items() if g not in val_set for i in idxs]
            progress_cb(
                f"Held-out DSOs for validation: {', '.join(sorted(val_set))} "
                f"({len(va)}/{n} frames); training on {n - len(va)}"
            )
            return (*pick(tr), *pick(va))
        progress_cb("Fewer than two pairable DSOs — using a per-frame holdout for validation")

    # Per-frame holdout: still disjoint, but val frames share scenes with train
    perm = list(rng.permutation(n))
    n_val = int(round(val_frac * n))
    if n - n_val < 2:
        n_val = max(0, n - 2)
    return (*pick(perm[n_val:]), *pick(perm[:n_val]))


def train(
    filter_name: str,
    frames: list[np.ndarray],
    model_path: Path,
    group_ids: Optional[list[str]] = None,
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-3,
    patch_size: int = 256,
    pairs_per_epoch: int = 2000,
    val_dsos: int = 2,
    progress_cb: Callable[[str], None] = print,
) -> dict:
    """Train a Noise2Noise U-Net and save the best checkpoint.

    Returns a summary dict with final_loss, epochs, n_frames.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress_cb(f"[{filter_name}] Training on {len(frames)} frames — device: {device}")

    # Hold out frames the model never trains on, so val measures generalization.
    # The previous random_split was a no-op: N2NDataset ignores the sample index
    # and redraws a random pair on every access, so train/val drew from the same
    # frame pool and val never measured anything held out.
    tr_frames, tr_groups, va_frames, va_groups = _holdout_frames(
        frames, group_ids, val_frac=0.2, progress_cb=progress_cb, min_val_groups=val_dsos
    )
    train_ds = N2NDataset(tr_frames, group_ids=tr_groups, patch_size=patch_size, pairs_per_epoch=pairs_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))

    val_loader = None
    if va_frames:
        val_pairs = max(batch_size, int(0.2 * pairs_per_epoch))
        val_ds = N2NDataset(va_frames, group_ids=va_groups, patch_size=patch_size, pairs_per_epoch=val_pairs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))
    else:
        progress_cb(f"[{filter_name}] No held-out validation set possible — checkpointing on train loss")

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer — logs to local/runs/n2n_{filter}/
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(_root, "local", "runs", f"n2n_{filter_name.replace(' ', '_')}")
        writer = SummaryWriter(log_dir=log_dir)
        progress_cb(f"[{filter_name}] TensorBoard: tensorboard --logdir {os.path.join(_root, 'local', 'runs')}")
    except ImportError:
        progress_cb(f"[{filter_name}] TensorBoard not available (pip install tensorboard to enable)")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = None
        if val_loader is not None:
            model.eval()
            v = 0.0
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    v += criterion(model(inp), tgt).item()
            val_loss = v / len(val_loader)
        scheduler.step()

        # Select the checkpoint on held-out val when available, else train loss
        metric = val_loss if val_loss is not None else train_loss

        if writer is not None:
            scalars = {"train": train_loss}
            if val_loss is not None:
                scalars["val"] = val_loss
            writer.add_scalars("loss", scalars, epoch)

        if metric < best_val_loss:
            best_val_loss = metric
            torch.save({"model_state": model.state_dict(), "filter": filter_name, "epoch": epoch}, model_path)

        if epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            val_str = f"{val_loss:.6f}" if val_loss is not None else "n/a"
            progress_cb(
                f"[{filter_name}] epoch {epoch}/{epochs}  "
                f"train={train_loss:.6f}  val={val_str}  "
                f"best={best_val_loss:.6f}  elapsed={elapsed:.0f}s"
            )

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    progress_cb(f"[{filter_name}] Done — best val loss {best_val_loss:.6f}  ({elapsed:.0f}s)  saved → {model_path}")
    return {"final_loss": best_val_loss, "epochs": epochs, "n_frames": len(frames)}
