#!/usr/bin/env python3
"""Save sample N2N training pairs as side-by-side JPEGs for inspection.

Usage:
    python scripts/n2n_show_pairs.py <filter> <seconds> [--n 16] [--seed 42] [--register]

Examples:
    python scripts/n2n_show_pairs.py L 300
    python scripts/n2n_show_pairs.py L 300 --n 20 --seed 7
    python scripts/n2n_show_pairs.py L 300 --register

Each JPEG shows the input patch (left) and target patch (right) as the
model sees them — frame-level normalised, with augmentation applied.
Saved to local/pair_samples/{filter}_{seconds}s/pair_NNN.jpg.
With --register, frames are aligned before pairing (use to verify registration).
"""

import os
import sys
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import socket
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import config
from nn import denoiser, registration
from nn.trainer import N2NDataset


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/n2n_show_pairs.py <filter> <seconds> [--n N] [--seed S]")
        sys.exit(1)

    args = sys.argv[1:]
    n_pairs = 16
    seed = 42
    do_register = False
    clean_args = []
    i = 0
    while i < len(args):
        if args[i] == "--n" and i + 1 < len(args):
            n_pairs = int(args[i + 1]); i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--register":
            do_register = True; i += 1
        else:
            clean_args.append(args[i]); i += 1

    filter_name = clean_args[0].strip()
    exptime_s   = int(clean_args[1])

    cfg = config.data()
    hostname = socket.gethostname()
    machine_cfg = cfg.get("machine", {}).get(hostname)
    if machine_cfg is None:
        print(f"Error: no 'machine.{hostname}' entry in config_public.py")
        sys.exit(1)
    subs_dir = Path(machine_cfg["subs_dir"])

    print(f"Scanning for {filter_name} {exptime_s}s LIGHT frames…")
    frames, dso_names = denoiser.collect_all_frames(subs_dir, filter_name, exptime_s)
    print(f"Found {len(frames)} frames")

    if do_register:
        print("Registering frames…")
        frames = registration.register_frames(frames, dso_names, progress_cb=print)

    dataset = N2NDataset(frames, group_ids=dso_names, patch_size=256, pairs_per_epoch=n_pairs)
    dataset.rng = np.random.default_rng(seed)

    suffix = "_registered" if do_register else ""
    out_dir = Path(_root) / "local" / "pair_samples" / f"{filter_name}_{exptime_s}s{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {n_pairs} pair images to {out_dir}")
    for idx in range(n_pairs):
        inp, tgt = dataset[idx]
        pa = inp.numpy()[0]
        pb = tgt.numpy()[0]

        # Clip to a visible range (most pixels in [0,1], stars go higher)
        vmin = min(pa.min(), pb.min())
        vmax = np.percentile(np.concatenate([pa.ravel(), pb.ravel()]), 99.5)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor("#0d0d1a")
        for ax, data, label in zip(axes, [pa, pb], ["Input (frame i)", "Target (frame j)"]):
            ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(label, color="white", fontsize=11)
            ax.axis("off")
            ax.set_facecolor("#0d0d1a")
        fig.suptitle(f"{filter_name} {exptime_s}s  —  pair {idx + 1}/{n_pairs}  "
                     f"(seed={seed})", color="white", fontsize=10)
        fig.tight_layout()
        out_path = out_dir / f"pair_{idx + 1:03d}.jpg"
        fig.savefig(out_path, format="jpeg", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  {out_path.name}")

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
