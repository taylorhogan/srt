#!/usr/bin/env python3
"""Train a Noise2Noise denoising model for a given filter and exposure time.

Usage:
    python scripts/n2n_train.py <filter> <seconds>

Examples:
    python scripts/n2n_train.py R 90
    python scripts/n2n_train.py R 300
    python scripts/n2n_train.py Ha 300

The subs root directory is read from configs/config_public.py under
cfg["machine"][hostname]["subs_dir"]. All LIGHT frames matching the given
FILTER header and EXPTIME (rounded to the nearest second) across every DSO
subdirectory are used for training.

The trained model is saved to local/models/n2n_{filter}_{seconds}s.pt.
"""

import os
import shutil
import socket
import sys
import time
from pathlib import Path

# Project root
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs import config
from nn import denoiser, registration, trainer


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/n2n_train.py <filter> <seconds>")
        print("  e.g. python scripts/n2n_train.py R 90")
        print("  e.g. python scripts/n2n_train.py Ha 300")
        sys.exit(1)

    filter_name = sys.argv[1].strip()
    try:
        exptime_s = int(sys.argv[2])
    except ValueError:
        print(f"Error: seconds must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    cfg = config.data()
    hostname = socket.gethostname()
    machine_cfg = cfg.get("machine", {}).get(hostname)
    if machine_cfg is None:
        print(f"Error: no 'machine.{hostname}' entry in config_public.py")
        print(f"  Known hosts: {list(cfg.get('machine', {}).keys())}")
        sys.exit(1)

    if "subs_dir" not in machine_cfg:
        print(f"Error: machine.{hostname} has no 'subs_dir' in config_public.py")
        print("  N2N reads frames from subs_dir; only hosts that define it can run this.")
        sys.exit(1)
    subs_dir = Path(machine_cfg["subs_dir"])
    if not subs_dir.exists():
        print(f"Error: subs_dir does not exist: {subs_dir}")
        sys.exit(1)

    # Clear old artefacts so stale results don't accumulate
    model_path_early = denoiser.get_model_path(filter_name, exptime_s)
    if model_path_early.exists():
        model_path_early.unlink()
        print(f"Cleared old model: {model_path_early.name}")
    denoised_dir = subs_dir / "denoised" / f"{filter_name}_{exptime_s}s"
    if denoised_dir.exists():
        shutil.rmtree(denoised_dir)
        print(f"Cleared old denoised frames: {denoised_dir}")
    tb_dir = Path(_root) / "local" / "runs" / f"n2n_{filter_name}_{exptime_s}s"
    if tb_dir.exists():
        shutil.rmtree(tb_dir)
        print(f"Cleared TensorBoard logs: {tb_dir}")

    nn_cfg = cfg.get("nn", {})
    min_frames   = int(nn_cfg.get("min_frames_to_train", 20))
    epochs       = int(nn_cfg.get("epochs", 60))
    batch_size   = int(nn_cfg.get("batch_size", 8))
    patch_size   = int(nn_cfg.get("patch_size", 256))
    pairs_per_ep = int(nn_cfg.get("pairs_per_epoch", 2000))
    val_dsos     = int(nn_cfg.get("val_dsos", 2))

    print(f"Scanning {subs_dir} for filter '{filter_name}' {exptime_s}s LIGHT frames…")
    t0 = time.time()
    frames, dso_names = denoiser.collect_all_frames(subs_dir, filter_name, exptime_s)
    dso_counts = {}
    for d in dso_names:
        dso_counts[d] = dso_counts.get(d, 0) + 1
    print(f"Found {len(frames)} frames across {len(dso_counts)} DSOs in {time.time() - t0:.1f}s")
    for dso, count in sorted(dso_counts.items()):
        print(f"  {dso}: {count} frames")

    if len(frames) < min_frames:
        print(f"Error: need at least {min_frames} frames to train, found {len(frames)}")
        sys.exit(1)

    print("Registering frames to correct for dithering…")
    frames = registration.register_frames(frames, dso_names, progress_cb=print)

    model_path = denoiser.get_model_path(filter_name, exptime_s)
    print(f"Model will be saved to: {model_path}")

    trainer.train(
        filter_name=f"{filter_name}_{exptime_s}s",
        frames=frames,
        model_path=model_path,
        group_ids=dso_names,
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        pairs_per_epoch=pairs_per_ep,
        val_dsos=val_dsos,
        progress_cb=print,
    )


if __name__ == "__main__":
    main()
