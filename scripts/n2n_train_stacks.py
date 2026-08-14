#!/usr/bin/env python3
"""Train a Noise2Noise model on split-half STACKS rather than on subs.

Usage:
    python scripts/n2n_train_stacks.py <filter> <seconds> [--seed N]

The sub-based chain (n2n_train.py) denoises every frame and then stacks. This
trains on pairs of disjoint half-stacks instead, so the model is applied once to
a finished stack. See nn/stacks.py for why that is better posed, and
docs/N2N_LAB_MANUAL.md for what the sub-based route measured.

Model goes to local/models/n2n_stack_{filter}_{seconds}s.pt — a separate name
from the sub-based model on purpose. The two are trained on different noise
levels and are not interchangeable; loading one where the other is expected
would be the same class of silent mismatch this pipeline has hit twice.
"""

import os
import socket
import sys
import time
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs import config
from nn import denoiser, stacks, trainer


def stack_model_path(filter_name: str, exptime_s: int) -> Path:
    safe = filter_name.replace(" ", "_").replace("/", "_")
    return Path(_root) / "local" / "models" / f"n2n_stack_{safe}_{exptime_s}s.pt"


def main() -> int:
    argv = sys.argv[1:]
    # --seed makes a run reproducible. Without it the result cannot be
    # recovered: on 2026-08-14 an unseeded run drew a model with val 0.7152
    # (checkpointed at epoch 7) where a seeded experiment on the same data and
    # config had reached 0.6957, and there was no way to get the better draw
    # back. Patch selection dominates that variance and comes from OS entropy
    # unless N2NDataset is given a seed — torch.manual_seed() does not touch it.
    seed = None
    if "--seed" in argv:
        i = argv.index("--seed")
        try:
            seed = int(argv[i + 1])
        except (IndexError, ValueError):
            print("Error: --seed needs an integer")
            return 1
        del argv[i:i + 2]

    # How many disjoint sub-stacks per DSO. 4 gives six pairs per target against
    # 2's one, and was tried as the cheap answer to data thinness. It measured
    # worse: real-source photometry barely moved (+0.03 at best) while the
    # response to an *injected* source collapsed, 1.07 -> 0.56 at SNR 5.7.
    # Quarter-stacks are sqrt(2) noisier than the full stack the model is
    # applied to, so they teach shrinkage calibrated for noisier data, and that
    # over-suppresses at inference. Depth match beat pair count. See
    # docs/N2N_LAB_MANUAL.md step 16.
    max_splits = 2
    if "--splits" in argv:
        i = argv.index("--splits")
        try:
            max_splits = int(argv[i + 1])
        except (IndexError, ValueError):
            print("Error: --splits needs an integer")
            return 1
        del argv[i:i + 2]

    if len(argv) < 2:
        print("Usage: python scripts/n2n_train_stacks.py <filter> <seconds> "
              "[--seed N] [--splits N]")
        return 1
    filter_name = argv[0].strip()
    try:
        exptime_s = int(argv[1])
    except ValueError:
        print(f"Error: seconds must be an integer, got '{argv[1]}'")
        return 1

    cfg = config.data()
    machine_cfg = cfg.get("machine", {}).get(socket.gethostname()) or {}
    if "subs_dir" not in machine_cfg:
        print(f"Error: machine.{socket.gethostname()} has no 'subs_dir'")
        return 1
    subs_dir = Path(machine_cfg["subs_dir"])

    nn_cfg = cfg.get("nn", {})
    epochs = int(nn_cfg.get("epochs", 60))
    batch_size = int(nn_cfg.get("batch_size", 8))
    patch_size = int(nn_cfg.get("patch_size", 256))
    pairs_per_ep = int(nn_cfg.get("pairs_per_epoch", 2000))

    print(f"Scanning {subs_dir} for '{filter_name}' {exptime_s}s LIGHT frames…")
    t0 = time.time()
    frames, dso_names = denoiser.collect_all_frames(subs_dir, filter_name, exptime_s)
    print(f"Found {len(frames)} frames in {time.time() - t0:.1f}s")
    if not frames:
        return 1

    print(f"Building split stacks (max {max_splits} per DSO)…")
    st, groups = stacks.build_split_stacks(frames, dso_names,
                                           max_splits=max_splits,
                                           seed=0 if seed is None else seed,
                                           progress_cb=print)
    del frames                       # ~50 GB; the stacks are all that is needed
    n_dso = len(set(groups))
    print(f"{len(st)} stacks across {n_dso} DSOs")
    if n_dso < 3:
        print("Error: need at least 3 DSOs (2 to train on, 1 held out)")
        return 1

    model_path = stack_model_path(filter_name, exptime_s)
    if model_path.exists():
        model_path.unlink()
        print(f"Cleared old model: {model_path.name}")
    print(f"Model will be saved to: {model_path}")

    # val_dsos=1: there is one pair per DSO here rather than hundreds of frames,
    # so holding out two of six costs a third of the training signal.
    trainer.train(
        filter_name=f"stack_{filter_name}_{exptime_s}s",
        frames=st,
        model_path=model_path,
        group_ids=groups,
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        pairs_per_epoch=pairs_per_ep,
        val_dsos=1,
        seed=seed,
        progress_cb=print,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
