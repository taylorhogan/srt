#!/usr/bin/env python3
"""Train one N2N stack model pooled across several filters.

Usage:
    python scripts/n2n_train_pooled.py <filters> <seconds> [--seed N]

    python scripts/n2n_train_pooled.py L,R,G,B 300 --seed 0

Why one model rather than one per filter: data thinness is what limits this
pipeline, and L+R+G+B at 300 s is 807 frames against R's 204. `normalise()`
divides by each frame's own robust sky sigma, so the sky-brightness differences
between passbands are removed before the network sees anything; what is left —
PSF shape, noise correlation, star profiles — comes from the optics and sensor
and is shared. The model learns the instrument, not the passband.

Unlike quartering the stacks (which also multiplied pairs, and measured worse
because shallower stacks teach over-aggressive shrinkage — lab manual step 16),
pooling adds *scenes* at the same stack depth.

Pairs are formed within (dso, filter), never across filters: an L stack and an R
stack of one target are two different scenes photometrically.

Narrowband is not pooled here by default. Ha/O-III sky is far darker, which
shifts the noise regime from sky-dominated toward read/dark-dominated — a change
in noise character normalisation cannot reconcile — and the content is extended
nebulosity rather than point sources.

Model goes to local/models/n2n_pooled_{filters}_{seconds}s.pt.
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
from nn import stacks, trainer


def main() -> int:
    argv = sys.argv[1:]
    # L2, not trainer.train's "l1" default. Step 13 measured L1 as better and
    # step 18 reversed that on the fixed pipeline, where L2 is now the default
    # everywhere else in the N2N chain (n2n_holdout_run, n2n_pool_ab). This
    # script never passed `loss` at all, so it silently kept training L1 after
    # the reversal — the only runner still doing so.
    loss = "l2"
    if "--loss" in argv:
        i = argv.index("--loss")
        if i + 1 >= len(argv) or argv[i + 1] not in ("l1", "l2"):
            print("Error: --loss needs l1 or l2")
            return 1
        loss = argv[i + 1]
        del argv[i:i + 2]
    seed = None
    if "--seed" in argv:
        i = argv.index("--seed")
        try:
            seed = int(argv[i + 1])
        except (IndexError, ValueError):
            print("Error: --seed needs an integer")
            return 1
        del argv[i:i + 2]

    # Whole DSOs removed from training, every filter of them.
    #
    # This has to be DSO-level, not group-level. Groups are keyed (dso, filter)
    # so that pairs never cross filters, but that means trainer's own holdout
    # would hold out "m92|R" while happily training on "m92|L" and "m92|B" —
    # the same stars, the same field, a different passband. Evaluating on m92
    # would then not be a held-out test at all.
    exclude: set[str] = set()
    if "--exclude" in argv:
        i = argv.index("--exclude")
        try:
            exclude = {d.strip().lower() for d in argv[i + 1].split(",") if d.strip()}
        except IndexError:
            print("Error: --exclude needs a dso list")
            return 1
        del argv[i:i + 2]

    if len(argv) < 2:
        print("Usage: python scripts/n2n_train_pooled.py <filters> <seconds> "
              "[--seed N] [--exclude dso,dso] [--loss l1|l2]")
        print("   e.g. python scripts/n2n_train_pooled.py L,R,G,B 300 --seed 0 "
              "--exclude m92")
        return 1
    filters = [f for f in argv[0].split(",") if f.strip()]
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

    print(f"Pooling filters {filters} at {exptime_s}s from {subs_dir}")
    t0 = time.time()
    st, groups = stacks.build_split_stacks(
        subs_dir, filters, exptime_s,
        exclude_dsos=exclude,
        seed=0 if seed is None else seed, progress_cb=print,
    )
    print(f"built in {time.time() - t0:.0f}s")

    n_groups = len(set(groups))
    if n_groups < 3:
        print(f"Error: need at least 3 (dso,filter) groups, got {n_groups}")
        return 1

    safe = "".join(f for f in "".join(filters) if f.isalnum())
    model_path = Path(_root) / "local" / "models" / f"n2n_pooled_{safe}_{exptime_s}s.pt"
    if model_path.exists():
        model_path.unlink()
        print(f"Cleared old model: {model_path.name}")
    print(f"Model will be saved to: {model_path}")

    trainer.train(
        filter_name=f"pooled_{safe}_{exptime_s}s",
        frames=st,
        model_path=model_path,
        group_ids=groups,
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        pairs_per_epoch=pairs_per_ep,
        val_dsos=2,
        loss=loss,
        seed=seed,
        progress_cb=print,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
