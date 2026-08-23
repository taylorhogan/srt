#!/usr/bin/env python3
"""Train a pooled narrowband model from split stacks that already exist.

    python scripts/n2n_pooled_nb.py --test ngc6888 \
        --group "soapbubble|Ha=local/n2n_depth_ab/stacks/bubble__Ha_cap0" \
        --group "soapbubble|O-III=local/n2n_depth_ab/stacks/bubble__O-III_cap0" \
        --group "ngc7635|Ha=local/n2n_multichannel/stacks/ngc7635__Ha" \
        --group "ngc7635|O-III=local/n2n_multichannel/stacks/ngc7635__O-III" \
        --group "ngc7635|S-II=local/n2n_multichannel/stacks/ngc7635__S-II"

Pooling across *targets* whose stacks were built by different tools, in
different epochs, with different calibration. Each `--group` names a prefix and
the four files `{prefix}_train0.npy`, `_train1.npy`, `_val0.npy`, `_val1.npy`
that the ladder and multichannel builders both already write, so nothing is
re-stacked and each target keeps the calibration appropriate to when it was
shot.

Single-channel: pooling shares weights, and pairs are formed strictly within a
group, exactly as steps 19-22 established. Cross-filter alignment is therefore
*not* required here — each filter is denoised alone — which is what makes it
legitimate to mix stacks whose references were chosen independently.

**Depth spread is the thing to watch.** Step 25 measured that pooling groups of
unequal depth gives you the shallowest group's photometric bias rather than an
average: bubble at 22/29 pooled with sh2-92 at 51/72 read 1.074/1.090, worse
than either parent. This script prints the spread and warns past 2x, because the
warning is the only thing standing between a convenient pool and a silently
biased model.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs import config

TEST_DIR = Path(_root) / "local" / "n2n_lrgb_render"


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                    help='"label=prefix" — prefix+_train0/_train1/_val0/_val1.npy')
    ap.add_argument("--test", default="ngc6888")
    ap.add_argument("--test-filters", default="Ha,O-III,S-II")
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=("l1", "l2"), default="l2")
    ap.add_argument("--patch", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--tag", default="nb2")
    args = ap.parse_args()

    import sep
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn import denoiser
    from nn.noise2noise_model import UNet
    from nn.trainer import N2NDataset
    sep.set_extract_pixstack(5_000_000)

    frames, gids, vframes, vgids, depths = [], [], [], [], {}
    for spec in args.group:
        label, _, prefix = spec.partition("=")
        paths = [Path(_root) / f"{prefix}_{k}{i}.npy"
                 for k in ("train", "val") for i in (0, 1)]
        missing = [p for p in paths if not p.exists()]
        if missing:
            log(f"{label}: missing {missing[0].name}")
            return 1
        a, b, va, vb = (np.load(p) for p in paths)
        frames += [a, b]; gids += [label, label]
        vframes += [va, vb]; vgids += [f"{label}|val", f"{label}|val"]
        depths[label] = (a.shape, b.shape)
        log(f"{label:22s} {a.shape}")

    cfg = config.data().get("nn", {})
    patch = int(args.patch or cfg.get("patch_size", 256))
    batch = int(args.batch or cfg.get("batch_size", 8))
    pairs = int(args.pairs or cfg.get("pairs_per_epoch", 2000))
    log(f"\n{len(gids)//2} groups, patch {patch}, batch {batch}, {pairs} draws/epoch")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    ds = N2NDataset(frames, group_ids=gids, patch_size=patch,
                    pairs_per_epoch=pairs, seed=args.seed)
    vs = N2NDataset(vframes, group_ids=vgids, patch_size=patch,
                    pairs_per_epoch=max(batch, pairs // 5), seed=args.seed + 1)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)
    log(f"{len(ds._valid_pairs)} training pairs")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(residual="linear").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss() if args.loss == "l2" else nn.L1Loss()

    best, best_sd, best_ep = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for a, b in dl:
            a, b = a.to(dev), b.to(dev)
            opt.zero_grad()
            loss = crit(model(a), b); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sch.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for a, b in vl:
                v += crit(model(a.to(dev)), b.to(dev)).item()
        v /= max(len(vl), 1)
        if v < best:
            best, best_ep = v, ep
            best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs:
            log(f"  epoch {ep:3d} train={tot/max(len(dl),1):.5f} val={v:.5f} "
                f"best={best:.5f} ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_sd)
    mp = Path(_root) / "local" / "models" / f"n2n_{args.tag}_{args.exptime}s.pt"
    torch.save({"model_state": best_sd, "in_ch": 1, "epoch": best_ep,
                "loss": args.loss, "seed": args.seed, "patch_size": patch,
                "groups": [g for g in dict.fromkeys(gids)], "test_dso": args.test,
                "asinh_sigma_mult": denoiser.ASINH_SIGMA_MULT}, mp)
    log(f"\nbest val {best:.5f} at epoch {best_ep} -> {mp.name}")

    tdir = Path(args.test_dir) if args.test_dir else TEST_DIR
    labels = [f.strip() for f in args.test_filters.split(",") if f.strip()]
    raws, keep = [], []
    for f in labels:
        p = tdir / f"{args.test}_{f}_raw.npy"
        if p.exists():
            raws.append(np.load(p).astype(np.float32)); keep.append(f)
    if not raws:
        log("no test stacks — trained only")
        return 0
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "xc", os.path.join(_root, "scripts", "n2n_extended_check.py"))
    xc = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]; spec.loader.exec_module(xc); sys.argv = argv

    EDGES = [-2, 0, 1, 2, 4, 8, 16, 32, 1e9]
    res = {f: xc.retention(r, denoiser.denoise_frame(r, model, device=dev), EDGES, 64)
           for f, r in zip(keep, raws)}
    log(f"\nextended-flux retention on held-out {args.test}")
    hdr = "  " + "SB bin".ljust(12) + "".join(f"{l:>12s}" for l in keep)
    log(hdr); log("  " + "-" * (len(hdr) - 2))
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        cells = [f"{res[l]['bins'][i]['kept']:12.3f}" if res[l]["bins"][i]
                 else f"{'-':>12s}" for l in keep]
        log("  " + lbl.ljust(12) + "".join(cells))
    log("\n  cross-channel spread")
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        vals = [res[l]["bins"][i]["kept"] for l in keep if res[l]["bins"][i]]
        if len(vals) < 2:
            continue
        lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        log(f"    {lbl:12s} {max(vals)-min(vals):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
