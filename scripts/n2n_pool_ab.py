#!/usr/bin/env python3
"""A/B: one model pooled across filters vs one model per filter.

    python scripts/n2n_pool_ab.py --train bubble --test sh2-92

Re-runs lab manual step 17 on the post-fix pipeline. That step measured pooling
as worse on every metric and concluded "keep per-filter models" — but it ran on
2026-08-14 through `build_split_stacks`, which registered each half of a training
pair to its own reference and left them 8-36 px apart (fixed 2026-08-17). Its
sibling verdict on L2 (step 13) has already reversed under the same fix, so step
17's conclusion cannot be trusted until it is re-measured.

Everything except pooling is held identical to the per-filter run this compares
against: same seed, same `split_depths`, same permutation per filter, so the
stacks are built from exactly the same frames. The test stacks are not rebuilt at
all — they are loaded from the per-filter run's output, so both arms are scored
on byte-identical data.

Pairs are still formed strictly within (dso, filter). Pooling shares *weights*,
never pairs: an Ha stack and an O-III stack of one target are two different
scenes, and pairing them would be the same class of error as misregistration.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "hr", os.path.join(_root, "scripts", "n2n_holdout_run.py"))
hr = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["x"]
_spec.loader.exec_module(hr)
sys.argv = _argv

from configs import config


def log(m=""):
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="bubble")
    ap.add_argument("--test", default="sh2-92")
    ap.add_argument("--filters", default="Ha,O-III")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=("l1", "l2"), default="l2")
    ap.add_argument("--baseline-dir", default="local/n2n_holdout/bubble2sh292fix",
                    help="per-filter run whose TEST stacks are reused")
    ap.add_argument("--tag", default="pooledNB")
    args = ap.parse_args()

    import socket
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn import denoiser, stacks
    from nn.noise2noise_model import UNet
    from nn.trainer import N2NDataset

    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    subs_dir = Path(machine["subs_dir"])
    nn_cfg = cfg.get("nn", {})
    patch = int(nn_cfg.get("patch_size", 256))
    batch = int(nn_cfg.get("batch_size", 8))
    pairs = int(nn_cfg.get("pairs_per_epoch", 2000))
    base = Path(args.baseline_dir)
    out_dir = Path(_root) / "local" / "n2n_holdout" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    import sep
    sep.set_extract_pixstack(5_000_000)

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    log(f"pooled A/B: train={args.train} test={args.test} filters={filters} "
        f"loss={args.loss} seed={args.seed}")
    log(f"reusing test stacks from {base}")

    tr_frames, tr_groups, va_frames, va_groups, depths = [], [], [], [], {}
    for filt in filters:
        idx = stacks.index_frames(subs_dir, [filt], args.exptime)
        tr_paths = idx[(args.train, filt)]
        tp, vp = hr.split_depths(len(tr_paths))
        depths[filt] = tp
        # Same rng construction as run_filter, so the permutation — and so the
        # exact frames in each stack — match the per-filter run.
        rng = np.random.default_rng(args.seed)
        o = rng.permutation(len(tr_paths))
        ref = stacks.shared_reference_for(tr_paths, filt, progress_cb=lambda m: log(m))
        t0 = time.time()

        def chunk(lo, n):
            return [tr_paths[i] for i in o[lo:lo + n]]

        made = [stacks.stack_paths(chunk(k * tp, tp), filt, progress_cb=lambda m: None,
                                   shared_reference=ref) for k in range(2)]
        vmade = [stacks.stack_paths(chunk(2 * tp + k * vp, vp), filt,
                                    progress_cb=lambda m: None, shared_reference=ref)
                 for k in range(2)]
        if any(x is None for x in made + vmade):
            log(f"  {filt}: stacking failed")
            return 1
        made, vmade = stacks.crop_to_common(made), stacks.crop_to_common(vmade)
        dy, dx = hr._peak_offset(*(np.ascontiguousarray(s) for s in made))
        log(f"  {filt}: 2x{tp} train + 2x{vp} val [{time.time()-t0:.0f}s], "
            f"pair offset dy={dy} dx={dx}")
        if max(abs(dy), abs(dx)) > 2:
            log("  *** misaligned — aborting ***")
            return 1
        tr_frames += made
        tr_groups += [f"{args.train}|{filt}"] * 2
        va_frames += vmade
        va_groups += [f"{args.train}|{filt}|val"] * 2

    log(f"pooled: {len(tr_frames)} train stacks over {len(set(tr_groups))} groups "
        f"({len(set(tr_groups))} pairs), {len(va_frames)} val stacks")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ds = N2NDataset(tr_frames, group_ids=tr_groups, patch_size=patch,
                    pairs_per_epoch=pairs, seed=args.seed)
    vs = N2NDataset(va_frames, group_ids=va_groups, patch_size=patch,
                    pairs_per_epoch=max(batch, pairs // 5), seed=args.seed + 1)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(residual="linear").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss() if args.loss == "l2" else nn.L1Loss()

    best, best_sd, best_ep = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for a, b in dl:
            a, b = a.to(dev), b.to(dev)
            opt.zero_grad()
            loss = crit(model(a), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sch.step()
        model.eval()
        v = 0.0
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
    torch.save({"model_state": best_sd, "filter": args.tag, "epoch": best_ep,
                "pooled_filters": filters, "train_dso": args.train,
                "test_dso": args.test, "seed": args.seed, "loss": args.loss}, mp)
    log(f"  best val {best:.5f} at epoch {best_ep} -> {mp.name}")

    log(f"\n{'=' * 60}\nPOOLED, scored on {args.test} (test stacks reused)\n{'=' * 60}")
    results = {}
    model = model.cpu()
    for filt in filters:
        raw = np.load(base / f"{filt}_{args.test}_raw.npy")
        other = np.load(base / f"{filt}_{args.test}_raw_b.npy")
        den = denoiser.denoise_frame(raw, model, device=dev)
        np.save(out_dir / f"{filt}_{args.test}_denoised.npy", den)
        chk = hr.collapse_check(raw, other, den)
        srv = hr.source_survival(raw, den)
        results[filt] = {**chk, **srv, "train_depth": depths[filt]}
        log(f"  {filt:6s} corr {chk['corr_in_out']:.4f}/{chk['ceiling']:.4f} "
            f"({100*chk['fraction_of_ceiling']:.0f}%) | sources "
            f"{100*srv['source_survival']:.0f}% | flux "
            f"{srv.get('flux_retained_median', float('nan')):.4f}")
        del raw, other, den

    import json
    (out_dir / "summary.json").write_text(json.dumps(
        {"pooled": results, "best_val": best, "best_epoch": best_ep,
         "loss": args.loss, "filters": filters}, indent=2, default=str))
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
