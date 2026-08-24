#!/usr/bin/env python3
"""Does training on deeper stacks stop the denoiser eating faint nebulosity?

    python scripts/n2n_depth_ab.py --train sh2-92 --caps 22,0

Step 23 measured the denoiser deleting ic1396: only 25% of extended flux
survives at 1-2 sigma and 38% at 4-8, which is where that nebula lives. The
lever this tests comes from the depth measurement in step 21's follow-up — the
model removes ~3x the sky noise when inference depth matches training and
**17-26x** when the input is deeper than anything it trained on. Its shrinkage
is calibrated to the noise level it learned, and it over-smooths whenever the
input is cleaner than that.

Every narrowband model so far learned on 22-29 frame half-stacks. sh2-92 has 137
Ha and 193 O-III, enough for 51 and 72. If the prior really is depth-keyed, the
deeper-trained model should hold low-surface-brightness structure better.

**Depth is the only variable.** Both arms train on the same target, the same two
filters, the same seed, the same pooled architecture and loss; they differ only
in how many frames go into each half-stack. Training on one target and capping
the other arm — rather than comparing against the existing bubble-trained
pooledNB — is what keeps it that way. Two variables moving at once has inverted
a conclusion in this project three times.

Scoring uses the **existing ic1396 stacks** from the HOO render, so the numbers
are directly comparable to the pooledNB row already measured in step 23 rather
than to a freshly built and subtly different test set. ic1396 is held out of
training entirely; it is worth more as the failure case than as a third scene,
and step 22 showed extra scenes are a wash anyway.

The primary metric is extended-flux retention by surface brightness
(`n2n_extended_check`), not corr-against-ceiling — the whole point is the part
of the frame the ceiling metric averages over.
"""

import argparse
import gc
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs import config


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_root, path))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


hr = _load("hr", "scripts/n2n_holdout_run.py")
xc = _load("xc", "scripts/n2n_extended_check.py")

OUT = Path(_root) / "local" / "n2n_depth_ab"
TEST_DIR = Path(_root) / "local" / "n2n_lrgb_render"


def log(m: str = "") -> None:
    print(m, flush=True)


def stack_file(dso: str, filt: str, cap: int, kind: str, k: int) -> Path:
    safe = filt.replace(" ", "_").replace("/", "_")
    return OUT / "stacks" / f"{dso}__{safe}_cap{cap}_{kind}{k}.npy"


def build_group(dso: str, filt: str, cap: int, exptime: int, seed: int,
                subs_dir: Path) -> dict:
    """Two train + two val stacks for one (dso, filter) at a given depth cap."""
    from nn import stacks

    idx = stacks.index_frames(subs_dir, [filt], exptime)
    paths = idx.get((dso, filt))
    if not paths:
        log(f"  [{dso}|{filt}] no frames")
        return {}
    train_per, val_per = hr.split_depths(len(paths))
    if cap:
        train_per = min(train_per, cap)
        val_per = min(val_per, max(2, cap // 3))

    files = [stack_file(dso, filt, cap, k, i) for k in ("train", "val") for i in (0, 1)]
    if all(f.exists() for f in files):
        log(f"  [{dso}|{filt}] cap{cap}: cached 2x{train_per} + 2x{val_per}")
        return {"train_per": train_per, "val_per": val_per,
                "files": [str(f) for f in files]}

    log(f"  [{dso}|{filt}] cap{cap}: {len(paths)} frames -> 2x{train_per} train "
        f"+ 2x{val_per} val")
    t0 = time.time()
    # Same rng construction as everywhere else in this chain, so the frames in a
    # given chunk are reproducible and the two caps draw from the same order.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    ref = stacks.shared_reference_for(paths, filt, progress_cb=lambda m: log(f"    {m}"))
    if ref is None:
        log(f"  [{dso}|{filt}] no shared reference — skipped")
        return {}

    def chunk(lo, n):
        return [paths[i] for i in order[lo:lo + n]]

    metas = [{} for _ in range(4)]
    made = [stacks.stack_paths(chunk(k * train_per, train_per), filt,
                               progress_cb=lambda m: None, shared_reference=ref,
                               meta_out=metas[k]) for k in range(2)]
    vmade = [stacks.stack_paths(chunk(2 * train_per + k * val_per, val_per), filt,
                                progress_cb=lambda m: None, shared_reference=ref,
                                meta_out=metas[2 + k]) for k in range(2)]
    if any(x is None for x in made + vmade):
        log(f"  [{dso}|{filt}] stacking failed")
        return {}
    made, vmade = stacks.crop_to_common(made), stacks.crop_to_common(vmade)
    dy, dx = hr._peak_offset(*(np.ascontiguousarray(s) for s in made))
    acc = [int(m.get("n_frames", 0)) for m in metas]
    log(f"  [{dso}|{filt}] cap{cap}: {time.time()-t0:.0f}s, pair offset ({dy},{dx}) px, "
        f"kept {acc[0]},{acc[1]} of {train_per}")
    if max(abs(dy), abs(dx)) > 2:
        log(f"  [{dso}|{filt}] *** pair misaligned — refusing to train on it ***")
        return {}
    (OUT / "stacks").mkdir(parents=True, exist_ok=True)
    for f, arr in zip(files, made + vmade):
        np.save(f, arr.astype(np.float32))
    del made, vmade
    gc.collect()
    return {"train_per": train_per, "val_per": val_per, "accepted": acc,
            "offset": [int(dy), int(dx)], "files": [str(f) for f in files]}


def train_arm(groups: dict, tag: str, args) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn import denoiser
    from nn.noise2noise_model import UNet
    from nn.trainer import N2NDataset

    tseed = args.seed if args.train_seed is None else args.train_seed
    torch.manual_seed(tseed)
    np.random.seed(tseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = config.data().get("nn", {})
    # Overridable so a patch-size experiment does not require editing config,
    # which every other path also reads.
    patch = int(args.patch or cfg.get("patch_size", 256))
    batch = int(args.batch or cfg.get("batch_size", 8))
    pairs = args.pairs or int(cfg.get("pairs_per_epoch", 2000))

    tr, trg, va, vag = [], [], [], []
    for key, g in groups.items():
        fs = [Path(p) for p in g["files"]]
        tr += [np.load(f) for f in fs[:2]]
        trg += [key] * 2
        va += [np.load(f) for f in fs[2:]]
        vag += [f"{key}|val"] * 2

    # source_bias weights patch origins toward detected sources. At the default
    # 0.7, 70% of training patches are drawn near a source — so a star-dense
    # field feeds the network mostly star patches and a sparse one feeds it
    # mostly extended emission. Suspected cause of bubble beating sh2-92: 207
    # stars against 747 in the same window, with identical extended power.
    ds = N2NDataset(tr, group_ids=trg, patch_size=patch, pairs_per_epoch=pairs,
                    source_bias=args.source_bias, seed=tseed)
    vs = N2NDataset(va, group_ids=vag, patch_size=patch,
                    pairs_per_epoch=max(batch, pairs // 5),
                    source_bias=args.source_bias, seed=tseed + 1)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)
    log(f"  {len(ds._valid_pairs)} pairs over {len(set(trg))} groups, "
        f"{pairs} draws/epoch")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feats = tuple(int(f) for f in args.features.split(",")) if args.features         else (32, 64, 128, 256)
    model = UNet(residual="linear", features=feats).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.loss == "msl2":
        # Headroom-weighted multi-scale L2, lab manual step 32/34. Raises the
        # price of mid-band (16-256 px) error, where the Wiener bound showed
        # +0.07..0.27 of claimable signal, without licensing fine-band error.
        from nn.losses import MultiScaleL2
        crit = MultiScaleL2().to(dev)
    else:
        crit = nn.MSELoss() if args.loss == "l2" else nn.L1Loss()

    # Per-epoch curves. Every experiment trained through this script for a week
    # left nothing but six console lines each; the epoch-plateau finding (steps
    # 31/33) was invisible for exactly that reason.
    writer = None
    try:
        import shutil

        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(_root) / "local" / "runs" / f"depth_{tag}_{args.exptime}s"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=str(log_dir))
        log(f"  tensorboard: {log_dir}")
    except Exception as exc:
        log(f"  no tensorboard ({exc})")

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
        if writer is not None:
            writer.add_scalars("loss", {"train": tot / max(len(dl), 1), "val": v}, ep)
            writer.flush()
        if ep % 10 == 0 or ep == args.epochs:
            log(f"  [{tag}] epoch {ep:3d} train={tot/max(len(dl),1):.5f} "
                f"val={v:.5f} best={best:.5f} ({time.time()-t0:.0f}s)")

    if writer is not None:
        writer.close()
    path = Path(_root) / "local" / "models" / f"n2n_depth_{tag}_{args.exptime}s.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_sd, "filter": tag, "epoch": best_ep,
                "asinh_sigma_mult": denoiser.ASINH_SIGMA_MULT, "loss": args.loss,
                "train_dso": args.train, "test_dso": args.test, "seed": args.seed,
                "groups": sorted(groups), "epochs": args.epochs,
                "source_bias": args.source_bias, "train_seed": tseed,
                "patch_size": patch, "batch_size": batch, "features": feats,
                "train_depth": {k: g["train_per"] for k, g in groups.items()}}, path)
    log(f"  best val {best:.5f} at epoch {best_ep} -> {path.name}")
    del tr, va, ds, vs
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return path


def evaluate(model_path: Path, args) -> dict:
    import torch

    from nn import denoiser
    from nn.noise2noise_model import UNet

    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    # Rebuild the architecture the checkpoint was trained with. This was the
    # one loader not taught about `features` when 5-level support landed, and
    # it crashed the l5 seed-0 eval after a full training run — the checkpoint
    # itself was fine.
    model = UNet(residual="linear",
                 features=tuple(ck.get("features") or (32, 64, 128, 256)))
    model.load_state_dict(ck["model_state"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    EDGES = [-2, 0, 1, 2, 4, 8, 16, 32, 1e9]
    out = {}
    for filt in [f.strip() for f in args.filters.split(",")]:
        raw_p = TEST_DIR / f"{args.test}_{filt}_raw.npy"
        if not raw_p.exists():
            log(f"  no test stack {raw_p.name}")
            continue
        raw = np.load(raw_p).astype(np.float32)
        den = denoiser.denoise_frame(raw, model, device=dev)
        r = xc.retention(raw, den, EDGES, 64)
        srv = hr.source_survival(raw, den)
        out[filt] = {"retention": r, "sources": srv}
        del raw, den
        gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="sh2-92",
                    help="comma list; groups are pooled across every (dso, filter)")
    ap.add_argument("--tag", default="",
                    help="prefix for model and results filenames; keeps runs "
                         "from overwriting each other")
    ap.add_argument("--test", default="ic1396")
    ap.add_argument("--filters", default="Ha,O-III")
    ap.add_argument("--caps", default="22,0",
                    help="comma list of per-stack frame caps; 0 = natural depth")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the frame permutation, so it fixes WHICH frames "
                         "land in each half-stack")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="seeds patch sampling and weight init only, leaving the "
                         "stacks cached. Vary this to measure run-to-run spread "
                         "without re-stacking; the manual records patch selection "
                         "as the dominant source of variance (step 15).")
    ap.add_argument("--loss", choices=("l1", "l2", "msl2"), default="l2")
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--features", default="",
                    help='UNet feature widths, e.g. "32,64,128,256,256" for a '
                         "fifth level (~2x the receptive field). Stored in the "
                         "checkpoint so loaders can rebuild the right shape.")
    ap.add_argument("--patch", type=int, default=0,
                    help="training patch size; 0 uses cfg['nn'] (256). The "
                         "receptive field is ~140 px, so only 21%% of a 256 "
                         "patch has full context against 53%% of a 512 one.")
    ap.add_argument("--batch", type=int, default=0,
                    help="batch size; 0 uses cfg['nn'] (8)")
    ap.add_argument("--source-bias", type=float, default=0.7,
                    help="fraction of patches drawn toward sources; 0 = uniform")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)

    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    subs = Path(machine["subs_dir"])
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    caps = [int(c) for c in args.caps.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.train.split(",") if t.strip()]
    want = len(filters) * len(targets)
    results = {}
    for cap in caps:
        tag = (args.tag + "_" if args.tag else "") + (f"cap{cap}" if cap else "full")
        if args.source_bias != 0.7:
            tag += f"_sb{args.source_bias:g}"
        if args.train_seed is not None:
            tag += f"_ts{args.train_seed}"
        if args.patch:
            tag += f"_p{args.patch}"
        if args.features:
            tag += f"_L{len(args.features.split(','))}"
        log(f"\n{'='*66}\narm {tag}: train {'+'.join(targets)}, test {args.test}\n{'='*66}")
        groups = {}
        for filt in filters:
            for dso in targets:
                g = build_group(dso, filt, cap, args.exptime, args.seed, subs)
                if g:
                    groups[f"{dso}|{filt}"] = g
        if len(groups) < want:
            log(f"arm {tag}: {len(groups)} of {want} groups — skipped")
            continue
        depths = {k: g["train_per"] for k, g in groups.items()}
        log(f"  depths {depths}")
        mp = train_arm(groups, tag, args)
        results[tag] = {"model": str(mp), "depths": depths,
                        "eval": evaluate(mp, args)}
        # Run-specific filename: a second run with different targets must not
        # silently replace the first run's numbers.
        rf = OUT / (f"results_{args.tag}.json" if args.tag else "results.json")
        rf.write_text(json.dumps(results, indent=2, default=str))

    # report
    log(f"\n{'='*66}\nextended-flux retention on held-out {args.test}\n{'='*66}")
    EDGES = [-2, 0, 1, 2, 4, 8, 16, 32, 1e9]
    for filt in filters:
        log(f"\n{filt}")
        hdr = "  " + "SB bin".ljust(12) + "".join(f"{t:>12s}" for t in results)
        log(hdr)
        log("  " + "-" * (len(hdr) - 2))
        for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
            cells = []
            for t in results:
                b = results[t]["eval"].get(filt, {}).get("retention", {}).get("bins", [None] * 8)[i]
                cells.append(f"{b['kept']:12.3f}" if b else f"{'-':>12s}")
            log("  " + lbl.ljust(12) + "".join(cells))
        for t in results:
            s = results[t]["eval"].get(filt, {}).get("sources", {})
            if s:
                log(f"  {t:10s} sources {s['n_denoised']}/{s['n_raw']} "
                    f"({100*s['source_survival']:.0f}%)  "
                    f"flux {s.get('flux_retained_median', float('nan')):.4f}")
    log(f"\nwrote {OUT / (f'results_{args.tag}.json' if args.tag else 'results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
