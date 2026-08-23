#!/usr/bin/env python3
"""Denoise several filters jointly: one model, C channels in, C channels out.

    python scripts/n2n_multichannel.py stacks --train sh2-92 --filters Ha,O-III
    python scripts/n2n_multichannel.py train  --train sh2-92 --test ic1396

Every model in this manual so far denoises one filter at a time. That makes
per-channel retention differences structurally independent, and those
differences are colour casts: L/R/G/B kept at 76/36/51/41% on ngc5907 (step 21),
S-II at 0.556 against Ha's 0.887 on NGC 6888. A single-channel model cannot
represent "this structure is in every channel, so it is real"; a joint one can.

**The channels must share one reference.** Pairs are still two half-stacks of the
same scene, but now each half is a C-channel tensor, so all C filters have to
land on the same pixel grid — otherwise the network is shown channels offset
from each other, which is the misregistration failure of step 18 wearing a new
hat. The ladder's per-group references are *not* good enough here: measured on
ngc5907, its filters sat up to (44, -56) px apart.

**Normalisation is per channel.** Each filter is divided by its own sky sigma, as
`denoiser.normalise` does, because narrowband and broadband sky levels differ by
more than an order of magnitude and a shared scale would hand the network one
channel of noise and one of nothing. The scales are kept so the output can be
returned to ADU per channel — which is what makes the colour ratio recoverable.

What this is expected to fix: cross-channel *consistency*. It is not expected to
fix the band limit (step 29) — that is a spatial-frequency property and adding
channels does not change it.
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

OUT = Path(_root) / "local" / "n2n_multichannel"
TEST_DIR = Path(_root) / "local" / "n2n_lrgb_render"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_root, rel))
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


hr = _load("hr", "scripts/n2n_holdout_run.py")
xc = _load("xc", "scripts/n2n_extended_check.py")


def log(m: str = "") -> None:
    print(m, flush=True)


def stack_file(dso, filt, kind, k):
    safe = filt.replace(" ", "_").replace("/", "_")
    return OUT / "stacks" / f"{dso}__{safe}_{kind}{k}.npy"


def build(args) -> int:
    """Split stacks for every filter of one target, all on ONE reference."""
    from nn import stacks

    (OUT / "stacks").mkdir(parents=True, exist_ok=True)
    machine = config.data().get("machine", {}).get(socket.gethostname()) or {}
    subs = Path(args.root) if args.root else Path(machine["subs_dir"])
    # Archive lights need their own epoch's calibration; config resolves the
    # current one, and 2026 masters at -10C on 2024 lights at -20C is worse
    # than none. Same helper the render path uses.
    rnd = _load("rnd", "scripts/n2n_lrgb_render.py") if args.cal_root else None
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    idx = stacks.index_frames(subs, filters, args.exptime)

    # One reference for every filter, taken from the deepest — the same rule
    # color_process uses, and the reason the channels can be stacked as a tensor.
    ref_filt = max(filters, key=lambda f: len(idx.get((args.train, f), [])))
    ref_paths = idx.get((args.train, ref_filt), [])
    if not ref_paths:
        log(f"no {ref_filt} frames for {args.train}")
        return 1
    log(f"shared reference from {len(ref_paths)} {ref_filt} frames")
    ref = stacks.shared_reference_for(ref_paths, ref_filt,
                                      progress_cb=lambda m: log(f"  {m}"))
    if ref is None:
        log("no shared reference — refusing to build channels on separate grids")
        return 1

    meta = {"train": args.train, "filters": filters, "ref_filter": ref_filt,
            "exptime": args.exptime, "seed": args.seed, "channels": {}}
    for filt in filters:
        paths = idx.get((args.train, filt), [])
        train_per, val_per = hr.split_depths(len(paths))
        if train_per < 6:
            log(f"[{filt}] {len(paths)} frames — too thin, aborting")
            return 1
        files = [stack_file(args.train, filt, k, i)
                 for k in ("train", "val") for i in (0, 1)]
        if all(f.exists() for f in files) and not args.force:
            log(f"[{filt}] cached 2x{train_per} + 2x{val_per}")
            meta["channels"][filt] = {"train_per": train_per, "val_per": val_per,
                                      "files": [str(f) for f in files]}
            continue
        log(f"[{filt}] {len(paths)} frames -> 2x{train_per} + 2x{val_per} val")
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(paths))

        def chunk(lo, n):
            return [paths[i] for i in order[lo:lo + n]]

        if rnd is not None:
            from stacking import stacker
            bias, dark, flat = rnd.archive_calibration(
                Path(args.cal_root), filt, args.exptime, args.cal_epoch)
            log(f"[{filt}] calibration {len(bias)} bias / {len(dark)} dark / "
                f"{len(flat)} flat")

            def _st(ps):
                img, _ = stacker.stack(
                    list(ps), method=stacker.StackMethod.SIGMA_CLIP_FWHM,
                    shared_reference=ref, bias_paths=bias or None,
                    dark_paths=dark or None, flat_paths=flat or None,
                    register=True, progress_cb=lambda m: None)
                return img
        else:
            def _st(ps):
                return stacks.stack_paths(ps, filt, progress_cb=lambda m: None,
                                          shared_reference=ref)
        made = [_st(chunk(k * train_per, train_per)) for k in range(2)]
        vmade = [_st(chunk(2 * train_per + k * val_per, val_per)) for k in range(2)]
        if any(x is None for x in made + vmade):
            log(f"[{filt}] stacking failed")
            return 1
        made, vmade = stacks.crop_to_common(made), stacks.crop_to_common(vmade)
        dy, dx = hr._peak_offset(*(np.ascontiguousarray(s) for s in made))
        log(f"[{filt}] {time.time()-t0:.0f}s, pair offset ({dy},{dx}) px")
        if max(abs(dy), abs(dx)) > 2:
            log(f"[{filt}] *** pair misaligned — aborting ***")
            return 1
        for f, arr in zip(files, made + vmade):
            np.save(f, arr.astype(np.float32))
        meta["channels"][filt] = {"train_per": train_per, "val_per": val_per,
                                  "files": [str(f) for f in files]}
        del made, vmade
        gc.collect()

    (OUT / f"meta_{args.train}.json").write_text(json.dumps(meta, indent=2))
    log(f"\nwrote {OUT / f'meta_{args.train}.json'}")
    return 0


class MultiChannelPairs:
    """Patch pairs of shape (C, patch, patch) from two C-channel half-stacks."""

    def __init__(self, a, b, patch, pairs, seed, source_bias=0.7):
        from nn import denoiser

        self.patch = patch
        self.pairs = pairs
        self.rng = np.random.default_rng(seed)
        # Per channel, exactly as inference will do it.
        self.a = np.stack([denoiser.normalise(denoiser.subtract_background(x)[0])[0]
                           for x in a])
        self.b = np.stack([denoiser.normalise(denoiser.subtract_background(x)[0])[0]
                           for x in b])
        self.source_bias = source_bias
        # Sample where there is content, using the deepest-signal channel as the
        # guide so every channel is cropped at the same place.
        guide = self.a.mean(axis=0)
        self._cdf = xc.smoothed_sb(guide, 64)

    def __len__(self):
        return self.pairs

    def __getitem__(self, _):
        import torch

        c, h, w = self.a.shape
        ps = self.patch
        for _try in range(8):
            y = int(self.rng.integers(0, max(1, h - ps)))
            x = int(self.rng.integers(0, max(1, w - ps)))
            if self.rng.random() > self.source_bias:
                break
            if self._cdf[y:y + ps, x:x + ps].mean() > 0.15:
                break
        pa = self.a[:, y:y + ps, x:x + ps]
        pb = self.b[:, y:y + ps, x:x + ps]
        if self.rng.integers(2):
            pa, pb = pa[:, :, ::-1], pb[:, :, ::-1]
        if self.rng.integers(2):
            pa, pb = pa[:, ::-1, :], pb[:, ::-1, :]
        k = int(self.rng.integers(4))
        if k:
            pa, pb = np.rot90(pa, k, (1, 2)), np.rot90(pb, k, (1, 2))
        # Direction is symmetric in N2N; swapping halves the correlation between
        # consecutive draws for free.
        if self.rng.integers(2):
            pa, pb = pb, pa
        return (torch.from_numpy(np.ascontiguousarray(pa, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(pb, dtype=np.float32)))


def denoise_multi(arrs, model, device, tile=512, overlap=64):
    """Tiled inference over a C-channel frame, per-channel normalisation."""
    import torch

    from nn import denoiser

    subs, backs, scales, norms = [], [], [], []
    for a in arrs:
        s, bg = denoiser.subtract_background(a.astype(np.float32))
        n, sc = denoiser.normalise(s)
        subs.append(s); backs.append(bg); scales.append(sc); norms.append(n)
    stack = np.stack(norms)
    c, h, w = stack.shape
    out = np.zeros_like(stack, dtype=np.float64)
    wt = np.zeros((h, w), dtype=np.float64)

    def win1d(n):
        v = np.ones(n, dtype=np.float32)
        for i in range(overlap):
            t = 0.5 - 0.5 * np.cos(np.pi * i / overlap)
            v[i] = t
            v[n - overlap + i] = 1.0 - t
        return v

    win = np.outer(win1d(tile), win1d(tile))
    step = tile - overlap
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    model = model.to(device)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                patch = np.zeros((c, tile, tile), dtype=np.float32)
                patch[:, :y1 - y0, :x1 - x0] = stack[:, y0:y1, x0:x1]
                pred = model(torch.from_numpy(patch)[None].to(device))[0].cpu().numpy()
                out[:, y0:y1, x0:x1] += pred[:, :y1 - y0, :x1 - x0] * win[:y1 - y0, :x1 - x0]
                wt[y0:y1, x0:x1] += win[:y1 - y0, :x1 - x0]
    out /= np.maximum(wt, 1e-8)[None]
    return [denoiser.denormalise(out[i], scales[i]) + backs[i] for i in range(c)]


def train(args) -> int:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn.noise2noise_model import UNet

    meta = json.loads((OUT / f"meta_{args.train}.json").read_text())
    filters = meta["filters"]
    C = len(filters)
    log(f"{C} channels: {filters}")

    tr_a = [np.load(meta["channels"][f]["files"][0]) for f in filters]
    tr_b = [np.load(meta["channels"][f]["files"][1]) for f in filters]
    va_a = [np.load(meta["channels"][f]["files"][2]) for f in filters]
    va_b = [np.load(meta["channels"][f]["files"][3]) for f in filters]
    hh = min(x.shape[0] for x in tr_a + tr_b)
    ww = min(x.shape[1] for x in tr_a + tr_b)
    tr_a = [x[:hh, :ww] for x in tr_a]; tr_b = [x[:hh, :ww] for x in tr_b]
    hv = min(x.shape[0] for x in va_a + va_b)
    wv = min(x.shape[1] for x in va_a + va_b)
    va_a = [x[:hv, :wv] for x in va_a]; va_b = [x[:hv, :wv] for x in va_b]

    cfg = config.data().get("nn", {})
    patch = int(args.patch or cfg.get("patch_size", 256))
    batch = int(args.batch or cfg.get("batch_size", 8))
    pairs = int(args.pairs or cfg.get("pairs_per_epoch", 2000))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    ds = MultiChannelPairs(tr_a, tr_b, patch, pairs, args.seed)
    vs = MultiChannelPairs(va_a, va_b, patch, max(batch, pairs // 5), args.seed + 1)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.single:
        # Arm A: today's approach — one model, filters pooled as separate
        # groups, each denoised alone. Trained from the *same* stacks as the
        # joint arm so the only difference is whether the channels are seen
        # together, not which pixels went in.
        from nn.trainer import N2NDataset
        frames, gids, vframes, vgids = [], [], [], []
        for i, f in enumerate(filters):
            frames += [tr_a[i], tr_b[i]]; gids += [f, f]
            vframes += [va_a[i], va_b[i]]; vgids += [f + "|val", f + "|val"]
        ds = N2NDataset(frames, group_ids=gids, patch_size=patch,
                        pairs_per_epoch=pairs, seed=args.seed)
        vs = N2NDataset(vframes, group_ids=vgids, patch_size=patch,
                        pairs_per_epoch=max(batch, pairs // 5), seed=args.seed + 1)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
        vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)
        log(f"  single-channel arm: {len(ds._valid_pairs)} pairs over "
            f"{len(set(gids))} groups")
    model = UNet(residual="linear", in_ch=1 if args.single else C).to(dev)
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
            loss = crit(model(a), b)
            loss.backward()
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
    from nn import denoiser
    kind = "sc" if args.single else f"mc{C}"
    mp = Path(_root) / "local" / "models" / f"n2n_{kind}_{args.tag}_{args.exptime}s.pt"
    mp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_sd, "in_ch": 1 if args.single else C,
                "single": bool(args.single), "filters": filters,
                "epoch": best_ep, "loss": args.loss, "seed": args.seed,
                "patch_size": patch, "train_dso": args.train, "test_dso": args.test,
                "asinh_sigma_mult": denoiser.ASINH_SIGMA_MULT}, mp)
    log(f"  best val {best:.5f} at epoch {best_ep} -> {mp.name}")

    # Evaluate on the held-out target: per-channel retention and, the point of
    # the exercise, the spread between channels.
    raws, labels = [], []
    tdir = Path(args.test_dir) if args.test_dir else TEST_DIR
    for f in filters:
        p = tdir / f"{args.test}_{f}_raw.npy"
        if not p.exists():
            log(f"  no test stack {p.name} — skipping evaluation")
            return 0
        raws.append(np.load(p).astype(np.float32)); labels.append(f)
    hh = min(x.shape[0] for x in raws); ww = min(x.shape[1] for x in raws)
    raws = [x[:hh, :ww] for x in raws]
    if args.single:
        from nn import denoiser as _dn
        dens = [_dn.denoise_frame(r, model, device=dev) for r in raws]
    else:
        dens = denoise_multi(raws, model.cpu(), dev)

    EDGES = [-2, 0, 1, 2, 4, 8, 16, 32, 1e9]
    log(f"\nextended-flux retention on held-out {args.test}")
    res = {}
    for lab, r, d in zip(labels, raws, dens):
        res[lab] = xc.retention(r, d, EDGES, 64)
    hdr = "  " + "SB bin".ljust(12) + "".join(f"{l:>12s}" for l in labels)
    log(hdr); log("  " + "-" * (len(hdr) - 2))
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        cells = []
        for l in labels:
            b = res[l]["bins"][i]
            cells.append(f"{b['kept']:12.3f}" if b else f"{'-':>12s}")
        log("  " + lbl.ljust(12) + "".join(cells))
    log("\n  cross-channel spread (the colour-safety number)")
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        vals = [res[l]["bins"][i]["kept"] for l in labels if res[l]["bins"][i]]
        if len(vals) < 2:
            continue
        lbl = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        log(f"    {lbl:12s} spread {max(vals)-min(vals):.3f}")
    (OUT / f"results_{args.tag}.json").write_text(json.dumps(
        {"filters": filters, "retention": {k: v["bins"] for k, v in res.items()}},
        indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stacks", "train"))
    ap.add_argument("--train", default="sh2-92")
    ap.add_argument("--test", default="ic1396")
    ap.add_argument("--filters", default="Ha,O-III")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=("l1", "l2"), default="l2")
    ap.add_argument("--patch", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--root", default="", help="where LIGHT frames live")
    ap.add_argument("--cal-root", default="", help="explicit calibration tree")
    ap.add_argument("--cal-epoch", default="")
    ap.add_argument("--test-dir", default="", help="where the test stacks live")
    ap.add_argument("--tag", default="nb")
    ap.add_argument("--single", action="store_true",
                    help="train one filter at a time (pooled groups) instead of "
                         "jointly — the baseline arm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)
    OUT.mkdir(parents=True, exist_ok=True)
    return build(args) if args.stage == "stacks" else train(args)


if __name__ == "__main__":
    sys.exit(main())
