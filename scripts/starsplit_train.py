"""Learned star/nebula split: prepare, train, score, split (2026-11-05).

  prepare  --dsos ic410,melotte15,...   raw planes -> local/starsplit/prepared
  train    --name arm --seed 0 [--epochs 40 --pairs 2000 --batch 4]
  score    --name arm                   synthetic holdout metrics (known truth)
  split    --name arm --dso ic410 --filter Ha [--out dir]   starless + stars

Recipe and rationale: nn/starsplit_data. Entry points guarded: the stacker
spawns workers and a script without a guard re-runs in each.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from nn import starsplit, starsplit_data as D  # noqa: E402

RENDER_CACHE = Path(__file__).resolve().parent.parent / "local" / "n2n_lrgb_render"
NARROW = ("Ha", "O-III", "S-II")


def _planes_on_disk(dsos):
    for dso in dsos:
        for f in NARROW:
            p = RENDER_CACHE / f"{dso}_{f}_raw.npy"
            if p.exists():
                yield dso, f, p


def cmd_prepare(args):
    entries = []
    for dso, f, p in _planes_on_disk(args.dsos.split(",")):
        name = f"{dso}__{f}"
        if (D.PREPARED / f"{name}_bg.npy").exists() and not args.force:
            print(f"  {name}: prepared, skipped")
            continue
        t0 = time.time()
        print(f"{name}: {p.name}", flush=True)
        try:
            entries.append(D.prepare_plane(np.load(p, mmap_mode="r"), name))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
            continue
        print(f"  {time.time() - t0:.0f}s", flush=True)
        D.write_manifest(entries)
    D.write_manifest(entries)
    print(f"manifest: {D.PREPARED / 'manifest.json'}")


def _names(split: str):
    man = json.loads((D.PREPARED / "manifest.json").read_text())["planes"]
    names = sorted(man)
    dso = lambda n: n.split("__")[0]  # noqa: E731
    if split == "train":
        return [n for n in names if dso(n) not in D.HOLDOUT | D.VAL]
    if split == "val":
        return [n for n in names if dso(n) in D.VAL]
    return [n for n in names if dso(n) in D.HOLDOUT]


def _loss(out, tgt, lam: float):
    """L2 in asinh units. The deblur's Poisson chi^2 term is off by default
    here (lam 0): there the input and the target differ by noise, here they
    differ by the STARS, up to 1e6 ADU linear, and the chi^2 of an
    identity-initialised net on a saturated star is 1e5 (smoke run: train
    354, val 37946, all of it that term). The asinh L2 already judges a
    residual at a star site in sky sigmas."""
    import torch
    l2 = ((out - tgt) ** 2).mean()
    if lam <= 0:
        return l2
    lin_o, lin_t = torch.sinh(out), torch.sinh(tgt)
    chi = ((lin_o - lin_t) ** 2 / (lin_t.abs() + 1.0)).mean()
    return l2 + lam * chi


class _Pairs:
    """An IterableDataset over make_pair, so the synthesis (0.2 s a pair on
    one core: the star stamps and knots) runs in DataLoader workers."""

    def __init__(self, ds, n, seed):
        import torch
        self.ds, self.n, self.seed = ds, n, seed
        self._base = torch.utils.data.IterableDataset

    def loader(self, batch, workers):
        import torch
        ds, n, seed = self.ds, self.n, self.seed

        class It(torch.utils.data.IterableDataset):
            def __iter__(self):
                info = torch.utils.data.get_worker_info()
                wid = info.id if info else 0
                nw = info.num_workers if info else 1
                rng = np.random.default_rng([seed, wid])
                for _ in range(n // nw):
                    x, t, _ = ds.make_pair(rng)
                    yield torch.from_numpy(x), torch.from_numpy(t)
        return torch.utils.data.DataLoader(It(), batch_size=batch, num_workers=workers,
                                           persistent_workers=False)


def _batch(ds, rng, n):
    import torch
    xs, ts = [], []
    for _ in range(n):
        x, t, _ = ds.make_pair(rng)
        xs.append(x)
        ts.append(t)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ts))


def cmd_train(args):
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vrng = np.random.default_rng(10_000 + args.seed)
    train = D.StarSplitDataset(_names("train"))
    val = D.StarSplitDataset(_names("val") or _names("train"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = starsplit.build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * (args.pairs // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    vbatches = [_batch(val, vrng, args.batch) for _ in range(25)]
    out = D.MODELS / f"{args.name}.pt"
    hist, best = [], math.inf
    print(f"train {len(train.planes)} planes {[p['name'] for p in train.planes]}; val {[p['name'] for p in val.planes]}; {dev}")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr = 0.0
        n_steps = 0
        for x, t in _Pairs(train, args.pairs, args.seed * 1000 + ep).loader(args.batch, args.workers):
            x, t = x.to(dev), t.to(dev)
            loss = _loss(model(x), t, args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr += loss.item()
            n_steps += 1
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(_loss(model(x.to(dev)), t.to(dev), args.lam)) for x, t in vbatches]))
        n_steps = max(n_steps, 1)
        hist.append({"epoch": ep, "train": tr / n_steps, "val": vl, "t": time.time() - t0})
        flag = ""
        if vl < best:
            best = vl
            starsplit.save(model, out, {"name": args.name, "seed": args.seed, "epoch": ep, "val": vl,
                                        "features": [32, 64, 128, 256], "recipe": "starsplit_data v1"})
            flag = " *"
        starsplit.save(model, D.MODELS / f"{args.name}_last.pt", {"name": args.name, "seed": args.seed, "epoch": ep})
        (D.MODELS / f"{args.name}.json").write_text(json.dumps({"name": args.name, "seed": args.seed, "history": hist,
                                                                "best_val": best}, indent=2))
        print(f"epoch {ep:3d}  train {tr / n_steps:.4f}  val {vl:.4f}  {time.time() - t0:.0f}s{flag}", flush=True)
    print(f"best val {best:.4f} -> {out}")


def cmd_score(args):
    """Known-truth metrics on the holdout planes' synthetic patches: how much
    of each injected star's light is removed, by brightness bin; the RMS the
    starless output differs from the true background away from stars and
    under them; the residual at bright stars' positions in sigma."""
    import torch
    model, meta, dev = starsplit.load(D.MODELS / f"{args.name}.pt")
    ds = D.StarSplitDataset(_names("holdout"))
    rng = np.random.default_rng(777)
    bins = [(1.5, 5), (5, 20), (20, 100), (100, 1000), (1000, 1e9)]
    rem = {b: [] for b in bins}
    far_rms, under_rms, bright_res = [], [], []
    for _ in range(args.n):
        x, t, scale, truth = ds.make_pair(rng, with_truth=True)
        with torch.no_grad():
            y = model(torch.from_numpy(x)[None].to(dev))[0, 0].cpu().numpy()
        starless = np.sinh(y) * scale
        tgt = truth["target"]
        rms = truth["rms"]
        err = starless - tgt
        under = truth["stars"] > 2 * rms
        far_rms.append(float(np.sqrt(np.mean(err[~under] ** 2))) / rms if (~under).any() else np.nan)
        if under.any():
            under_rms.append(float(np.sqrt(np.mean(err[under] ** 2))) / rms)
        for (yy, xx, A) in truth["table"]:
            iy, ix = int(round(yy)), int(round(xx))
            if not (6 <= iy < err.shape[0] - 6 and 6 <= ix < err.shape[1] - 6):
                continue
            s_in = truth["stars"][iy - 6:iy + 7, ix - 6:ix + 7].sum()
            s_out = (truth["input"] - starless)[iy - 6:iy + 7, ix - 6:ix + 7].sum()
            frac = s_out / s_in if s_in > 0 else np.nan
            a_sig = A / rms
            for b in bins:
                if b[0] <= a_sig < b[1]:
                    rem[b].append(frac)
            if a_sig > 100:
                bright_res.append(float(np.abs(err[iy - 3:iy + 4, ix - 3:ix + 4]).max()) / rms)
    print(f"{args.name} (epoch {meta.get('epoch')}, val {meta.get('val', 0):.4f}) on {len(ds.planes)} holdout planes, {args.n} patches")
    print(f"  background preserved away from stars: rms err {np.nanmean(far_rms):.3f} sigma")
    print(f"  background under stars:                rms err {np.nanmean(under_rms):.3f} sigma")
    print(f"  residual at bright stars (>100 sigma):  max |err| {np.nanmedian(bright_res):.2f} sigma (median star)")
    for b in bins:
        v = np.array(rem[b], dtype=float)
        if v.size:
            print(f"  star light removed, peak {b[0]:>6g}-{b[1]:<6g} sigma: {np.nanmedian(v):.3f} (n={v.size})")


def cmd_split(args):
    from stacking import color_process
    model, meta, dev = starsplit.load(D.MODELS / f"{args.name}.pt")
    p = RENDER_CACHE / f"{args.dso}_{args.filter}_raw.npy"
    plane = np.load(p)
    t0 = time.time()
    starless, stars = starsplit.split_frame(plane, model, dev)
    print(f"{p.name}: split in {time.time() - t0:.0f}s")
    out = Path(args.out) if args.out else (D.MODELS.parent / "splits")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.dso}_{args.filter}_starless.npy", starless)
    np.save(out / f"{args.dso}_{args.filter}_stars.npy", stars)
    for tag, arr in (("input", plane), ("starless", starless), ("stars", stars)):
        a = np.nan_to_num(arr, nan=float(np.nanmedian(arr)))
        lo, hi = np.percentile(a, [30, 99.9]) if tag != "stars" else (0.0, np.percentile(a, 99.95))
        img = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        img = np.arcsinh(img / 0.03) / np.arcsinh(1 / 0.03)
        color_process.save_rgb(np.dstack([img] * 3), out / f"{args.dso}_{args.filter}_{tag}.jpg", max_px=3000)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--dsos", required=True); a.add_argument("--force", action="store_true")
    a = sub.add_parser("train"); a.add_argument("--name", required=True); a.add_argument("--seed", type=int, required=True)
    a.add_argument("--epochs", type=int, default=40); a.add_argument("--pairs", type=int, default=2000)
    a.add_argument("--batch", type=int, default=4); a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--lam", type=float, default=0.0); a.add_argument("--workers", type=int, default=6)
    a = sub.add_parser("score"); a.add_argument("--name", required=True); a.add_argument("--n", type=int, default=200)
    a = sub.add_parser("split"); a.add_argument("--name", required=True); a.add_argument("--dso", required=True)
    a.add_argument("--filter", required=True); a.add_argument("--out", default=None)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train, "score": cmd_score, "split": cmd_split}[args.cmd](args)


if __name__ == "__main__":
    main()
