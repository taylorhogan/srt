#!/usr/bin/env python3
"""Phase 1/2 of docs/LEARNED_DEBLUR_PLAN.md: prepare the stacks, train the model.

    python scripts/deconv_train.py prepare [--dsos a,b] [--domains raw,den] [--halves train0,train1]
    python scripts/deconv_train.py train --name NAME [--domain raw|den|both] [--seed N] [--epochs 60]
    python scripts/deconv_train.py smoke          # one short epoch end to end
    python scripts/deconv_train.py list

`prepare` walks the ladder manifest (local/n2n_ladder/manifest.json; every
usable group's two training half-stacks) and writes, for each half in each
domain, the background-subtracted stack and its kernel sidecar under
local/deconv/prepared/ (nn/deblur_data.prepare_stack). The denoised domain
is the raw half through the broadband model, or the narrowband one for
Ha/O-III/S-II, exactly as the morning render does it. HOLDOUT targets are
prepared too -- they are the test set -- and flagged in the manifest so
`train` never sees them. ~1.5 min and 0.5 GB per half per domain.

`train` fits nn/deblur_model on the non-holdout stacks of the chosen domain
(the kernel it learns to remove is the measured one's core, --taper; see
nn/deblur_data.taper_kernel for why the far wing is left as scene),
epoch selection on VAL_DSOS (whole targets, so no scene leaks between train
and val), the loss L2 in asinh space plus a Poisson-weighted linear chi^2
(Phase 0 found the linear term helps extended structure even though it does
not fix flux; flux is fixed by the head). Checkpoint: local/deconv/models/NAME.pt
with the training recipe in its meta. Every run must be seeded.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from nn import deblur_data as dd

BASE = Path(_root) / "local" / "deconv"
MODELS = BASE / "models"
LADDER_MANIFEST = Path(_root) / "local" / "n2n_ladder" / "manifest.json"
HOLDOUT = {"ngc5907", "ngc6946", "ngc7320"}        # the plan's test targets
VAL_DSOS = {"m13", "abell2218"}                    # epoch selection; never scored
NARROWBAND = {"Ha", "O-III", "S-II"}
DOMAIN_MODELS = {"broadband": "local/models/n2n_ladder_groups_galaxies_300s.pt",
                 "narrowband": "local/models/n2n_ladder_groups_narrowband_up_300s.pt"}


def log(msg: str) -> None:
    print(msg, flush=True)


def read_manifest() -> dict:
    p = dd.PREPARED / "manifest.json"
    return json.load(open(p)) if p.exists() else {"stacks": {}}


def write_manifest(man: dict) -> None:
    dd.PREPARED.mkdir(parents=True, exist_ok=True)
    with open(dd.PREPARED / "manifest.json", "w") as f:
        json.dump(man, f, indent=1)


# ---------------------------------------------------------------- prepare ----
def cmd_prepare(a):
    import torch
    from nn import denoiser
    lad = json.load(open(LADDER_MANIFEST))
    only = {d.strip().lower() for d in a.dsos.split(",") if d.strip()}
    domains = [d.strip() for d in a.domains.split(",") if d.strip()]
    halves = [h.strip() for h in a.halves.split(",") if h.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    man = read_manifest()
    n_done = 0
    for key, g in sorted(lad["groups"].items()):
        if not g.get("usable", True):
            continue
        dso, filt = g["dso"], g["filter"]
        if only and dso.lower() not in only:
            continue
        files = {Path(f).stem.rsplit("_", 1)[1]: Path(_root) / f for f in g["files"]}
        for half in halves:
            src_path = files.get(half)
            if src_path is None or not src_path.exists():
                log(f"[{key}] no {half} stack — skipped")
                continue
            src = None
            for domain in domains:
                name = f"{dso}__{filt}_{half}_{domain}"
                if (dd.PREPARED / f"{name}.psf.npz").exists() and not a.force:
                    log(f"[{name}] cached")
                    continue
                if src is None:
                    src = np.load(src_path)
                model = None
                if domain == "den":
                    mdom = "narrowband" if filt in NARROWBAND else "broadband"
                    mp = os.path.join(_root, DOMAIN_MODELS[mdom])
                    if mp not in loaded:
                        loaded[mp] = denoiser.load_model(Path(mp))
                    model = loaded[mp]
                t0 = time.time()
                log(f"[{name}] preparing ({'denoise + ' if model else ''}background + kernel)")
                extra = None
                if domain == "den":
                    # the raw half's plateaus: a denoised stack hides its ceiling
                    raw_side = dd.PREPARED / f"{dso}__{filt}_{half}_raw.psf.npz"
                    if raw_side.exists():
                        extra = dd.load_sidecar(raw_side)["flat"]
                side = dd.prepare_stack(src, name, denoise_model=model, device=dev, log=log, extra_flat=extra)
                if side is None:
                    man["stacks"][name] = {"dso": dso, "filter": filt, "half": half, "domain": domain,
                                           "usable": False, "holdout": dso.lower() in HOLDOUT}
                    write_manifest(man)
                    continue
                man["stacks"][name] = {
                    "dso": dso, "filter": filt, "half": half, "domain": domain, "usable": True,
                    "holdout": dso.lower() in HOLDOUT, "val": dso.lower() in VAL_DSOS,
                    "fwhm": side["fwhm"], "stars": side["stars"], "sky_sigma": side["sky_sigma"],
                    "sat": side["sat"], "accepted": g.get("accepted"), "source": str(src_path.relative_to(_root)),
                }
                write_manifest(man)
                n_done += 1
                log(f"[{name}] done in {time.time() - t0:.0f}s")
            del src
    log(f"prepared {n_done} new stacks; {sum(1 for s in man['stacks'].values() if s['usable'])} usable in all")


def cmd_list(a):
    man = read_manifest()
    rows = sorted(man["stacks"].items())
    for name, s in rows:
        flag = "HOLDOUT" if s.get("holdout") else ("val" if s.get("val") else "")
        if s.get("usable"):
            print(f"{name:34s} FWHM {s['fwhm']:5.2f} stars {s['stars']:4d} sigma {s['sky_sigma']:6.2f} {flag}")
        else:
            print(f"{name:34s} UNUSABLE {flag}")
    print(f"{len(rows)} stacks")


# ------------------------------------------------------------------ train ----
def _split(domain: str):
    items = [it for it in dd.list_prepared() if it.domain in (("raw", "den") if domain == "both" else (domain,))]
    man = read_manifest()
    train, val = [], []
    for it in items:
        meta = man["stacks"].get(it.name, {})
        if not meta.get("usable", True) or meta.get("holdout") or it.dso.lower() in HOLDOUT:
            continue
        (val if (meta.get("val") or it.dso.lower() in VAL_DSOS) else train).append(it)
    return train, val


def _loss(out, tgt, mask, lam, logits=None, pen=0.0):
    """L2 in asinh units + lam * Poisson-weighted linear chi^2, and with
    *logits* and *pen* a logit-norm term pen * mean(logits^2).

    The logit term (2026-10-22): the first core-kernel arm collapsed in its
    first epoch -- the softmax saturated at every pixel (one weight > 0.999,
    logits in the thousands), the gradient vanished and the model froze on a
    hard identity worse than the identity it started from. Reproduced on the
    first 400 steps: a burst of 1e3-1e5 gradient norms round step 50-100
    (clipped to 1, but Adam's per-parameter steps still carry the direction)
    drives the UNet's features, which have no normalisation, until the head
    saturates; the whole-kernel recipe rode the same burst out at |logit|
    < 260. With pen 1e-6 the run excursed to |logit| 18000 and came back
    within ten steps -- the penalty keeps a gradient when the softmax has
    none -- and settled where the whole-kernel run sits (|logit| 50-130,
    mean logit^2 ~ 16 on the promoted model: 1.6e-5 of loss, nothing). A
    learning-rate warm-up alone did not hold.
    """
    import torch
    l2 = (((out - tgt) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    xo, xt = torch.sinh(out), torch.sinh(tgt)
    chi2 = (((xo - xt) ** 2) / (xt.abs() + 1.0) * mask).sum() / mask.sum().clamp(min=1)
    loss = l2 + lam * chi2
    if logits is not None and pen:
        loss = loss + pen * logits.pow(2).mean()
    return loss


def _taper(text: str):
    """--taper: 'none' (the whole measured kernel, the recipe before 2026-10-22)
    or 'inner,outer' in FWHM units (nn/deblur_data.taper_kernel)."""
    t = (text or "").strip().lower()
    if t in ("", "none", "whole", "off"):
        return None
    lo, hi = (float(v) for v in t.split(","))
    return (lo, hi)


def cmd_train(a):
    import torch
    from torch.utils.data import DataLoader
    from nn import deblur_model as dm
    if a.seed is None:
        log("Error: --seed is required (an unseeded run cannot be recovered; lab manual 2026-08-14)")
        return 1
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = _split(a.domain)
    if not train:
        log("no training stacks — run prepare first")
        return 1
    log(f"{a.name}: domain={a.domain} train={len(train)} stacks ({len({t.dso for t in train})} targets) "
        f"val={len(val)} stacks seed={a.seed} epochs={a.epochs} patch={a.patch} pairs={a.pairs} device={dev} "
        f"kernel={'whole' if _taper(a.taper) is None else 'core %s FWHM' % (_taper(a.taper),)}")
    taper = _taper(a.taper)
    tds = dd.DeblurDataset(train, patch_size=a.patch, pairs_per_epoch=a.pairs, seed=a.seed, kernel_taper=taper)
    vds = dd.DeblurDataset(val, patch_size=a.patch, pairs_per_epoch=max(a.batch, a.pairs // 10),
                           seed=a.seed + 1, kernel_taper=taper) if val else None
    tl = DataLoader(tds.torch_dataset(), batch_size=a.batch, num_workers=a.workers, pin_memory=(dev == "cuda"),
                    persistent_workers=a.workers > 0)
    vl = DataLoader(vds.torch_dataset(), batch_size=a.batch, num_workers=min(a.workers, 2)) if vds else None
    model = dm.build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    MODELS.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    meta = {"name": a.name, "domain": a.domain, "seed": a.seed, "epochs": a.epochs, "patch": a.patch,
            "pairs": a.pairs, "lam": a.lam, "lr": a.lr, "logit_pen": a.logit_pen, "kernel_scale": list(dd.KERNEL_SCALE),
            "kernel_taper": None if taper is None else list(taper),
            "train": sorted({t.dso for t in train}), "val": sorted({v.dso for v in val}),
            "holdout": sorted(HOLDOUT), "history": []}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb, mb in tl:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            out, logits = model(xb, return_logits=True)
            loss = _loss(out, yb, mb, a.lam, logits, a.logit_pen)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.shape[0]; n += xb.shape[0]
        sched.step()
        tr = tot / max(n, 1)
        va = float("nan")
        if vl is not None:
            model.eval(); vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb, mb in vl:
                    xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
                    vt += _loss(model(xb), yb, mb, a.lam).item() * xb.shape[0]; vn += xb.shape[0]
            va = vt / max(vn, 1)
        meta["history"].append({"epoch": ep + 1, "train": tr, "val": va, "t": time.time() - t0})
        mark = ""
        score = va if vl is not None else tr
        if score < best:
            best = score; mark = " *"
            meta["best_epoch"] = ep + 1
            dm.save(model, MODELS / f"{a.name}.pt", meta, kernel_taper=taper)
        log(f"  epoch {ep + 1:3d}  train {tr:.5f}  val {va:.5f}  {time.time() - t0:.0f}s{mark}")
    dm.save(model, MODELS / f"{a.name}_last.pt", meta, kernel_taper=taper)
    with open(MODELS / f"{a.name}.json", "w") as f:
        json.dump(meta, f, indent=1)
    log(f"best epoch {meta.get('best_epoch')} -> {MODELS / (a.name + '.pt')}")
    return 0


def cmd_smoke(a):
    a.name, a.domain, a.seed, a.epochs, a.pairs, a.batch = "smoke", "both", 0, 1, 16, 4
    a.patch, a.workers, a.lr, a.lam = 256, 2, 3e-4, 0.2
    a.taper = "%g,%g" % dd.KERNEL_TAPER
    a.logit_pen = 1e-6
    return cmd_train(a)


def _phase0_module(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(_root) / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _real_full_frames(pr, t, i, dev) -> dict:
    """Raw and denoised full half-stacks for the real harness, cached under
    its BASE as <t>_{raw,den}<i>_full.npy (245 MB each) so five models do
    not denoise the same frame five times."""
    out = {}
    for dom in ("raw", "den"):
        p = pr.BASE / f"{t}_{dom}{i}_full.npy"
        if p.exists():
            out[dom] = np.load(p, mmap_mode="r")
    if len(out) == 2:
        return out
    raw = np.array(np.load(pr.STACKS / f"{t}__L_train{i}.npy", mmap_mode="r"), dtype=np.float32)
    raw[~np.isfinite(raw)] = np.nanmedian(raw)
    from nn import denoiser
    den = denoiser.denoise_frame(raw, denoiser.load_model(pr.MODEL), device=dev)[:raw.shape[0], :raw.shape[1]]
    np.save(pr.BASE / f"{t}_raw{i}_full.npy", raw); np.save(pr.BASE / f"{t}_den{i}_full.npy", den.astype(np.float32))
    return {"raw": raw, "den": den.astype(np.float32)}


def cmd_score(a):
    """Score a trained model with both Phase 0 harnesses.

    Synthetic: deconv_phase0's test fields against their truth S (star flux
    is the number that matters; the results JSON joins `deconv_phase0.py
    report`). Real: the ngc5907 / ngc7320 L half-stack crops that
    deconv_phase0_real.py `run` saved, each deblurred with the kernel its
    prepared sidecar measured; `deconv_phase0_real.py report` picks the
    outputs up as a method named after --name.
    """
    import torch
    from nn import deblur_model as dm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = dm.load(a.model, dev)
    name = a.name or Path(a.model).stem
    p0 = _phase0_module("deconv_phase0")
    K = p0.load_kernel(p0.BASE / "kernel.npz")
    images = {}
    for p in sorted((p0.BASE / "fields").glob("test*.npz")):
        sc = np.load(p)
        t0 = time.time()
        out, info = dm.deblur_frame(sc["X"].astype(np.float32), model, K["psf"], device=dev,
                                    sat_level=K["sat"], sky_sigma=K["sky_sigma"])
        images[p.stem] = out
        np.save(p0.BASE / "results" / f"{name}_{p.stem}.npy", out)
        print(f"synthetic {p.stem}: {time.time() - t0:.1f}s, {info['plateau_px']} plateau px")
    p0.evaluate(name, images, K)
    pr = _phase0_module("deconv_phase0_real")
    for t in pr.CENTRES:
        for i in (0, 1):
            frames = _real_full_frames(pr, t, i, dev)       # the whole half-stack, like the baselines
            for dom in ("raw", "den"):
                side = dd.load_sidecar(dd.PREPARED / f"{t}__L_train{i}_{dom}.psf.npz")
                t0 = time.time()
                out, info = dm.deblur_frame(frames[dom], model, side["psf"], device=dev,
                                            sat_level=side["sat"] or None, sky_sigma=side["sky_sigma"])
                np.save(pr.BASE / f"{t}_{dom}{i}_{name}.npy", pr._crop(out, pr.CENTRES[t]))
                print(f"{t} {dom}{i}: {time.time() - t0:.1f}s, kernel FWHM {side['fwhm']:.2f}, "
                      f"{info['plateau_px']} plateau px, {info['tiles']} tiles")
    print("now: deconv_phase0.py report; deconv_phase0_real.py report")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--dsos", default=""); p.add_argument("--domains", default="raw,den")
    p.add_argument("--halves", default="train0,train1"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    t = sub.add_parser("train"); t.add_argument("--name", required=True)
    t.add_argument("--domain", default="both", choices=("raw", "den", "both"))
    t.add_argument("--seed", type=int); t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patch", type=int, default=dd.PATCH); t.add_argument("--pairs", type=int, default=2000)
    t.add_argument("--batch", type=int, default=4); t.add_argument("--workers", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    # chi^2 weight. Measured at identity on 24 real 512 px patches (abell2151 +
    # m13, 2026-10-19): L2 in asinh units 1.29, chi^2 in linear units 5.7 on
    # average with a tail to 51 (bright stars), so 0.2 makes the two terms
    # comparable at init. Phase 0's 1.0 was right for synthetic fields with
    # few bright stars and would let the stars' cores own the loss here.
    t.add_argument("--lam", type=float, default=0.2)
    # the kernel the model learns to remove: the measured kernel's core. 'none'
    # is the whole 145 px stamp (every arm to 2026-10-21), which taught the
    # model to halve bright stars' halos out to the stamp's edge.
    t.add_argument("--taper", default="%g,%g" % dd.KERNEL_TAPER,
                   help="inner,outer in FWHM units, or none (default %(default)s)")
    # logit-norm penalty against softmax collapse (see _loss); 0 = off, as every
    # arm to 2026-10-21 ran. Negligible on a healthy run (1.6e-5 of the loss).
    t.add_argument("--logit-pen", type=float, default=1e-6)
    t.set_defaults(fn=cmd_train)
    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    sc = sub.add_parser("score"); sc.add_argument("--model", required=True); sc.add_argument("--name", default="")
    sc.set_defaults(fn=cmd_score)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
