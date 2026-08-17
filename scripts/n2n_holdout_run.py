#!/usr/bin/env python3
"""Train N2N on one DSO's split stacks, then denoise a *different* DSO.

    python scripts/n2n_holdout_run.py --train bubble --test sh2-92 \
        --filters Ha,O-III --exptime 300 --tag bubble2sh2

This is the cross-target generalisation run: nothing from --test is used for
training or for choosing the checkpoint, so the output is what the model does on
a field it has never seen. It replaces a scratchpad script that produced the
2026-08-15 `nbcal` checkpoints and was then lost with the machine's /tmp; the
numbers it printed are not reproducible because the script is gone. Hence this
being a committed file with its arguments in the log.

Three things here differ deliberately from that lost script:

1. **Validation is held inside --train**, on frames excluded from the training
   stacks. The old script validated on half-stacks of the *test* target, so the
   held-out DSO chose the checkpoint and "held out" was not true of the run.

2. **The test stack is depth-matched to the training stacks.** Stacks shallower
   than the frame inference runs on teach shrinkage calibrated for noisier data
   and over-suppress at inference — measured, docs/N2N_LAB_MANUAL.md step 16,
   where quartering dropped injected-source response 1.07 -> 0.56. Denoising a
   137-frame stack with a model trained on 22-frame stacks is that same mismatch
   in the same direction, so --test is stacked to the training depth and the
   spare frames go unused.

3. **The reported metric is corr against the measurable ceiling**, not the
   background RMS the old script printed. `sep.Background().globalrms` only says
   how smooth the sky came out, so a net that smears everything drives it to
   zero and looks excellent: the lost run reported 1.2793 -> 0.0846 on that
   metric, which is unfalsifiable as a quality claim. See `collapse_check`.
"""

import argparse
import gc
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


def log(msg: str = "") -> None:
    print(msg, flush=True)


def split_depths(n: int) -> tuple[int, int]:
    """Return (train_per, val_per) for n frames of one (dso, filter).

    Two training stacks and two validation stacks, all disjoint. Validation
    stacks are the shallow ones: they only have to rank epochs against each
    other, while the training stacks set the noise level the model learns for
    and so get the depth.

    The floor of 2 exists for --limit smoke tests only; at the real frame counts
    (59 Ha, 77 O-III on the bubble) n // 8 dominates it.
    """
    val_per = max(2, n // 8)
    train_per = (n - 2 * val_per) // 2
    return train_per, val_per


def _peak_offset(a: np.ndarray, b: np.ndarray, bin_: int = 4) -> tuple[int, int]:
    """Integer (dy, dx) of b relative to a, by phase correlation on a binned copy.

    Needed because the two test stacks come from two separate `stack_paths`
    calls, and each registers its frames onto its *own* reference frame — so the
    two finished stacks sit on different pixel grids. Measured on the
    2026-08-17 bubble->sh2-92 run: Ha off by (-8, +8), O-III by (+36, 0).
    """
    from numpy.fft import fft2, ifft2

    def prep(x: np.ndarray) -> np.ndarray:
        h, w = (s // bin_ * bin_ for s in x.shape)
        y = x[:h, :w].reshape(h // bin_, bin_, w // bin_, bin_).mean(axis=(1, 3))
        return np.clip(y - np.median(y), 0, None)

    A, B = prep(a), prep(b)
    cc = np.abs(ifft2(fft2(A) * np.conj(fft2(B))))
    iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(cc)), cc.shape))
    if iy > cc.shape[0] // 2:
        iy -= cc.shape[0]
    if ix > cc.shape[1] // 2:
        ix -= cc.shape[1]
    return iy * bin_, ix * bin_


def collapse_check(raw: np.ndarray, other: np.ndarray,
                   denoised: np.ndarray) -> dict:
    """corr(input, output) against the ceiling set by two independent stacks.

    A collapsed net — one that has learned to emit a constant — still writes
    correctly-shaped output and still drives the loss down, so shape and loss
    prove nothing. corr does, but *not* against 1.0: these frames are noise
    dominated, so a perfect denoiser returns the clean scene while its input is
    clean+noise, and corr(input, perfect output) is bounded well below 1. The
    bound is measurable from two independent stacks of the same scene at the
    same depth:

        ceiling = sqrt(corr(stack_a, stack_b))

    which was 0.37 on abell2151 R 300s, not 0.9. Judge the run as a fraction of
    that. A naive ">0.9 is healthy" threshold failed a working model once
    already.

    **The two stacks must be aligned before that correlation is taken.** They are
    not aligned as they come out of the stacker (see `_peak_offset`), and an
    unaligned corr reads misalignment as noise, which pushes the ceiling *down*
    and the run's score up. Unaligned, the 2026-08-17 run scored O-III at 285%
    of its ceiling — impossible, and the only reason the bug was caught. Aligned,
    the same run read Ha 30% and O-III 77%.

    `std_ratio` compares the denoised frame to the raw on ONE scale. Normalising
    each independently divides each by its own sky sigma and forces the ratio to
    ~1 no matter what the network did — that number was meaningless in the first
    version of this function. Its ideal value is the ceiling, not 1: a perfect
    denoiser returns the clean scene, whose std is exactly `ceiling` times the
    noisy input's.
    """
    from nn import denoiser

    def sub(a: np.ndarray) -> np.ndarray:
        out, _ = denoiser.subtract_background(a.astype(np.float32))
        return out

    a_s, b_s, d_s = sub(raw), sub(other), sub(denoised)

    dy, dx = _peak_offset(a_s, b_s)
    h, w = a_s.shape
    ya, yb = (0, -dy) if dy < 0 else (dy, 0)
    xa, xb = (0, -dx) if dx < 0 else (dx, 0)
    hh, ww = h - abs(dy), w - abs(dx)
    a_c = np.ascontiguousarray(a_s[ya:ya + hh, xa:xa + ww])
    b_c = np.ascontiguousarray(b_s[yb:yb + hh, xb:xb + ww])

    def norm(a: np.ndarray) -> np.ndarray:
        out, _ = denoiser.normalise(a)
        return out.ravel()

    r_ab = float(np.corrcoef(norm(a_c), norm(b_c))[0, 1])
    ceiling = float(np.sqrt(max(r_ab, 0.0)))
    corr = float(np.corrcoef(norm(a_s), norm(d_s))[0, 1])

    a_n, scale = denoiser.normalise(a_s)
    d_shared = np.arcsinh(d_s / (scale if scale else 1.0))
    return {
        "stack_offset": [dy, dx],
        "corr_stacks": r_ab,
        "ceiling": ceiling,
        "corr_in_out": corr,
        "fraction_of_ceiling": corr / ceiling if ceiling > 0 else float("nan"),
        "std_ratio": float(np.std(d_shared) / max(np.std(a_n), 1e-12)),
        "std_ratio_ideal": ceiling,
    }


def source_survival(raw: np.ndarray, denoised: np.ndarray,
                    nsigma: float = 5.0, aperture: float = 3.0) -> dict:
    """Do sources survive, and with how much of their flux?

    The decisive test, and the one corr cannot answer: a net can score well on
    correlation while erasing most of the stars.

    Both images are thresholded at `nsigma` times the **raw** sky in ADU, never
    each at its own. The denoised sky is 8-30x quieter, so a per-image threshold
    sits far lower in ADU on the denoised frame and finds *more*, fainter sources
    than the raw one — 203% and 382% on the 2026-08-17 run, with fluxes inflated
    to match. That measures the threshold, not the network. Flux then comes from
    fixed apertures at the raw frame's own positions, so both frames are summed
    over identical pixels.
    """
    import sep

    sep.set_extract_pixstack(5_000_000)
    a = np.ascontiguousarray(raw.astype(np.float32))
    d = np.ascontiguousarray(denoised.astype(np.float32))
    bg_a, bg_d = sep.Background(a), sep.Background(d)
    a_s, d_s = a - bg_a.back(), d - bg_d.back()
    rms = float(bg_a.globalrms)
    thresh = nsigma * rms

    src_a = sep.extract(a_s, thresh, minarea=5)
    src_d = sep.extract(d_s, thresh, minarea=5)

    out = {
        "sky_rms_raw": rms,
        "sky_rms_denoised": float(bg_d.globalrms),
        "threshold_adu": thresh,
        "n_raw": int(len(src_a)),
        "n_denoised": int(len(src_d)),
        "source_survival": len(src_d) / max(len(src_a), 1),
    }
    if len(src_a):
        f_a, _, _ = sep.sum_circle(a_s, src_a["x"], src_a["y"], aperture)
        f_d, _, _ = sep.sum_circle(d_s, src_a["x"], src_a["y"], aperture)
        keep = f_a > 0
        if keep.any():
            ratio = f_d[keep] / f_a[keep]
            order = np.argsort(f_a[keep])
            out["flux_retained_median"] = float(np.median(ratio))
            out["flux_retained_quintiles"] = [
                float(np.median(ratio[q])) for q in np.array_split(order, 5)
            ]
    return out


def save_pair_pngs(raw: np.ndarray, denoised: np.ndarray, out_dir: Path,
                   stem: str, max_px: int = 1600) -> list[Path]:
    """Write before/after PNGs under ONE stretch computed from the raw frame.

    The shared stretch is the whole point. Stretching each frame on its own
    percentiles would rescale them independently, and the pair would differ in
    brightness and contrast for reasons that have nothing to do with the
    network — which is the same error that makes a per-channel stretch destroy
    colour ratios in color_process. Black and white come from `raw` and are then
    applied unchanged to both, so any visible difference is the model's.
    """
    from PIL import Image

    from stacking.color_process import _remove_gradient, _stretch

    flat_raw = _remove_gradient(raw.astype(np.float32))
    flat_den = _remove_gradient(denoised.astype(np.float32))

    # Deliberately not crushing the background: the sky grain is the thing being
    # judged, so a black point that clips it would hide exactly the difference
    # the pair exists to show.
    black_pct = 25.0
    white = float(np.nanpercentile(flat_raw, 99.9))
    log(f"    stretch from raw: p{black_pct} black, white {white:.2f} ADU "
        f"(applied to both)")

    paths = []
    for label, arr in (("before", flat_raw), ("after", flat_den)):
        y = _stretch(arr, black_pct, white)
        img = Image.fromarray((np.clip(y, 0, 1) * 255).astype(np.uint8), mode="L")
        if max(img.size) > max_px:
            scale = max_px / max(img.size)
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))),
                             Image.LANCZOS)
        p = out_dir / f"{stem}_{label}.png"
        img.save(p, optimize=True)
        paths.append(p)
        log(f"    wrote {p.name} ({img.width}x{img.height})")
    return paths


def run_filter(filt: str, args, subs_dir: Path, out_dir: Path) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn import denoiser, stacks
    from nn.noise2noise_model import UNet
    from nn.trainer import N2NDataset

    log(f"\n{'=' * 64}\n{filt}\n{'=' * 64}")
    idx = stacks.index_frames(subs_dir, [filt], args.exptime)
    tr_key, te_key = (args.train, filt), (args.test, filt)
    for key in (tr_key, te_key):
        if key not in idx or len(idx[key]) < 4:
            log(f"  no usable frames for {key} — skipping filter")
            return {}

    tr_paths, te_paths = idx[tr_key], idx[te_key]
    if args.limit:
        tr_paths, te_paths = tr_paths[:args.limit], te_paths[:args.limit]
        log(f"  --limit {args.limit}: smoke test, NOT a result")
    train_per, val_per = split_depths(len(tr_paths))
    if train_per < 4:
        log(f"  {args.train} {filt}: only {len(tr_paths)} frames — skipping")
        return {}
    log(f"  train {args.train}: {len(tr_paths)} subs -> 2x{train_per} + "
        f"2x{val_per} val")
    log(f"  test  {args.test}: {len(te_paths)} subs -> 2x{train_per} "
        f"(depth-matched; {len(te_paths) - 2 * train_per} unused)")

    rng = np.random.default_rng(args.seed)
    prog = (lambda m: log(f"      {m}")) if args.verbose else (lambda m: None)

    def stack_chunks(paths, order, per, count, offset=0):
        made = []
        for k in range(count):
            lo = offset + k * per
            chunk = [paths[i] for i in order[lo:lo + per]]
            img = stacks.stack_paths(chunk, filt, progress_cb=prog)
            if img is None:
                return None
            made.append(img)
        return made

    t0 = time.time()
    o = rng.permutation(len(tr_paths))
    tr_st = stack_chunks(tr_paths, o, train_per, 2)
    va_st = stack_chunks(tr_paths, o, val_per, 2, offset=2 * train_per)
    if tr_st is None or va_st is None:
        log("  stacking failed — skipping filter")
        return {}
    tr_st = stacks.crop_to_common(tr_st)
    va_st = stacks.crop_to_common(va_st)
    log(f"  {args.train} stacked [{time.time() - t0:.0f}s]")

    t0 = time.time()
    o2 = rng.permutation(len(te_paths))
    te_st = stack_chunks(te_paths, o2, train_per, 2)
    if te_st is None:
        log("  test stacking failed — skipping filter")
        return {}
    te_st = stacks.crop_to_common(te_st)
    log(f"  {args.test} stacked [{time.time() - t0:.0f}s]")

    np.save(out_dir / f"{filt}_{args.test}_raw.npy", te_st[0])
    np.save(out_dir / f"{filt}_{args.test}_raw_b.npy", te_st[1])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_nn = config.data().get("nn", {})
    patch = int(cfg_nn.get("patch_size", 256))
    batch = int(cfg_nn.get("batch_size", 8))
    pairs = args.pairs or int(cfg_nn.get("pairs_per_epoch", 2000))

    ds = N2NDataset(tr_st, group_ids=[f"{args.train}|{filt}"] * 2,
                    patch_size=patch, pairs_per_epoch=pairs, seed=args.seed)
    vs = N2NDataset(va_st, group_ids=[f"{args.train}|{filt}|val"] * 2,
                    patch_size=patch, pairs_per_epoch=max(batch, pairs // 5),
                    seed=args.seed + 1)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(residual="linear").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.L1Loss()

    best, best_sd, best_ep = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for a, b in dl:
            a, b = a.to(device), b.to(device)
            opt.zero_grad()
            loss = crit(model(a), b)
            loss.backward()
            # The grad-norm tail is heavy enough to poison the checkpoint
            # silently rather than error; see nn/trainer.py.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sch.step()
        model.eval()
        v = 0.0
        with torch.no_grad():
            for a, b in vl:
                v += crit(model(a.to(device)), b.to(device)).item()
        v /= max(len(vl), 1)
        if v < best:
            best, best_ep = v, ep
            best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs:
            log(f"  epoch {ep:3d} train={tot / max(len(dl), 1):.5f} "
                f"val={v:.5f} best={best:.5f} ({time.time() - t0:.0f}s)")

    model.load_state_dict(best_sd)
    safe = filt.replace(" ", "_").replace("/", "_")
    model_path = Path(_root) / "local" / "models" / f"n2n_{args.tag}_{safe}_{args.exptime}s.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save({"model_state": best_sd, "filter": f"{args.tag}_{filt}",
             "epoch": best_ep, "asinh_sigma_mult": denoiser.ASINH_SIGMA_MULT,
             "train_dso": args.train, "test_dso": args.test,
             "train_depth": train_per, "seed": args.seed}, model_path)
    log(f"  best val {best:.5f} at epoch {best_ep} -> {model_path.name}")

    den = denoiser.denoise_frame(te_st[0], model.cpu(), device=device)
    np.save(out_dir / f"{filt}_{args.test}_denoised.npy", den)

    chk = collapse_check(te_st[0], te_st[1], den)
    log(f"  stacks offset by {chk['stack_offset']} px (aligned before the ceiling)")
    log(f"  corr(in,out) {chk['corr_in_out']:.4f} vs ceiling "
        f"{chk['ceiling']:.4f} = {100 * chk['fraction_of_ceiling']:.0f}% "
        f"of ceiling | std ratio {chk['std_ratio']:.4f} "
        f"(ideal {chk['std_ratio_ideal']:.4f})")
    if chk["corr_in_out"] < 0.3 * chk["ceiling"]:
        log("  *** FAILS the collapse check — treat this checkpoint as dead ***")

    srv = source_survival(te_st[0], den)
    log(f"  sources at {srv['threshold_adu']:.3f} ADU (5 sigma of the RAW sky): "
        f"{srv['n_denoised']} / {srv['n_raw']} survive "
        f"({100 * srv['source_survival']:.0f}%)")
    if "flux_retained_median" in srv:
        q = " ".join(f"{v:.3f}" for v in srv["flux_retained_quintiles"])
        log(f"  flux retained: median {srv['flux_retained_median']:.4f} | "
            f"faint->bright {q}")
    chk.update(srv)

    pngs = save_pair_pngs(te_st[0], den, out_dir, f"{args.test}_{safe}")

    result = {"filter": filt, "best_val": best, "best_epoch": best_ep,
              "train_depth": train_per, "model": str(model_path),
              "pngs": [str(p) for p in pngs], **chk}
    del tr_st, va_st, te_st, den
    gc.collect()
    if device == "cuda":
        _t.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="DSO to train on")
    ap.add_argument("--test", required=True, help="DSO to denoise, never trained on")
    ap.add_argument("--filters", default="Ha,O-III")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="model/output name, default {train}2{test}")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--verbose", action="store_true", help="stacker progress lines")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap subs per DSO — smoke tests only, not a result")
    ap.add_argument("--pairs", type=int, default=0,
                    help="override nn.pairs_per_epoch (smoke tests)")
    args = ap.parse_args()

    args.train = args.train.lower().replace(" ", "")
    args.test = args.test.lower().replace(" ", "")
    if args.train == args.test:
        log("Error: --train and --test must differ; that is the whole point")
        return 1
    if args.tag is None:
        args.tag = f"{args.train}2{args.test}".replace("-", "")

    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    if "subs_dir" not in machine:
        log(f"Error: machine.{socket.gethostname()} has no 'subs_dir' — "
            "this chain only runs on the Spark")
        return 1
    subs_dir = Path(machine["subs_dir"])

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(_root) / "local" / "n2n_holdout" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    import sep
    sep.set_extract_pixstack(3_000_000)

    log(f"train={args.train} test={args.test} filters={args.filters} "
        f"exptime={args.exptime}s epochs={args.epochs} seed={args.seed}")
    log(f"subs={subs_dir}  out={out_dir}")

    results = []
    for filt in [f.strip() for f in args.filters.split(",") if f.strip()]:
        try:
            r = run_filter(filt, args, subs_dir, out_dir)
            if r:
                results.append(r)
        except Exception:
            import traceback
            log(f"  {filt} FAILED:")
            traceback.print_exc()

    log(f"\n{'=' * 64}\nSUMMARY  train={args.train} -> test={args.test}")
    for r in results:
        log(f"  {r['filter']:6s} val={r['best_val']:.5f}@{r['best_epoch']:3d} "
            f"depth={r['train_depth']:3d} corr={r['corr_in_out']:.4f}/"
            f"{r['ceiling']:.4f} ({100 * r['fraction_of_ceiling']:.0f}%)")
    import json
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log("ALL DONE")
    return 0


# DataLoader workers re-import this module. Without the guard the whole run
# starts again inside every worker — the failure mode the transit runners hit.
if __name__ == "__main__":
    sys.exit(main())
