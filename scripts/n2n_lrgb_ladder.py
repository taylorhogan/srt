#!/usr/bin/env python3
"""The LRGB thinness ladder: is the denoiser limited by how little data it sees?

    python scripts/n2n_lrgb_ladder.py stacks
    python scripts/n2n_lrgb_ladder.py train --arm per-filter
    python scripts/n2n_lrgb_ladder.py train --arm pooled-filters
    python scripts/n2n_lrgb_ladder.py train --arm pooled-scenes
    python scripts/n2n_lrgb_ladder.py report

Lab manual step 19 showed one model pooled over bubble Ha + O-III beating the
per-filter models on identical test data, reversing step 17. That was 2 groups.
The open question it left is what actually limits this pipeline, with data
thinness back as the leading hypothesis, and the experiment named there is this
one: 26 pairable (dso, filter) groups exist at 300 s against the 2 used so far.

Three arms, one held-out target (default ngc5907), all four broadband filters.
The arms differ in **exactly one thing** — which groups are in the training pool:

    per-filter      {base|F} alone, one model per filter   (4 models, 1 group each)
    pooled-filters  {base|L, base|R, base|G, base|B}       (1 model,  4 groups)
    pooled-scenes   every broadband group except the test  (1 model, ~14 groups)

per-filter -> pooled-filters adds passbands at fixed scenes. pooled-filters ->
pooled-scenes adds scenes. A ladder rather than a single A/B, because step 19
moved both at once and could not say which one paid.

Why the stacks are built once, up front, into `local/n2n_ladder/stacks/`:

- **Every arm must train on the same pixels.** Rebuilding per arm would redraw
  the permutation and restack, so an arm could win on a luckier draw. Here arm 3
  trains on arm 2's four groups plus ten more, byte for byte.
- **Every arm must be scored on the same pixels.** The test target's stacks are
  built by the same code path as everything else and then simply never trained
  on, so the three arms' numbers differ only by the model.
- It is idempotent. Stacking is ~50 min of the run; a crash in training does not
  pay it again.

Validation is frame-disjoint and scene-shared, identically in all three arms:
every group yields 2 training stacks and 2 shallower validation stacks from
frames the training stacks never saw. The pooled path's own group-level holdout
is deliberately not used here — it would hold out `abell2151|L` while training on
`abell2151|B` (the same field in another passband) in arms 2 and 3, and cannot
hold out anything at all in arm 1, so the arms would not be comparable. Val only
picks the epoch; the reported numbers come from a target no arm has ever seen.

`pairs_per_epoch` is held at 2000 across all arms, as in step 19, so every arm
gets the same gradient budget and the question is where that budget is best
spent. It does mean arm 3's 14 groups are each sampled ~14x less than arm 1's
one. If arm 3 loses, that confound is the first thing to re-run at higher
`--pairs`, not a verdict on scenes.

## Two caveats this run cannot remove, both measured rather than assumed

**Stack depth is not what `train_per` says.** The quality gate cuts on FWHM and
star count after the split, and it cuts hard and unevenly: 12 and 14 of 17 on
abell2151 G, but 6 and 7 of **12** on abell2151 B. `accepted` in the manifest
records what survived, and `effective_depth` is what the arms are reported
against. Nominal depth overstates by 25-46% depending on the group.

**So arm 3 is also a depth-diversity arm, not purely a scene-count arm.** Its
pool spans roughly 6 to 50 frames per stack while arm 1 sits at a single depth.
Step 16 established that stacks shallower than the inference frame teach
over-aggressive shrinkage, so if arm 3 wins, some of the win may be depth
variety rather than scene count. The ladder still isolates the *comparison* —
all three arms are scored on byte-identical test stacks — but the mechanism
behind a win is not fully pinned by this design. The follow-up that would pin it
is arm 3 restricted to groups within one depth band.
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

# The metrics live in n2n_holdout_run and are reused rather than reimplemented:
# collapse_check and source_survival each encode a trap that produced a
# confident wrong answer once already (an unaligned ceiling, and a per-image
# detection threshold). A second copy is a second chance to get them wrong.
_spec = importlib.util.spec_from_file_location(
    "hr", os.path.join(_root, "scripts", "n2n_holdout_run.py"))
hr = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["x"]
_spec.loader.exec_module(hr)
sys.argv = _argv

LADDER = Path(_root) / "local" / "n2n_ladder"
STACK_DIR = LADDER / "stacks"
MANIFEST = LADDER / "manifest.json"
ARMS = ("per-filter", "pooled-filters", "pooled-scenes")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def subs_dir() -> Path:
    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    if "subs_dir" not in machine:
        raise SystemExit(f"machine.{socket.gethostname()} has no 'subs_dir' — "
                         "N2N runs on the Spark only")
    return Path(machine["subs_dir"])


def gkey(dso: str, filt: str) -> str:
    return f"{dso}|{filt}"


def stack_files(dso: str, filt: str) -> list[Path]:
    safe = filt.replace(" ", "_").replace("/", "_")
    return [STACK_DIR / f"{dso}__{safe}_{kind}{k}.npy"
            for kind in ("train", "val") for k in (0, 1)]


def read_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"groups": {}}


def write_manifest(man: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True))


# --------------------------------------------------------------------------
# stage: stacks
# --------------------------------------------------------------------------

def build_stacks(args) -> int:
    from nn import stacks

    STACK_DIR.mkdir(parents=True, exist_ok=True)
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    idx = stacks.index_frames(subs_dir(), filters, args.exptime)
    man = read_manifest()
    man["exptime"] = args.exptime
    man["seed"] = args.seed
    man["test"] = args.test

    prog = (lambda m: log(f"      {m}")) if args.verbose else (lambda m: None)

    for (dso, filt) in sorted(idx):
        key = gkey(dso, filt)
        paths = idx[(dso, filt)]
        train_per, val_per = hr.split_depths(len(paths))
        if train_per < args.min_depth:
            log(f"[{key}] {len(paths)} frames -> train_per {train_per} "
                f"< {args.min_depth} — skipped")
            man["groups"].pop(key, None)
            continue

        files = stack_files(dso, filt)
        if all(f.exists() for f in files) and key in man["groups"]:
            g = man["groups"][key]
            log(f"[{key}] cached: 2x{g['train_per']} + 2x{g['val_per']}, "
                f"pair offset ({g['offset_dy']},{g['offset_dx']}) px")
            continue

        log(f"[{key}] {len(paths)} frames -> 2x{train_per} train + "
            f"2x{val_per} val")
        t0 = time.time()
        # Fresh rng per group, keyed only on the seed, so a group's frames are
        # the same no matter which groups were built before it or in what order.
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(paths))
        ref = stacks.shared_reference_for(paths, filt, progress_cb=lambda m: log(f"  {m}"))
        if ref is None:
            log(f"[{key}] no shared reference — skipped rather than risk "
                f"per-split references")
            man["groups"].pop(key, None)
            continue

        def chunk(lo: int, n: int) -> list:
            return [paths[i] for i in order[lo:lo + n]]

        # Nominal depth is train_per; the quality gate then drops frames on FWHM
        # and star count, so what the model actually trains on is `accepted`.
        metas = [{} for _ in range(4)]
        made = [stacks.stack_paths(chunk(k * train_per, train_per), filt,
                                   progress_cb=prog, shared_reference=ref,
                                   meta_out=metas[k])
                for k in range(2)]
        vmade = [stacks.stack_paths(chunk(2 * train_per + k * val_per, val_per),
                                    filt, progress_cb=prog, shared_reference=ref,
                                    meta_out=metas[2 + k])
                 for k in range(2)]
        if any(x is None for x in made + vmade):
            log(f"[{key}] stacking failed — skipped")
            man["groups"].pop(key, None)
            continue

        made, vmade = stacks.crop_to_common(made), stacks.crop_to_common(vmade)
        dy, dx = hr._peak_offset(*(np.ascontiguousarray(s) for s in made))
        vdy, vdx = hr._peak_offset(*(np.ascontiguousarray(s) for s in vmade))
        accepted = [int(m.get("n_frames", 0)) for m in metas]
        log(f"[{key}] built in {time.time() - t0:.0f}s, shape {made[0].shape}, "
            f"pair offset ({dy},{dx}) px, val pair ({vdy},{vdx}) px")
        log(f"[{key}] depth after the quality gate: train {accepted[0]},{accepted[1]} "
            f"of {train_per}  val {accepted[2]},{accepted[3]} of {val_per}")

        # The whole 2026-08-17 collapse was a pair sitting 8-36 px apart while
        # every log line said success. A group that fails this is recorded and
        # excluded, never silently trained on.
        ok = max(abs(dy), abs(dx)) <= args.max_offset
        if not ok:
            log(f"[{key}] *** pair misaligned by more than {args.max_offset} px "
                f"— recorded as unusable ***")

        for f, arr in zip(files, made + vmade):
            np.save(f, arr.astype(np.float32))
        man["groups"][key] = {
            "dso": dso, "filter": filt, "n_frames": len(paths),
            "train_per": train_per, "val_per": val_per,
            "accepted": accepted,
            "offset_dy": int(dy), "offset_dx": int(dx),
            "val_offset_dy": int(vdy), "val_offset_dx": int(vdx),
            "shape": list(made[0].shape), "usable": bool(ok),
            "files": [str(f.relative_to(_root)) for f in files],
        }
        write_manifest(man)
        del made, vmade
        gc.collect()

    write_manifest(man)
    usable = [k for k, g in man["groups"].items() if g["usable"]]
    log(f"\n{len(usable)} usable groups: {', '.join(sorted(usable))}")
    bad = [k for k, g in man["groups"].items() if not g["usable"]]
    if bad:
        log(f"{len(bad)} unusable (misaligned): {', '.join(sorted(bad))}")
    return 0


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def arm_groups(man: dict, arm: str, args, filt: str = None) -> list[str]:
    """Group keys this arm trains on. The only thing that differs between arms."""
    usable = {k: g for k, g in man["groups"].items() if g["usable"]}
    out = []
    for key, g in sorted(usable.items()):
        if g["dso"] == args.test:
            continue                      # never trained on, in any arm
        if arm == "per-filter":
            if g["dso"] == args.base and g["filter"] == filt:
                out.append(key)
        elif arm == "pooled-filters":
            if g["dso"] == args.base:
                out.append(key)
        elif arm == "pooled-scenes":
            out.append(key)
    return out


def effective_depth(g: dict) -> int:
    """Frames actually in a training stack, after the quality gate.

    Falls back to the nominal `train_per` for groups stacked before `accepted`
    was recorded (the three G groups of 2026-08-19). Those are the only entries
    where the two can disagree, and the gate has been seen to drop ~25%, so a
    nominal figure is an overestimate rather than an equivalent.
    """
    acc = g.get("accepted")
    if acc and acc[0] and acc[1]:
        return int(round((acc[0] + acc[1]) / 2))
    return int(g["train_per"])


def load_group(man: dict, key: str) -> tuple[list, list]:
    files = [Path(_root) / p for p in man["groups"][key]["files"]]
    train = [np.load(f) for f in files[:2]]
    val = [np.load(f) for f in files[2:]]
    return train, val


def fit(train_frames, train_gids, val_frames, val_gids, args, tag: str):
    """Train a UNet. Hyperparameters are run_filter's, unchanged."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn.noise2noise_model import UNet
    from nn.trainer import N2NDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_nn = config.data().get("nn", {})
    patch = int(cfg_nn.get("patch_size", 256))
    batch = int(cfg_nn.get("batch_size", 8))
    pairs = args.pairs or int(cfg_nn.get("pairs_per_epoch", 2000))

    ds = N2NDataset(train_frames, group_ids=train_gids, patch_size=patch,
                    pairs_per_epoch=pairs, seed=args.seed)
    vs = N2NDataset(val_frames, group_ids=val_gids, patch_size=patch,
                    pairs_per_epoch=max(batch, pairs // 5), seed=args.seed + 1)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(vs, batch_size=batch, shuffle=False, num_workers=2)
    log(f"  {len(ds._valid_pairs)} training pairs over "
        f"{len(set(train_gids))} groups; {pairs} patch draws/epoch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(residual="linear").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss() if args.loss == "l2" else nn.L1Loss()

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
            log(f"  [{tag}] epoch {ep:3d} train={tot / max(len(dl), 1):.5f} "
                f"val={v:.5f} best={best:.5f} ({time.time() - t0:.0f}s)")

    model.load_state_dict(best_sd)
    return model, best, best_ep


def evaluate(model, man: dict, args, filters: list[str], out_dir: Path,
             arm: str) -> list[dict]:
    """Score one model on the held-out target, filter by filter."""
    import torch

    from nn import denoiser

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for filt in filters:
        key = gkey(args.test, filt)
        if key not in man["groups"] or not man["groups"][key]["usable"]:
            log(f"  {filt}: no usable test stacks for {args.test} — skipped")
            continue
        test, _ = load_group(man, key)
        raw, other = test[0], test[1]
        depth = effective_depth(man["groups"][key])
        log(f"\n  --- {arm} on {args.test} {filt} "
            f"(2 test stacks of {depth}) ---")

        den = denoiser.denoise_frame(raw, model, device=device)
        safe = filt.replace(" ", "_").replace("/", "_")
        np.save(out_dir / f"{args.test}_{safe}_denoised.npy", den)

        chk = hr.collapse_check(raw, other, den)
        log(f"  stacks offset by {chk['stack_offset']} px (aligned before the ceiling)")
        log(f"  corr(in,out) {chk['corr_in_out']:.4f} vs ceiling "
            f"{chk['ceiling']:.4f} = {100 * chk['fraction_of_ceiling']:.0f}% "
            f"of ceiling | std ratio {chk['std_ratio']:.4f} "
            f"(ideal {chk['std_ratio_ideal']:.4f})")
        if chk["corr_in_out"] < 0.3 * chk["ceiling"]:
            log("  *** FAILS the collapse check — treat this checkpoint as dead ***")

        srv = hr.source_survival(raw, den)
        log(f"  sources at {srv['threshold_adu']:.3f} ADU (5 sigma of the RAW sky): "
            f"{srv['n_denoised']} / {srv['n_raw']} survive "
            f"({100 * srv['source_survival']:.0f}%)")
        if "flux_retained_median" in srv:
            q = " ".join(f"{v:.3f}" for v in srv["flux_retained_quintiles"])
            log(f"  flux retained: median {srv['flux_retained_median']:.4f} | "
                f"faint->bright {q}")

        pngs = hr.save_pair_pngs(raw, den, out_dir, f"{arm}_{args.test}_{safe}")
        results.append({"arm": arm, "filter": filt, "test_depth": depth,
                        "pngs": [str(p) for p in pngs], **chk, **srv})
        del test, raw, other, den
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def train_arm(args) -> int:
    import torch

    man = read_manifest()
    if not man.get("groups"):
        log("No manifest — run the `stacks` stage first.")
        return 1
    if man.get("test") != args.test:
        log(f"Manifest was built for test={man.get('test')}, not {args.test}.")
        return 1

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    out_dir = LADDER / (args.arm + args.suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Arm 1 is four independent single-group models; arms 2 and 3 are one model
    # scored on every filter. That asymmetry is the arm, not an implementation
    # detail: "one model per filter" is the thing being tested.
    plans = ([(f, [f]) for f in filters] if args.arm == "per-filter"
             else [(None, filters)])

    for filt, eval_filters in plans:
        keys = arm_groups(man, args.arm, args, filt)
        if not keys:
            log(f"  no groups for arm {args.arm} filter {filt} — skipped")
            continue
        tag = f"{args.arm}{args.suffix}" + (f"_{filt}" if filt else "")
        depths = [effective_depth(man["groups"][k]) for k in keys]
        log(f"\n{'=' * 64}\n{tag}: {len(keys)} groups — {', '.join(keys)}\n"
            f"stack depths {min(depths)}-{max(depths)} "
            f"(median {int(np.median(depths))})\n{'=' * 64}")

        tr_frames, tr_gids, va_frames, va_gids = [], [], [], []
        for k in keys:
            tr, va = load_group(man, k)
            tr_frames += tr
            tr_gids += [k] * len(tr)
            va_frames += va
            va_gids += [f"{k}|val"] * len(va)

        model, best, best_ep = fit(tr_frames, tr_gids, va_frames, va_gids,
                                   args, tag)
        del tr_frames, va_frames
        gc.collect()

        safe_tag = tag.replace(" ", "_").replace("/", "_")
        model_path = (Path(_root) / "local" / "models" /
                      f"n2n_ladder_{safe_tag}_{args.exptime}s.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        from nn import denoiser
        torch.save({"model_state": model.state_dict(), "filter": tag,
                    "epoch": best_ep, "asinh_sigma_mult": denoiser.ASINH_SIGMA_MULT,
                    "arm": args.arm, "groups": keys, "test_dso": args.test,
                    "seed": args.seed, "loss": args.loss,
                    # The effective value, not args.pairs — that is 0 when the
                    # config supplies it, and a checkpoint recording "0 pairs
                    # per epoch" is worse than recording nothing.
                    "pairs_per_epoch": (args.pairs or int(
                        config.data().get("nn", {}).get("pairs_per_epoch", 2000))),
                    "epochs": args.epochs}, model_path)
        log(f"  best val {best:.5f} at epoch {best_ep} -> {model_path.name}")

        res = evaluate(model, man, args, eval_filters, out_dir, args.arm)
        for r in res:
            r.update({"n_groups": len(keys), "best_val": best,
                      "best_epoch": best_ep, "model": str(model_path),
                      "train_groups": keys})
        results += res
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_dir / 'results.json'}")
    return 0


# --------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------

def report(args) -> int:
    rows = []
    for arm in ARMS:
        p = LADDER / arm / "results.json"
        if p.exists():
            rows += json.loads(p.read_text())
    if not rows:
        log("No results yet.")
        return 1

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    hdr = f"{'filter':7s} {'arm':16s} {'grp':>4s} {'corr/ceil':>10s} " \
          f"{'sources':>8s} {'flux':>7s} {'std':>7s} {'ideal':>7s}"
    log(hdr)
    log("-" * len(hdr))
    for filt in filters:
        for arm in ARMS:
            for r in rows:
                if r["filter"] != filt or r["arm"] != arm:
                    continue
                log(f"{filt:7s} {arm:16s} {r['n_groups']:4d} "
                    f"{100 * r['fraction_of_ceiling']:9.0f}% "
                    f"{100 * r['source_survival']:7.0f}% "
                    f"{r.get('flux_retained_median', float('nan')):7.3f} "
                    f"{r['std_ratio']:7.3f} {r['std_ratio_ideal']:7.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("stacks", "train", "report"))
    ap.add_argument("--arm", choices=ARMS, default="per-filter")
    ap.add_argument("--test", default="ngc5907",
                    help="held-out target — never trained on by any arm")
    ap.add_argument("--base", default="abell2151",
                    help="single training target for arms 1 and 2")
    ap.add_argument("--filters", default="L,R,G,B")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=("l1", "l2"), default="l2")
    ap.add_argument("--pairs", type=int, default=0,
                    help="patch pairs per epoch; 0 uses cfg['nn'] (2000)")
    ap.add_argument("--min-depth", type=int, default=6,
                    help="minimum frames per training stack for a group to count")
    ap.add_argument("--max-offset", type=int, default=2,
                    help="training pair offset above which a group is unusable")
    ap.add_argument("--verbose", action="store_true", help="stacker progress lines")
    # Smoke tests must not be able to leave anything behind that a later reader
    # could mistake for a result. A 2-epoch run writes the same filenames as a
    # 60-epoch one and its results.json looks entirely plausible — this project
    # has been fooled by plausible output more than once. --suffix moves the
    # whole lot (output dir, checkpoint name) somewhere harmless.
    ap.add_argument("--suffix", default="",
                    help="appended to the arm's output dir and model name; "
                         "use for smoke tests, e.g. --suffix _smoke")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)

    if args.stage == "stacks":
        return build_stacks(args)
    if args.stage == "train":
        return train_arm(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
