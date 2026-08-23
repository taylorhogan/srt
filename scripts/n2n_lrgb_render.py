#!/usr/bin/env python3
"""Render a DSO as a colour image, raw vs denoised, in any of the recipes.

    python scripts/n2n_lrgb_render.py stacks --dso ic1396 --recipe HOO --lum Ha
    python scripts/n2n_lrgb_render.py compose --dso ic1396 --recipe HOO --model <path>

Recipes come from `color_process.RECIPES` — LRGB, HOO (R<-Ha, G/B<-O-III) and
SHO. HOO points two output channels at one O-III stack, so the filter list is
de-duplicated before stacking; without that the same frames get stacked twice.

The ladder's own stacks cannot be used for this. It builds one shared reference
**per (dso, filter) group**, which is all N2N needs — pairs never cross filters —
but leaves the four channels of one target on four different pixel grids.
Measured on ngc5907: R sits (40, -48) px from L, G (0, -48), B (44, -56). Feeding
those to `color_process.compose` would put a coloured fringe on every star, which
is the precise failure its shared-reference argument exists to prevent.

So here every filter is registered onto **one** reference frame chosen from the
luminance channel, exactly as `color_process` does it.

Two other differences from the ladder's stacks, both deliberate:

- **Full depth, not split-half.** This is a display product, so all 208 frames
  are used rather than the ~2x17 a training pair needs.
- **Which means inference runs deeper than training.** The models learned on
  16-22 frame stacks and this is 41-77. Step 16 measured the *shallow* direction
  of that mismatch (training shallower than inference) as harmful, so this is
  the same axis and the raw/denoised pair below should be read as a display
  comparison, not as evidence about the model. A depth-matched render is the
  honest one for judging the network, and `--depth` produces it.
"""

import argparse
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

OUT = Path(_root) / "local" / "n2n_lrgb_render"


def suffix(args) -> str:
    """Depth-capped runs get their own filenames.

    The full-depth render is the display product and the depth-matched one is
    the fair test of the model, so both have to survive — overwriting either
    with the other loses the comparison that makes them worth having.
    """
    return f"_d{args.depth}" if args.depth else ""


def meta_path(args) -> Path:
    """Keyed by target AND recipe — the npy files already carry the dso in their
    names, so without this a second target silently overwrites the first's
    manifest while leaving its arrays in place."""
    return OUT / f"meta_{args.dso}_{args.recipe}{suffix(args)}.json"


def log(m: str = "") -> None:
    print(m, flush=True)


def recipe_filters(args) -> list[str]:
    """Distinct FITS filters this recipe needs, in a stable order.

    HOO maps one O-III stack onto both G and B, so the filter list is shorter
    than the channel list and must be de-duplicated before stacking — otherwise
    the same 38 frames get stacked twice.
    """
    if args.filters:
        return [f.strip() for f in args.filters.split(",") if f.strip()]
    from stacking import color_process
    mapping = color_process.RECIPES[args.recipe]
    seen, out = set(), []
    for ch in ("L", "R", "G", "B"):
        f = mapping.get(ch)
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def archive_calibration(cal_root: Path, filt: str, exptime: int, epoch: str):
    """Explicit bias/dark/flat for archive lights, matched by epoch and filter.

    Config-resolved calibration is wrong for archive data — it points at the
    current epoch, and applying 2026 masters taken at -10C to 2024 lights taken
    at -20C is worse than not calibrating. Flats are matched through
    `canonical_filter`, without which the 2024 O-III flats (labelled `O2`) are
    invisible.
    """
    from astropy.io import fits as _fits

    from stacking.color_process import canonical_filter

    want = canonical_filter(filt) or filt
    bias, dark, flat = [], [], []
    seen = set()
    for fp in sorted(Path(cal_root).rglob("*.fits")):
        if "RECYCLE" in str(fp):
            continue
        rp = fp.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        parts = [x.upper() for x in fp.parts]
        kind = next((k for k in ("BIAS", "DARK", "FLAT") if k in parts), None)
        if not kind:
            continue
        try:
            h = _fits.getheader(fp)
            if epoch and not str(h.get("DATE-OBS", "")).startswith(epoch):
                continue
            e = round(float(h.get("EXPTIME", 0)))
            f = str(h.get("FILTER", "")).strip()
        except Exception:
            continue
        if kind == "BIAS":
            bias.append(fp)
        elif kind == "DARK" and e == exptime:
            dark.append(fp)
        elif kind == "FLAT" and (canonical_filter(f) or f) == want:
            flat.append(fp)
    return bias, dark, flat


def build(args) -> int:
    from nn import stacks
    from stacking import stacker

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    subs = Path(args.root) if args.root else Path(machine["subs_dir"])
    filters = recipe_filters(args)
    idx = stacks.index_frames(subs, filters, args.exptime)

    # One reference for every channel, taken from the luminance frames. Sharing
    # it is the whole point: each filter otherwise picks its own and the
    # channels land tens of pixels apart.
    lum = args.lum if args.lum in filters else filters[0]
    lum_paths = idx.get((args.dso, lum))
    if not lum_paths:
        log(f"no {lum} frames for {args.dso}")
        return 1
    log(f"building shared reference from {len(lum_paths)} {lum} frames")
    ref = stacks.shared_reference_for(lum_paths, lum, progress_cb=lambda m: log(f"  {m}"))
    if ref is None:
        log("could not build a shared reference — refusing to stack onto "
            "per-filter references")
        return 1

    meta = {"dso": args.dso, "exptime": args.exptime, "lum": lum, "channels": {}}
    for filt in filters:
        paths = idx.get((args.dso, filt))
        if not paths:
            log(f"[{filt}] no frames — skipped")
            continue
        if args.depth:
            paths = paths[:args.depth]
        out = OUT / f"{args.dso}_{filt}_raw{suffix(args)}.npy"
        if out.exists() and not args.force:
            log(f"[{filt}] cached {out.name}")
            meta["channels"][filt] = {"frames": len(paths), "file": out.name}
            continue
        t0 = time.time()
        m: dict = {}
        if args.cal_root:
            bias, dark, flat = archive_calibration(Path(args.cal_root), filt,
                                                   args.exptime, args.cal_epoch)
            log(f"[{filt}] stacking {len(paths)} frames onto the {lum} reference "
                f"— calibration: {len(bias)} bias, {len(dark)} dark, {len(flat)} flat")
            if not flat:
                log(f"[{filt}] no epoch-matched flats — vignetting stays in")
            img, m = stacker.stack(
                list(paths), method=stacker.StackMethod.SIGMA_CLIP_FWHM,
                shared_reference=ref, bias_paths=bias or None,
                dark_paths=dark or None, flat_paths=flat or None,
                register=True, progress_cb=lambda s: None)
        else:
            log(f"[{filt}] stacking {len(paths)} frames onto the {lum} reference")
            img = stacks.stack_paths(paths, filt, progress_cb=lambda s: None,
                                     shared_reference=ref, meta_out=m)
        if img is None:
            log(f"[{filt}] stacking failed")
            continue
        np.save(out, img.astype(np.float32))
        log(f"[{filt}] {time.time() - t0:.0f}s, kept {m.get('n_frames')} of "
            f"{len(paths)}, shape {img.shape} -> {out.name}")
        meta["channels"][filt] = {"frames": len(paths),
                                  "accepted": m.get("n_frames"),
                                  "file": out.name}
        del img
    meta_path(args).write_text(json.dumps(meta, indent=2))
    log(f"\nwrote {meta_path(args)}")
    return 0


def compose(args) -> int:
    import torch

    from nn import denoiser
    from nn.noise2noise_model import UNet
    from stacking import color_process

    mp = meta_path(args)
    if not mp.exists():
        # Recipes share channels — SHO builds Ha and O-III, which is all HOO
        # needs — so fall back to any manifest for this target that covers them
        # rather than re-stacking identical frames under another name.
        need = set(recipe_filters(args))
        for cand in sorted(OUT.glob(f"meta_{args.dso}_*{suffix(args)}.json")):
            ch = json.loads(cand.read_text()).get("channels", {})
            if need <= set(ch):
                log(f"reusing channels from {cand.name}")
                mp = cand
                break
    meta = json.loads(mp.read_text())
    mapping = color_process.RECIPES[args.recipe]
    filters = [f for f in recipe_filters(args) if f in meta["channels"]]
    log(f"recipe {args.recipe}: " + ", ".join(
        f"{ch}<-{mapping[ch]}" for ch in ("L", "R", "G", "B") if ch in mapping))

    if not args.model:
        log("no --model: rendering the raw composite only")
        raw_ch = {}
        for f in filters:
            raw_ch[f] = np.load(OUT / meta["channels"][f]["file"]).astype(np.float32)
        h = min(a.shape[0] for a in raw_ch.values())
        w = min(a.shape[1] for a in raw_ch.values())
        raw_ch = {k: v[:h, :w] for k, v in raw_ch.items()}
        raw_ch = {ch: raw_ch[mapping[ch]] for ch in ("L", "R", "G", "B")
                  if ch in mapping and mapping[ch] in raw_ch}
        opts = color_process.effective_options(
            white_pct=args.white_pct, black_pct=args.black_pct)
        log(f"compose: {color_process.describe_options(opts)}")
        subbed, white = color_process._prepare(
            raw_ch, opts["subtract_background"], opts["mesh"], opts["white_pct"])
        blacks = {c: float(np.nanpercentile(subbed[c], opts["black_pct"]))
                  for c in subbed}
        log(f"stretch: white {white:.2f} ADU, blacks "
            + ", ".join(f"{c} {v:.2f}" for c, v in sorted(blacks.items())))
        soft = opts["softening"]

        def st(chan, black):
            y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
            return np.arcsinh(y / soft) / np.arcsinh(1.0 / soft)

        rgb = np.dstack([st(subbed[c], blacks[c]) for c in ("R", "G", "B")])
        if "L" in subbed:
            lum = st(subbed["L"], blacks["L"])
            rl = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                sc = np.where(rl > 1e-4, lum / np.maximum(rl, 1e-4), 0.0)
            rgb = rgb * np.clip(sc, 0.0, color_process.MAX_LUM_BOOST)[:, :, None]
        pth = color_process.save_rgb(np.clip(np.nan_to_num(rgb), 0, 1),
                                     OUT / f"{meta['dso']}_{args.recipe}_raw{suffix(args)}.jpg",
                                     max_px=args.max_px)
        log(f"wrote {pth}")
        return 0

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    if abs(float(ck.get("asinh_sigma_mult", denoiser.ASINH_SIGMA_MULT))
           - denoiser.ASINH_SIGMA_MULT) > 1e-9:
        log("checkpoint asinh_sigma_mult disagrees with the denoiser — refusing")
        return 1
    model = UNet(residual="linear")
    model.load_state_dict(ck["model_state"])
    model.eval()
    log(f"model {Path(args.model).name}: arm={ck.get('arm')} "
        f"epoch={ck.get('epoch')} groups={len(ck.get('groups', []) or [])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_ch, den_ch = {}, {}
    for f in filters:
        a = np.load(OUT / meta["channels"][f]["file"]).astype(np.float32)
        raw_ch[f] = a
        t0 = time.time()
        den_ch[f] = denoiser.denoise_frame(a, model, device=device)
        log(f"  denoised {f} in {time.time() - t0:.0f}s")

    # Crop every channel to the common shape before composing — a one-pixel
    # difference in stack extent would otherwise raise inside dstack.
    h = min(a.shape[0] for a in raw_ch.values())
    w = min(a.shape[1] for a in raw_ch.values())
    raw_ch = {k: v[:h, :w] for k, v in raw_ch.items()}
    den_ch = {k: v[:h, :w] for k, v in den_ch.items()}

    # Filter arrays -> output channels. HOO points both G and B at the same
    # O-III stack, which is what makes its palette teal rather than green.
    def to_channels(d: dict) -> dict:
        return {ch: d[mapping[ch]] for ch in ("L", "R", "G", "B")
                if ch in mapping and mapping[ch] in d}

    raw_ch, den_ch = to_channels(raw_ch), to_channels(den_ch)

    # compose()'s defaults are tuned for nebulae and put the white point at
    # p99 of max(R,G,B), which on this field is 12 ADU against a 32,000 ADU
    # star: the galaxy and every star saturate in BOTH images, and the raw then
    # looks *more* detailed only because its noise dithers the clipping edge.
    # Measured star profiles are identical to 0.1% at the core, so anything the
    # pair appears to show about sharpness at the default stretch is the
    # stretch, not the model.
    opts = color_process.effective_options(
        white_pct=args.white_pct, black_pct=args.black_pct)
    log(f"compose: {color_process.describe_options(opts)}")

    # ONE stretch, taken from the raw channels and applied unchanged to both.
    #
    # color_process.compose cannot be used directly for a raw-vs-denoised pair.
    # Its black point is a *percentile* of each channel (BLACK_PCT = 65), and
    # denoising shrinks the sky's spread ~2.5x, so p65 lands at a different ADU
    # on the denoised channel than on the raw one — and by a different amount
    # per channel, because each has its own noise. Structure that sat below
    # black in the raw rises above it, gets asinh-amplified, and the per-channel
    # differences turn into colour. The first render of this pair came out in
    # rainbow blotches for exactly that reason.
    #
    # This is the same error class as the per-image detection threshold that
    # `source_survival` was fixed for: a relative threshold applied to two
    # images of different noise measures the noise, not the signal.
    subbed_raw, white = color_process._prepare(
        raw_ch, opts["subtract_background"], opts["mesh"], opts["white_pct"])
    subbed_den, _ = color_process._prepare(
        den_ch, opts["subtract_background"], opts["mesh"], opts["white_pct"])
    blacks = {c: float(np.nanpercentile(subbed_raw[c], opts["black_pct"]))
              for c in subbed_raw}
    log(f"shared stretch from raw: white {white:.2f} ADU, blacks "
        + ", ".join(f"{c} {v:.2f}" for c, v in sorted(blacks.items())))

    def stretch_fixed(chan: np.ndarray, black: float) -> np.ndarray:
        y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
        soft = opts["softening"]
        return np.arcsinh(y / soft) / np.arcsinh(1.0 / soft)

    def build(subbed: dict) -> np.ndarray:
        rgb = np.dstack([stretch_fixed(subbed[c], blacks[c])
                         for c in ("R", "G", "B")])
        if opts["scnr"] > 0.0:
            rgb = color_process._scnr(rgb, float(np.clip(opts["scnr"], 0.0, 1.0)))
        if "L" in subbed:
            lum = stretch_fixed(subbed["L"], blacks["L"])
            rgb_lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                sc = np.where(rgb_lum > 1e-4, lum / np.maximum(rgb_lum, 1e-4), 0.0)
            rgb = rgb * np.clip(sc, 0.0, color_process.MAX_LUM_BOOST)[:, :, None]
        return np.clip(np.nan_to_num(rgb), 0.0, 1.0)

    for tag, subbed in (("raw", subbed_raw), ("denoised", subbed_den)):
        rgb = build(subbed)
        p = color_process.save_rgb(
            rgb, OUT / f"{meta['dso']}_{args.recipe}_{tag}{suffix(args)}.jpg",
            max_px=args.max_px)
        log(f"wrote {p}")
        del rgb
    return 0


DOMAIN_MODELS = {
    # Domain-matched by default. The narrowband-trained model beat the
    # broadband one in every bin on both filters when applied to narrowband
    # (lab manual step 23), so picking by domain is not a convenience.
    "narrowband": "local/models/n2n_pooledNB_300s.pt",
    "broadband": "local/models/n2n_ladder_pooled-filters_300s.pt",
}


def pick_model(filters) -> tuple:
    nb = {"Ha", "O-III", "S-II"}
    domain = "narrowband" if set(filters) & nb else "broadband"
    return domain, os.path.join(_root, DOMAIN_MODELS[domain])


def routine(args) -> int:
    """Stack, export each channel, denoise, export again, compose both.

    The nightly path for a target that already has a suitable model: no
    training, no held-out scoring, just the products. Everything lands in
    `<image_dir>/Iris/<dso>/` beside that target's lights.

    **Denoised output is a display product for point-source-dominated fields
    only.** On extended emission the model is band-limited — it keeps 96-98% of
    structure above 512 px and 30-50% of the fine texture that makes a nebula
    read as one (lab manual step 29). Both composites are written so the pair
    can be compared rather than the denoised one silently replacing the raw.
    """
    from stacking import color_process, stacker

    if build(args) != 0:
        return 1
    meta = json.loads(meta_path(args).read_text())
    mapping = color_process.RECIPES[args.recipe]
    filters = [f for f in recipe_filters(args) if f in meta["channels"]]
    if not filters:
        log("no channels built — nothing to render")
        return 1

    domain, model_path = pick_model(filters)
    if args.model:
        model_path = args.model
    log("")
    log(f"recipe {args.recipe}  filters {filters}  domain {domain}")
    log(f"model {os.path.basename(model_path)}")
    if not os.path.exists(model_path):
        log(f"  missing — raw products only")
        model_path = ""

    # results_dir() creates its own; an explicit --out-dir has to be made here.
    out_dir = Path(args.out_dir) if args.out_dir else stacker.results_dir(args.dso)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"products -> {out_dir}")

    raw = {f: np.load(OUT / meta["channels"][f]["file"]).astype(np.float32)
           for f in filters}
    h = min(a.shape[0] for a in raw.values())
    w = min(a.shape[1] for a in raw.values())
    raw = {k: v[:h, :w] for k, v in raw.items()}

    den = {}
    if model_path:
        import torch

        from nn import denoiser
        from nn.noise2noise_model import UNet
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        m = UNet(residual="linear")
        m.load_state_dict(ck["model_state"])
        m.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for f in filters:
            t0 = time.time()
            den[f] = denoiser.denoise_frame(raw[f], m, device=dev)[:h, :w]
            log(f"  denoised {f} in {time.time() - t0:.0f}s")

    def to_channels(d):
        return {ch: d[mapping[ch]] for ch in ("L", "R", "G", "B")
                if ch in mapping and mapping[ch] in d}

    opts = color_process.effective_options(
        white_pct=args.white_pct, black_pct=args.black_pct)
    log(f"compose: {color_process.describe_options(opts)}")

    # ONE stretch, from the raw, applied to both sets — so the channel JPEGs and
    # the two composites are all on the same scale and every visible difference
    # between raw and denoised is the model's. compose() would instead pick
    # black per channel by percentile, which lands at a different ADU on a
    # denoised channel and manufactures a colour shift.
    subbed_raw, white = color_process._prepare(
        to_channels(raw), opts["subtract_background"], opts["mesh"], opts["white_pct"])
    blacks = {c: float(np.nanpercentile(subbed_raw[c], opts["black_pct"]))
              for c in subbed_raw}
    soft = opts["softening"]
    log(f"stretch from raw: white {white:.2f} ADU, blacks "
        + ", ".join(f"{c} {v:.2f}" for c, v in sorted(blacks.items())))

    def st(chan, black):
        y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
        return np.arcsinh(y / soft) / np.arcsinh(1.0 / soft)

    from PIL import Image
    written = []

    def emit(subbed, tag):
        for ch in ("R", "G", "B", "L"):
            if ch not in subbed:
                continue
            mono = st(subbed[ch], blacks[ch])
            arr = (np.clip(np.nan_to_num(mono), 0, 1) * 255).astype(np.uint8)[::-1]
            img = Image.fromarray(arr, mode="L")
            mx = color_process.CHANNEL_JPG_MAX_PX
            if max(img.size) > mx:
                r_ = mx / max(img.size)
                img = img.resize((int(img.width * r_), int(img.height * r_)),
                                 Image.LANCZOS)
            pth = out_dir / f"{args.dso}_{args.recipe}_{tag}_{mapping[ch]}.jpg"
            img.save(pth, quality=92, optimize=True)
            written.append(pth)
        rgb = np.dstack([st(subbed[c], blacks[c]) for c in ("R", "G", "B")])
        if opts["scnr"] > 0:
            rgb = color_process._scnr(rgb, float(np.clip(opts["scnr"], 0, 1)))
        if "L" in subbed:
            lum = st(subbed["L"], blacks["L"])
            rl = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                sc = np.where(rl > 1e-4, lum / np.maximum(rl, 1e-4), 0.0)
            rgb = rgb * np.clip(sc, 0.0, color_process.MAX_LUM_BOOST)[:, :, None]
        pth = color_process.save_rgb(np.clip(np.nan_to_num(rgb), 0, 1),
                                     out_dir / f"{args.dso}_{args.recipe}_{tag}.jpg",
                                     max_px=args.max_px)
        written.append(pth)

    emit(subbed_raw, "raw")
    if den:
        subbed_den, _ = color_process._prepare(
            to_channels(den), opts["subtract_background"], opts["mesh"],
            opts["white_pct"])
        emit(subbed_den, "denoised")

    log("")
    for pth in written:
        log(f"  wrote {pth.name}")
    log("")
    log(f"{len(written)} files in {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stacks", "compose", "routine"))
    ap.add_argument("--dso", default="ngc5907")
    ap.add_argument("--recipe", default="LRGB",
                    choices=("LRGB", "HOO", "SHO", "HSO"))
    ap.add_argument("--filters", default="",
                    help="override the recipe's filter list")
    ap.add_argument("--lum", default="",
                    help="filter supplying the shared reference; default is the "
                         "recipe's first, which should be the deepest channel")
    ap.add_argument("--exptime", type=int, default=300)
    # Takes the FIRST n frames, not a random sample, so a capped channel is
    # typically one night rather than a draw across sessions. That is a second
    # variable on top of depth: on ngc5907 --depth 22 left R with 11 frames
    # after the quality gate (50% cut) against 76% kept at full depth, because
    # the first 22 R frames happened to come from a poorer session.
    ap.add_argument("--depth", type=int, default=0,
                    help="cap frames per channel, taking the FIRST n (one "
                         "session, not a spread); 0 uses all")
    ap.add_argument("--model", default="",
                    help="checkpoint for the compose stage; omit to render only "
                         "the raw composite")
    ap.add_argument("--max-px", type=int, default=2400)
    ap.add_argument("--white-pct", type=float, default=99.95,
                    help="shared white point percentile; compose()'s 99.0 "
                         "saturates this field completely")
    ap.add_argument("--black-pct", type=float, default=40.0)
    ap.add_argument("--root", default="",
                    help="where to find LIGHT frames; default is this machine's subs_dir")
    ap.add_argument("--cal-root", default="",
                    help="take bias/dark/flat from here instead of config — "
                         "required for archive data, whose epoch config does not know")
    ap.add_argument("--cal-epoch", default="",
                    help="restrict calibration to frames whose DATE-OBS starts with this")
    ap.add_argument("--out-dir", default="",
                    help="default is <image_dir>/Iris/<dso>/")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)

    if args.stage == "routine":
        return routine(args)
    if args.stage == "stacks":
        return build(args)
    return compose(args)


if __name__ == "__main__":
    sys.exit(main())
