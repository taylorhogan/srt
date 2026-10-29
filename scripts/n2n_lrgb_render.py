#!/usr/bin/env python3
"""Render a DSO as a colour image -- raw, denoised and deblurred -- in any of the recipes.

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

The routine path writes THREE composites: raw, denoised, and (since the
afternoon of 2026-10-21, the owner's choice by eye on ngc6946 and ngc1961)
"learned" -- the denoised set (the raw one when no model applies) with its
LUMINANCE through the learned deconvolver (nn/deblur_model, the field's own
kernel, flux conserved) and the colour planes left as denoised. Deconvolving
R, G, B and Ha too was tried first and sharpened each plane's own noise into
coloured specks over the sky and turned the brightest star's core cyan
(ngc6946, 2026-10-21 noon), so `--learned-planes` is "L" by default and
"all" is the switch back. `--no-learned` turns it off and a missing
checkpoint skips it with a line in the log. The starless composite that was
the third from 2026-10-19 to 10-21 (`stacking/starless`: stars off, RL on
the luminance, stars back as imaged) is an optional fourth, "deblurred",
with `--deblur N`. All share the raw set's stretch, white balance and Ha
ratio, so what differs between them is the denoiser and the deconvolution
and nothing else.
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

# The image classes a routine render writes, in pipeline order. "learned"
# (2026-10-21) is the fourth: the denoised set through nn/deblur_model, the
# flux-moving network trained on re-blurred stacks of this telescope, beside
# the starless one so the two deconvolutions can be judged on every morning
# target before either is made the only one.
TAGS = ("raw", "denoised", "learned")
# Optional extras, written and pushed only when asked for (--deblur N adds
# the starless composite back for comparison).
EXTRA_TAGS = ("deblurred",)
# What the morning push sends, best first: ONE picture (owner, 2026-10-28:
# "you only need to send a picture of the learned, not all 3"), the learned
# composite, or the next class down when a field refused the deconvolution.
# --push-all restores the triple. Extras asked for are still sent.
PUSH_TAGS = ("learned", "denoised", "raw")


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


def stack_record(dso: str, out: Path) -> dict:
    """What the stack that wrote *out* recorded about itself (accepted
    count, coverage box, frame lists): the sidecar beside the npy, or for a
    stack from before the sidecar (2026-10-29), the newest entry naming the
    file in ANY recipe's meta of the target. The record belongs to the
    stack, not to the recipe: the morning's SHO rebuilt Melotte 15's stacks
    with coverage boxes and the HOO after it, reading only its own meta
    from the day before, rendered uncropped."""
    side = out.with_suffix(".json")
    try:
        rec = json.loads(side.read_text())
        if rec.get("file") == out.name:
            return rec
    except (OSError, ValueError):
        pass
    best: dict = {}
    for mp in OUT.glob(f"meta_{dso}_*.json"):
        try:
            chans = json.loads(mp.read_text()).get("channels", {})
        except (OSError, ValueError):
            continue
        for entry in chans.values():
            if entry.get("file") == out.name and entry.get("built", "") >= best.get("built", ""):
                best = entry
    return best


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
    for ch in ("L", "R", "G", "B", "HA"):
        f = mapping.get(ch)
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    # --stars rgb: the star-colour stacks ride along, on the same reference.
    # They are never palette channels; routine() keeps them out of to_channels.
    if getattr(args, "stars", "none") == "rgb":
        for f in ("R", "G", "B"):
            if f not in seen:
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
            # Keep what the stack that wrote this file recorded (its frame
            # lists, its coverage box); only the offered count is today's.
            entry = dict(stack_record(args.dso, out))
            entry.update({"frames": len(paths), "file": out.name})
            meta["channels"][filt] = entry
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
        # The frame lists are the stack's own record of what went in, for
        # anything that reports on the picture later (scripts/astro_output
        # writes the per-filter counts from them; a replay of the gate from
        # frame_stats.json differs by a few frames, 2026-10-22, because the
        # stacker measures the frames itself).
        rejected = {Path(p) for p in m.get("rejected", [])}
        meta["channels"][filt] = {"frames": len(paths),
                                  "accepted": m.get("n_frames"),
                                  "file": out.name,
                                  "coverage": m.get("coverage_box"),
                                  "built": time.strftime("%Y-%m-%dT%H:%M"),
                                  "used": [Path(p).name for p in paths if Path(p) not in rejected],
                                  "rejected": [Path(p).name for p in sorted(rejected)]}
        out.with_suffix(".json").write_text(json.dumps(meta["channels"][filt], indent=2))
        del img
    meta_path(args).write_text(json.dumps(meta, indent=2))
    log(f"\nwrote {meta_path(args)}")
    return 0



def sky_anchored_blacks(subbed: dict, black_sigma: float, black_pct: float,
                        log=print, gains=None) -> dict:
    """Lives in color_process now; kept here as the name every caller uses."""
    from stacking import color_process
    return color_process.sky_anchored_blacks(subbed, black_sigma, black_pct, log=log,
                                             gains=gains)


def shared_from_raw(raw_info: dict) -> dict:
    """_prepare kwargs that pin the denoised set to the raw set's measurements.

    The white balance is passed as factors (white_balance stays "none" so
    nothing is re-measured); the Ha continuum ratio likewise. A balance that
    failed on the raw set comes back as the identity, and the denoised set gets
    the identity too: the pair stays consistent either way.
    """
    wb = (raw_info.get("wb") or {}).get("factors") or None
    return {"white_balance": "none", "wb_factors": wb, "ha_ratio": raw_info.get("ha_ratio"),
            "palette_gain": raw_info.get("palette_gain"),
            "palette_stars": raw_info.get("palette_stars"),
            "palette_arms": raw_info.get("palette_arms")}


def common_box(meta: dict, filters, shape) -> tuple:
    """(y0, y1, x0, x1): the intersection of the filters' coverage boxes.

    A stack built onto a shared reference is not cropped by the stacker
    (its box would differ per filter); outside its footprint it is a flat
    fill at the global sky level, which at a vignetted edge sits above the
    real sky beside it and, through a palette gain, became a red bar along
    the top of every IC 410 render (owner, 2026-10-28). So every channel is
    cut to the area at least 80 % of EVERY filter's frames covered. A
    filter whose record has no box (a stack cached before this) counts as
    full-frame, and the stack has to be rebuilt (--force) to get one.
    """
    h, w = shape
    y0, y1, x0, x1 = 0, h, 0, w
    for f in filters:
        box = (meta.get("channels", {}).get(f) or {}).get("coverage")
        if not box:
            continue
        by0, by1, bx0, bx1 = (int(v) for v in box)
        y0, y1, x0, x1 = max(y0, by0), min(y1, by1), max(x0, bx0), min(x1, bx1)
    if y1 <= y0 or x1 <= x0:
        return 0, h, 0, w
    return y0, y1, x0, x1


def crop_common(chans: dict, meta: dict, log) -> dict:
    """Every channel cut to common_box; logged when anything was cut."""
    h = min(a.shape[0] for a in chans.values())
    w = min(a.shape[1] for a in chans.values())
    y0, y1, x0, x1 = common_box(meta, list(chans), (h, w))
    missing = [f for f in chans if not (meta.get("channels", {}).get(f) or {}).get("coverage")]
    if missing:
        log(f"common footprint: no coverage box for {', '.join(missing)} "
            f"(stacked before 2026-10-28; --force rebuilds it) — not cut on their account")
    if (y0, y1, x0, x1) != (0, h, 0, w):
        log(f"common footprint: rows {y0}:{y1}, cols {x0}:{x1} of {h}x{w} "
            f"(cut {y0} top, {h - y1} bottom, {x0} left, {w - x1} right)")
    return {k: np.ascontiguousarray(v[y0:y1, x0:x1]) for k, v in chans.items()}


def palette_gain_opt(args):
    """--gain as the recipe's {role: factor} or "hubble"; None when not asked."""
    spec = getattr(args, "gain", "") or ""
    if not spec:
        return None
    from stacking import color_process
    return color_process.palette_gain_roles(args.recipe, spec)


def log_shared_measurements(raw_info: dict, log) -> None:
    from stacking import color_process
    pg = raw_info.get("palette_gain")
    if pg:
        log("    palette gain: " + "  ".join(f"{c} x{f:.3f}" for c, f in sorted(pg.items()))
            + f"; {len(raw_info.get('palette_stars') or [])} stars found, kept out of the gain ({color_process.PALETTE_STAR_MODE})"
            + f", spikes on {len(raw_info.get('palette_arms') or [])}")
    wb = raw_info.get("wb") or {}
    if wb.get("mode") == "stars":
        f = wb.get("factors") or {}
        log("white balance from raw (%s): R x%.3f  B x%.3f — pinned on the denoised set"
            % (wb.get("why") or "%s stars" % wb.get("stars", "?"),
               f.get("R", 1.0), f.get("B", 1.0)))
    if raw_info.get("ha_ratio") is not None:
        log("Ha continuum ratio from raw: %.4f — pinned on the denoised set"
            % raw_info["ha_ratio"])


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
        f"{ch}<-{mapping[ch]}" for ch in ("L", "R", "G", "B", "HA") if ch in mapping))

    if not args.model:
        log("no --model: rendering the raw composite only")
        raw_ch = {}
        for f in filters:
            raw_ch[f] = np.load(OUT / meta["channels"][f]["file"]).astype(np.float32)
        raw_ch = crop_common(raw_ch, meta, log)
        raw_ch = {ch: raw_ch[mapping[ch]] for ch in ("L", "R", "G", "B", "HA")
                  if ch in mapping and mapping[ch] in raw_ch}
        opts = color_process.effective_options(
            white_pct=args.white_pct, black_pct=args.black_pct,
            palette_gain=palette_gain_opt(args),
        **({"softening": args.soft} if args.soft is not None else {}))
        log(f"compose: {color_process.describe_options(opts)}")
        subbed, white = color_process._prepare(
            raw_ch, opts["subtract_background"], opts["mesh"], opts["white_pct"],
            palette_gain=opts["palette_gain"])
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
    model = UNet.matching(ck["model_state"], residual="linear")
    model.eval()
    log(f"model {Path(args.model).name}: arm={ck.get('arm')} "
        f"epoch={ck.get('epoch')} groups={len(ck.get('groups', []) or [])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_ch, den_ch = {}, {}
    raw_ch = crop_common({f: np.load(OUT / meta["channels"][f]["file"]).astype(np.float32)
                          for f in filters}, meta, log)
    h, w = next(iter(raw_ch.values())).shape          # the denoiser may pad; cut back to this
    for f in filters:
        a = raw_ch[f]
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
        return {ch: d[mapping[ch]] for ch in ("L", "R", "G", "B", "HA")
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
        white_pct=args.white_pct, black_pct=args.black_pct, white_balance=args.wb,
        palette_gain=palette_gain_opt(args),
        **({"softening": args.soft} if args.soft is not None else {}),
        **({"scnr": args.scnr} if args.scnr is not None else {}))
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
    # The white balance and the Ha continuum ratio are measured on the raw set
    # and pinned on the denoised one, like the stretch (see _prepare).
    raw_info = {}
    subbed_raw, white = color_process._prepare(
        raw_ch, opts["subtract_background"], opts["mesh"], opts["white_pct"],
        white_balance=opts["white_balance"], info=raw_info,
        palette_gain=opts["palette_gain"])
    log_shared_measurements(raw_info, log)
    subbed_den, _ = color_process._prepare(
        den_ch, opts["subtract_background"], opts["mesh"], opts["white_pct"],
        **shared_from_raw(raw_info))
    if args.auto_stretch:
        log("auto stretch:")
        auto = color_process.auto_stretch(subbed_raw, white, lum="L", log=log)
        blacks = auto["blacks"]
        opts["softening"] = auto["softening"]
    else:
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
    # 2026-10-22: the bilinear-decoder `groups` retrain on twelve narrowband
    # groups (lab manual step 39). pooledNB's ConvTranspose2d decoder left a
    # 4 px / 2 px checkerboard at x410 the radial power on smooth emission;
    # this one is at the raw stack's level in every cell, and on the held-out
    # trunk pairs its N2N loss is the best of the four arms. It keeps
    # 0.03-0.06 less extended flux than pooledNB, which the transposed twin
    # (same data, same recipe) also does, so that is the recipe, not the
    # decoder, and the objective does not see it.
    "narrowband": "local/models/n2n_ladder_groups_narrowband_up_300s.pt",
    # 2026-10-09: the `groups` retrain on Abell 2151 LRGB + M33 L + NGC 7320
    # LRGB + NGC 2146 LRGB (lab manual step 38). The pooled-filters model it
    # replaces trained on Abell 2151 alone and smoothed resolved galaxies as
    # noise (step 37).
    "broadband": "local/models/n2n_ladder_groups_galaxies_300s.pt",
}


_NARROWBAND_FILTERS = {"Ha", "O-III", "S-II"}


def domain_for(filt: str) -> str:
    """Which model a single filter's stack is denoised with."""
    return "narrowband" if filt in _NARROWBAND_FILTERS else "broadband"


def pick_model(filters) -> tuple:
    """One (domain, model path) for a whole filter set — the pre-HALRGB rule,
    kept for callers that render a single-domain set. A mixed set (HALRGB:
    L R G B + Ha) must not go through this: it would hand the narrowband
    model every broadband channel. routine() picks per filter instead."""
    domain = "narrowband" if set(filters) & _NARROWBAND_FILTERS else "broadband"
    return domain, os.path.join(_root, DOMAIN_MODELS[domain])



# Pushover caps an attachment at 2.5 MB. The composite is written at --max-px
# (2400 by default) and lands around 0.5-1 MB, but a dense field can exceed the
# cap, and Pushover rejects an oversized attachment with a healthy-looking HTTP
# response — the same silent-failure mode utils/pushover.py exists to avoid. So
# shrink until it fits rather than trusting it to.
_PUSH_MAX_BYTES = 2_500_000


def _fit_for_push(src: Path, work: Path) -> Path:
    """Return a path whose JPEG is under the attachment cap."""
    if src.stat().st_size <= _PUSH_MAX_BYTES:
        return src
    from PIL import Image
    img = Image.open(src)
    for max_px in (1800, 1400, 1000, 700):
        img2 = img.copy()
        r = max_px / max(img2.size)
        if r < 1.0:
            img2 = img2.resize((int(img2.width * r), int(img2.height * r)),
                               Image.LANCZOS)
        out = work / f"push_{src.stem}_{max_px}.jpg"
        img2.save(out, quality=88, optimize=True)
        if out.stat().st_size <= _PUSH_MAX_BYTES:
            return out
    return out


def push_render(args, out_dir: Path, written: list, meta: dict, log=print) -> bool:
    """Pushover the composite(s): by default the one best picture (the first
    of PUSH_TAGS that was written), with --push-all every class in TAGS,
    plus any extra one that was written. Never raises -- a render must not
    fail because a notification did.

    A PAIR since 2026-09-08, a triple since 2026-10-19 while the two
    deconvolutions were being judged against the denoised set on every
    morning target; one picture since 2026-10-28, the learned composite
    being the default product. Pushover takes one attachment per message,
    so each is its own message. Returns True if at least one was accepted.
    """
    try:
        from utils import pushover
        chans = (meta or {}).get("channels", {}) or {}
        parts = [f"{k} {v.get('accepted', '?')}/{v.get('frames', '?')}"
                 for k, v in sorted(chans.items())]
        total = sum(int(v.get("accepted", 0)) for v in chans.values())
        tail = (", ".join(parts)
                + (f" ({total} frames x {args.exptime}s)" if total else ""))
        sent = 0
        extras = tuple(t for t in EXTRA_TAGS
                       if any(p.name == f"{args.dso}_{args.recipe}_{t}.jpg" for p in written))
        if getattr(args, "push_all", False):
            tags = TAGS + extras
        else:
            best = [t for t in PUSH_TAGS
                    if any(p.name == f"{args.dso}_{args.recipe}_{t}.jpg" for p in written)]
            tags = tuple(best[:1]) + extras
            if not best:
                log("  push: no composite to send")
        for tag in tags:
            pick = [p for p in written if p.name == f"{args.dso}_{args.recipe}_{tag}.jpg"]
            if not pick:
                log(f"  push: no {tag} composite to send")
                continue
            img = _fit_for_push(pick[0], out_dir)
            msg = f"{args.dso} {args.recipe} {tag} - {tail}"
            ok = pushover.push_message_with_picture(msg, str(img))
            log(f"  push {tag}: {'sent' if ok else 'NOT ACCEPTED'} ({img.name}, "
                f"{img.stat().st_size / 1024:.0f} KB)")
            sent += bool(ok)
        return sent > 0
    except Exception as e:  # noqa: BLE001
        log(f"  push failed: {e}")
        return False

def deblurred_channels(channels: dict, iters: int, log=print, stars: str = "imaged"):
    """The channel set through stacking/starless, or None with the reason logged.

    Never raises: this is an optional extra, and a target must not lose its
    raw, denoised and learned renders to a field the deconvolution cannot
    handle.
    """
    from stacking import color_process, deblur as _deblur
    t0 = time.time()
    log("")
    log(f"deblur: starless RL{int(iters)} on {', '.join(channels)}, stars {stars}")
    try:
        out, infos = _deblur.deblur_channels(channels, int(iters), clamp_to=color_process.DEBLUR_DERING,
                                             method="starless", log=lambda m: log(f"  {m}"), stars=stars)
    except Exception as e:  # noqa: BLE001
        log(f"  deblur failed — no deblurred composite: {e!r}")
        return None
    why = [f"{c}: {i['why']}" for c, i in infos.items() if i.get("why")]
    if not infos or all(i.get("why") for i in infos.values()):
        log(f"  deblur refused — no deblurred composite: {'; '.join(why) or 'nothing done'}")
        return None
    log(f"  deblurred in {time.time() - t0:.0f}s")
    return out


def learned_channels(channels: dict, model_path: str, log=print, planes=("L",), iters=40,
                     no_l=None):
    """(channel set, tag): *planes* through nn/deblur_model and the rest as
    given under tag "learned"; (None, "learned") with the reason logged when
    nothing could be done. A set with no luminance (HOO/SHO) follows
    *no_l* (None: stacking/deblur.NO_L_RULE, synthetic since 2026-10-22
    evening): "synthetic" makes a luminance, sharpens it and returns it
    beside its unsharpened self as LSYN / LSYN_REF, the composite scaled by
    their ratio; "starless" (and a synthetic path that refuses) comes back
    under tag "deblurred" (RL *iters*), so the product's name says which
    deconvolution made it.

    Never raises: a field the model cannot handle (no measurable kernel, no
    checkpoint, a failure inside) must not cost the target its raw and
    denoised renders.
    """
    t0 = time.time()
    log("")
    if not model_path or not os.path.exists(model_path):
        log(f"learned deblur: no checkpoint at {model_path or '(none)'} — no learned composite")
        return None, "learned"
    log(f"learned deblur: {os.path.basename(os.path.realpath(model_path))} on {', '.join(channels)}, "
        f"planes {'all' if planes is None else ','.join(planes)}")
    try:
        import torch
        from nn import deblur_model
        from stacking import deblur as _deblur
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = deblur_model.load(model_path, dev)
        # The same call `process` makes (stacking/deblur, method="learned"):
        # one rule for which planes are sharpened, one dedupe of shared planes.
        out, infos = _deblur.deblur_channels(channels, int(iters), method="learned", model=model,
                                             planes=planes, no_l=no_l, log=lambda m: log(f"  {m}"))
    except Exception as e:  # noqa: BLE001
        log(f"  learned deblur failed — no learned composite: {e!r}")
        return None, "learned"
    tag = "deblurred" if any(i.get("fallback") for i in infos.values()) else "learned"
    done = [c + (f" (synthetic, {i['synthetic']})" if i.get("synthetic") else "")
            for c, i in infos.items() if not i.get("why") and not i.get("kept")]
    kept = [c for c, i in infos.items() if i.get("kept")]
    why = [f"{c}: {i['why']}" for c, i in infos.items() if i.get("why")]
    if not done:
        log(f"  learned deblur refused — no {tag} composite: {'; '.join(why) or 'nothing done'}")
        return None, tag
    log(f"  sharpened {', '.join(done)}" + (f"; kept as given {', '.join(kept)}" if kept else "")
        + (" -- written as the 'deblurred' composite (starless fallback)" if tag == "deblurred" else ""))
    if why:
        log(f"  passed through unchanged: {'; '.join(why)}")
    log(f"  learned deblur in {time.time() - t0:.0f}s")
    return out, tag


def routine(args) -> int:
    """Stack, export each channel, denoise, deblur, compose all three.

    The nightly path for a target that already has a suitable model: no
    training, no held-out scoring, just the products. Everything lands in
    `<image_dir>/Iris/<dso>/` beside that target's lights.

    **Denoised output is a display product for point-source-dominated fields
    only.** On extended emission the model is band-limited — it keeps 96-98% of
    structure above 512 px and 30-50% of the fine texture that makes a nebula
    read as one (lab manual step 29). All three composites are written so they
    can be compared rather than one silently replacing another. The deblurred
    one is the denoised set (raw if there is no model) through the starless
    deconvolution; if that refuses or fails, the other two are still written
    and the log says why.
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

    # Model per FILTER, not per recipe: HALRGB mixes broadband L R G B with
    # narrowband Ha, and each stack gets the model trained on its own domain.
    # --model overrides every filter with the one file.
    model_by_filter: dict[str, str] = {}
    for f in filters:
        mp_ = args.model or os.path.join(_root, DOMAIN_MODELS[domain_for(f)])
        model_by_filter[f] = mp_ if os.path.exists(mp_) else ""
    log("")
    log(f"recipe {args.recipe}  filters {filters}")
    for f in filters:
        log(f"  {f}: {domain_for(f)} -> "
            + (os.path.basename(model_by_filter[f]) or "model missing — raw only"))
    model_path = next((m for m in model_by_filter.values() if m), "")

    # results_dir() creates its own; an explicit --out-dir has to be made here.
    out_dir = Path(args.out_dir) if args.out_dir else stacker.results_dir(args.dso)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"products -> {out_dir}")

    raw = crop_common({f: np.load(OUT / meta["channels"][f]["file"]).astype(np.float32)
                       for f in filters}, meta, log)
    h, w = next(iter(raw.values())).shape             # the denoiser may pad; cut back to this

    # Star colour from RGB: mask and star image are built once from the raw
    # R, G, B stacks and applied to every composite, raw and denoised alike.
    star_mask_arr, star_img = None, None
    palette_filters = {mapping[c] for c in mapping}
    if args.stars == "rgb":
        if all(f in raw for f in ("R", "G", "B")):
            sst = {c: raw[c] for c in ("R", "G", "B")}
            star_mask_arr, sinfo = color_process.star_mask(sst["R"] + sst["G"] + sst["B"])
            star_img = color_process.stars_image(
                sst, white_pct=args.white_pct,
                **({"softening": args.soft} if args.soft is not None else {}))
            log(f"stars from RGB: {sinfo['stars']} stars, {100 * sinfo['coverage']:.2f}% of the field")
        else:
            log("--stars rgb: no R, G and B stacks — stars stay palette")
        # the star stacks are not palette channels: never denoise or compose them
        filters = [f for f in filters if f in palette_filters]

    den = {}
    if model_path:
        import torch

        from nn import denoiser
        from nn.noise2noise_model import UNet
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        loaded: dict[str, UNet] = {}
        for f in filters:
            mp_ = model_by_filter.get(f)
            if not mp_:
                continue
            if mp_ not in loaded:
                ck = torch.load(mp_, map_location="cpu", weights_only=False)
                m = UNet.matching(ck["model_state"], residual="linear")
                m.eval()
                loaded[mp_] = m
            t0 = time.time()
            den[f] = denoiser.denoise_frame(raw[f], loaded[mp_], device=dev)[:h, :w]
            log(f"  denoised {f} ({domain_for(f)}) in {time.time() - t0:.0f}s")

    def to_channels(d):
        return {ch: d[mapping[ch]] for ch in ("L", "R", "G", "B", "HA")
                if ch in mapping and mapping[ch] in d}

    opts = color_process.effective_options(
        white_pct=args.white_pct, black_pct=args.black_pct, white_balance=args.wb,
        palette_gain=palette_gain_opt(args),
        **({"softening": args.soft} if args.soft is not None else {}),
        **({"scnr": args.scnr} if args.scnr is not None else {}))
    log(f"compose: {color_process.describe_options(opts)}")

    # ONE stretch, from the raw, applied to both sets — so the channel JPEGs and
    # the two composites are all on the same scale and every visible difference
    # between raw and denoised is the model's. compose() would instead pick
    # black per channel by percentile, which lands at a different ADU on a
    # denoised channel and manufactures a colour shift.
    raw_info = {}
    subbed_raw, white = color_process._prepare(
        to_channels(raw), opts["subtract_background"], opts["mesh"], opts["white_pct"],
        white_balance=opts["white_balance"], info=raw_info,
        palette_gain=opts["palette_gain"],
        # the gain's faint stars are found on the denoised planes (and the
        # bright ones on the raw), pinned on every set from here
        palette_quiet=to_channels(den) if den else None)
    log_shared_measurements(raw_info, log)
    if args.auto_stretch:
        log("auto stretch:")
        auto = color_process.auto_stretch(subbed_raw, white, lum="L", log=log)
        blacks, soft = auto["blacks"], auto["softening"]
    else:
        blacks = sky_anchored_blacks(subbed_raw, args.black_sigma, opts["black_pct"], log,
                                     gains=raw_info.get("palette_gain"))
        soft = opts["softening"]
    log(f"stretch from raw: white {white:.2f} ADU, blacks "
        + ", ".join(f"{c} {v:.2f}" for c, v in sorted(blacks.items())))

    def st(chan, black):
        y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
        return np.arcsinh(y / soft) / np.arcsinh(1.0 / soft)

    from PIL import Image
    written = []

    from stacking import stacker

    def emit(subbed, tag):
        # HALRGB: R and L are the blended planes; HA is the raw Ha stack and
        # HA_EXCESS the emission map that was added (black at 0: it is
        # zero-floored, so it has no sky to anchor to).
        names = {**mapping, "HA_EXCESS": "Ha_excess", "LSYN": "Lsyn"}
        if "LSYN" in subbed and "LSYN" not in blacks:
            # A synthetic luminance (stacking/deblur.synthetic_luminance, a
            # set with no L): one black for the pair, the same weighted sum
            # of the planes' blacks, so it sits on the pinned stretch.
            from stacking.deblur import LUM_WEIGHTS
            present = {c: LUM_WEIGHTS[c] for c in ("R", "G", "B") if c in blacks}
            tot = sum(present.values()) or 1.0
            blacks["LSYN"] = blacks["LSYN_REF"] = sum(w / tot * blacks[c] for c, w in present.items())
            log(f"  synthetic L black {blacks['LSYN']:.2f} ADU from the planes' blacks")
        for ch in ("R", "G", "B", "L", "HA", "HA_EXCESS", "LSYN"):
            if ch not in subbed:
                continue
            mono = st(subbed[ch], blacks.get(ch, 0.0))
            arr = stacker.sky_parity((np.clip(np.nan_to_num(mono), 0, 1) * 255).astype(np.uint8))
            img = Image.fromarray(arr, mode="L")
            mx = color_process.CHANNEL_JPG_MAX_PX
            if max(img.size) > mx:
                r_ = mx / max(img.size)
                img = img.resize((int(img.width * r_), int(img.height * r_)),
                                 Image.LANCZOS)
            pth = out_dir / f"{args.dso}_{args.recipe}_{tag}_{names[ch]}.jpg"
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
        elif "LSYN" in subbed and "LSYN_REF" in subbed:
            rgb = rgb * color_process.synthetic_gain(
                st(subbed["LSYN"], blacks["LSYN"]), st(subbed["LSYN_REF"], blacks["LSYN_REF"]))[:, :, None]
        if args.saturation != 1.0:
            # Chroma scaled about each pixel's own luminance, so brightness is
            # untouched and only the colour difference from grey grows (or
            # shrinks). Applied before the star recolouring: the stars keep
            # the colour the RGB stacks gave them.
            rgb = np.clip(np.nan_to_num(rgb), 0, 1)
            y = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[:, :, None]
            rgb = y + (rgb - y) * float(args.saturation)
        if star_mask_arr is not None:
            rgb = color_process.rgb_stars(np.clip(np.nan_to_num(rgb), 0, 1), star_img, star_mask_arr)
        pth = color_process.save_rgb(np.clip(np.nan_to_num(rgb), 0, 1),
                                     out_dir / f"{args.dso}_{args.recipe}_{tag}.jpg",
                                     max_px=args.max_px)
        written.append(pth)

    emit(subbed_raw, "raw")
    if den:
        # Pinned to the raw set's balance and Ha ratio: measured again here,
        # wb=stars swung ngc7320's factors by a third and turned the denoised
        # composite blue (2026-10-09), which reads as the model's fault.
        subbed_den, _ = color_process._prepare(
            to_channels(den), opts["subtract_background"], opts["mesh"],
            opts["white_pct"], **shared_from_raw(raw_info))
        emit(subbed_den, "denoised")

    # The third class: the denoised set (raw when no model applies) with its
    # luminance (by default) through the learned deconvolver, on the same
    # pinned stretch. Off with --no-learned; skipped without a checkpoint.
    # A refusal or a failure here costs the learned products only.
    lea_tag = "learned"
    if args.learned:
        chans = to_channels(den if den else raw)
        planes = tuple(chans) if args.learned_planes.strip().lower() == "all" else \
            tuple(p.strip().upper() for p in args.learned_planes.split(",") if p.strip())
        lea, lea_tag = learned_channels(chans, args.learned_model, log, planes=planes,
                                        iters=args.deblur or color_process.DEBLUR_ITERS,
                                        no_l=args.no_l)
        if lea is not None:
            subbed_lea, _ = color_process._prepare(
                lea, opts["subtract_background"], opts["mesh"],
                opts["white_pct"], **shared_from_raw(raw_info))
            emit(subbed_lea, lea_tag)

    # Optional: the same set through the starless deconvolution, the third
    # composite from 2026-10-19 to 10-21, kept for comparison (--deblur N).
    # Not again when the learned step already fell back to it (no L).
    if args.deblur > 0 and lea_tag != "deblurred":
        deb = deblurred_channels(to_channels(den if den else raw), args.deblur, log,
                                 stars=args.deblur_stars)
        if deb is not None:
            subbed_deb, _ = color_process._prepare(
                deb, opts["subtract_background"], opts["mesh"],
                opts["white_pct"], **shared_from_raw(raw_info))
            emit(subbed_deb, "deblurred")

    log("")
    for pth in written:
        log(f"  wrote {pth.name}")
    log("")
    log(f"{len(written)} files in {out_dir}")

    if args.push:
        push_render(args, out_dir, written, meta, log)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stacks", "compose", "routine"))
    ap.add_argument("--dso", default="ngc5907")
    ap.add_argument("--recipe", default="LRGB",
                    choices=("LRGB", "HALRGB", "HOO", "SHO", "HSO"))
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
    ap.add_argument("--max-px", type=int, default=2400,
                    help="longest side of the composite JPEGs; 0 keeps full resolution")
    # 2.5 by the operator's choice on 2026-10-20, after NGC 1961 at 1, 2 and 3:
    # the HII regions and the red field stars want it, and the one cost (the
    # cyan halos on bright stars) is the G stack's broader PSF, not the boost.
    ap.add_argument("--saturation", type=float, default=2.5,
                    help="scale each composite's chroma about its luminance; "
                         "1 leaves the colour as composed, 2.5 (default) boosts it")
    ap.add_argument("--white-pct", type=float, default=99.95,
                    help="shared white point percentile; compose()'s 99.0 "
                         "saturates this field completely")
    ap.add_argument("--black-pct", type=float, default=40.0,
                    help="only used when --black-sigma <= 0")
    # Sky-anchored black, default on. A percentile black is relative to a
    # distribution that denoising changes: on the 2026-08-24 trunk render p65
    # landed 0.36 sigma above sky, so raw sky pixels were dithered above it and
    # denoised ones fell below, and the render looked as though the model had
    # eaten the nebulosity (it had not — flux retention measured 104-117%).
    ap.add_argument("--soft", type=float, default=None,
                    help="asinh softening; lower = harder stretch "
                         f"(compose default {0.025})")
    ap.add_argument("--stars", default="none", choices=("none", "rgb"),
                    help="rgb: also stack R, G and B onto the shared reference and "
                         "recolour the stars from them (process's stars=rgb); the "
                         "palette keeps its brightness, the nebula is untouched")
    ap.add_argument("--wb", default="none", choices=("none", "stars"),
                    help="white balance as the process command's wb=: `stars` "
                         "scales R and B so the field's median star is neutral. "
                         "LRGB/HALRGB only; the palettes are not colours")
    ap.add_argument("--scnr", type=float, default=None,
                    help="average-neutral green removal, 0..1, as the process "
                         "command's scnr= (compose default 0): on a gained SHO "
                         "it takes the green cast out of the faint field")
    ap.add_argument("--gain", default="",
                    help="palette gain as the process command's gain=: `hubble` "
                         "(every filter's nebula level equalised, the classic "
                         "independently-stretched look) or `Ha:0.4/O-III:1.5`; "
                         "measured on the raw set and pinned on the denoised and "
                         "learned composites. HOO/SHO/HSO only")
    ap.add_argument("--auto-stretch", action="store_true",
                    help="black points and softening from the data "
                         "(color_process.auto_stretch) instead of the fixed "
                         "--black-sigma/--black-pct/--soft; white stays --white-pct")
    ap.add_argument("--black-sigma", type=float, default=0.5,
                    help="black point at sky_median - N*sigma, measured from the "
                         "raw; 0 or less restores the --black-pct percentile")
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
    ap.add_argument("--deblur", type=int, default=0,
                    help="also write the starless-RL composite (\"deblurred\", the third "
                         "composite until 2026-10-21) with this many iterations; 0 (default) "
                         "= not written")
    ap.add_argument("--deblur-stars", default="imaged", choices=("imaged", "points"),
                    help="starless star mode: the denoised stars put back as imaged "
                         "(default) or points rendered at their flux")
    ap.add_argument("--learned-model", default=None,
                    help="checkpoint for the learned-deconvolution composite "
                         "(default nn/deblur_model.DEFAULT_MODEL; missing = skipped)")
    ap.add_argument("--no-l", choices=("starless", "synthetic"), default=None,
                    help="a set with no luminance (HOO/SHO): a synthetic luminance "
                         "through the network (stacking/deblur.NO_L_RULE, the default "
                         "since 2026-10-22 evening) or starless on every plane, written "
                         "as 'deblurred'")
    ap.add_argument("--learned-planes", default="L",
                    help="which planes the learned deconvolver sharpens, comma-separated "
                         "(default L: the luminance; the colour planes stay denoised). "
                         "'all' deconvolves every plane by its own kernel")
    ap.add_argument("--no-learned", dest="learned", action="store_false",
                    help="skip the learned composite")
    ap.set_defaults(learned=True)
    ap.add_argument("--push", action="store_true",
                    help="Pushover the composites when the render finishes")
    ap.add_argument("--no-push", dest="push", action="store_false")
    ap.set_defaults(push=True)
    ap.add_argument("--push-all", action="store_true",
                    help="push every composite class (raw, denoised, learned) "
                         "instead of the one best picture")
    args = ap.parse_args()
    if args.learned_model is None:
        from nn import deblur_model
        args.learned_model = str(deblur_model.DEFAULT_MODEL)
    import sep
    sep.set_extract_pixstack(5_000_000)

    if args.stage == "routine":
        return routine(args)
    if args.stage == "stacks":
        return build(args)
    return compose(args)


if __name__ == "__main__":
    sys.exit(main())
