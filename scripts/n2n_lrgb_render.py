#!/usr/bin/env python3
"""Render the held-out DSO as an LRGB colour image, raw vs denoised.

    python scripts/n2n_lrgb_render.py stacks                    # build channels
    python scripts/n2n_lrgb_render.py compose --model <path>    # denoise + compose

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
    return OUT / f"meta{suffix(args)}.json"


def log(m: str = "") -> None:
    print(m, flush=True)


def build(args) -> int:
    from nn import stacks
    from stacking import stacker

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    subs = Path(machine["subs_dir"])
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
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
        log(f"[{filt}] stacking {len(paths)} frames onto the {lum} reference")
        t0 = time.time()
        m: dict = {}
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

    meta = json.loads(meta_path(args).read_text())
    filters = [f for f in ("L", "R", "G", "B") if f in meta["channels"]]

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
            rgb, OUT / f"{meta['dso']}_LRGB_{tag}{suffix(args)}.jpg",
            max_px=args.max_px)
        log(f"wrote {p}")
        del rgb
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stacks", "compose"))
    ap.add_argument("--dso", default="ngc5907")
    ap.add_argument("--filters", default="L,R,G,B")
    ap.add_argument("--lum", default="L", help="filter supplying the shared reference")
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--depth", type=int, default=0,
                    help="cap frames per channel; 0 uses all (display product)")
    ap.add_argument("--model", help="checkpoint for the compose stage")
    ap.add_argument("--max-px", type=int, default=2400)
    ap.add_argument("--white-pct", type=float, default=99.95,
                    help="shared white point percentile; compose()'s 99.0 "
                         "saturates this field completely")
    ap.add_argument("--black-pct", type=float, default=40.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import sep
    sep.set_extract_pixstack(5_000_000)

    if args.stage == "stacks":
        return build(args)
    if not args.model:
        log("compose needs --model")
        return 1
    return compose(args)


if __name__ == "__main__":
    sys.exit(main())
