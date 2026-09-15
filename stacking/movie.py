"""Stacking-progression movie: one field at n increasing depths, looping.

    python stacking/movie.py ngc7380 HSO 8
    python stacking/movie.py m33 LRGB 4 --width 1280 --hold 3

`movie ngc7380 hso 8` on a target with 80 subs gives eight frames, the first
from 10 subs and the last from all 80, each held about five seconds with a
short crossfade to the next, and a footer that counts the subs in the frame on
screen. The movie is meant to show what stacking does to noise, so everything
that is not the frame count is held fixed:

- **One registration.** Every filter is registered once, onto one shared
  reference chosen the way process_dso chooses it, and the registered frames
  are kept in memory at movie resolution. Each depth is then a combine over a
  prefix of that cube — the same sigma-clipped, FWHM-weighted method the
  stacker uses — rather than a fresh stack, which would cost n/2 times the
  full stack for the same pixels.
- **Chronological prefixes.** Frame k uses the first k/n of each filter's subs
  in the order they were taken, so the movie is the night(s) as they
  accumulated, not a random draw.
- **One stretch.** Black and white points are taken from the full-depth stack
  and applied unchanged to every shallower one. A stretch measured per frame
  would put the darkest 65% of a noisier frame at a different ADU and the
  frames would not be comparable. The defaults are the nightly render's, not
  compose()'s: black at sky - 0.5 sigma per channel and white at p99.95,
  because compose()'s p99 white saturates a galaxy field completely (the
  first M33 test came out as a white blob). `black=` switches to a percentile.
- **Levelled sky.** The frames are levelled to a common sky before combining
  and the level restored after, as stack() does; without it the sigma clip
  turns the night-to-night sky spread into noise, which is the opposite of what
  the movie is for.
- **An inset at 1:1 pixels.** Reducing 9576 pixels to 1920 averages 16 of them
  into each output pixel, a 4x noise reduction before stacking does anything;
  measured on the first NGC 7380 movie, the sky noise at movie scale was at
  the 8-bit quantisation floor in the 16-sub frame and the 125-sub frame
  alike, so the movie could not show what it was for. So one patch of the
  field is carried through registration at native resolution and shown
  magnified in the corner, where single-sub noise is tens of grey levels and
  visibly shrinks with depth. Its position is outlined on the field.

Output is H.264 MP4 (yuv420p, faststart) so it plays in a <video> tag in any
browser and can be dropped into a page as-is, plus a poster JPEG of the last
frame. Encoding goes through the ffmpeg that imageio-ffmpeg bundles, so no
system ffmpeg is needed.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_logger = logging.getLogger(__name__)

HOLD_SECONDS = 5.0      # each depth stays this long
FADE_SECONDS = 0.6      # crossfade to the next depth
FPS = 12                # holds are repeated frames; x264 makes them nearly free
WIDTH = 1920
FOOTER_FRACTION = 0.045 # footer height as a fraction of the picture width
WHITE_PCT = 99.95       # compose()'s 99.0 clips everything above ~12 ADU on a galaxy
BLACK_SIGMA = 0.5       # black at sky - this many sigma, per channel
INSET_PX = 480          # native pixels on a side of the 1:1 patch
INSET_ZOOM = 2          # magnification of the patch on the movie frame
ACCENT = (43, 87, 217)


def increments(counts: dict[str, int], steps: int) -> list[dict[str, int]]:
    """Per-filter sub counts at each of *steps* evenly spaced depths.

    The last entry is every sub. A filter with fewer subs than steps repeats
    counts at the shallow end rather than starting from zero.
    """
    return [{f: max(1, int(round(n * k / steps))) for f, n in counts.items()}
            for k in range(1, steps + 1)]


def find_dso_dir(image_dir: Path, name: str) -> Optional[Path]:
    """Directory under image_dir whose name contains *name*, newest lights winning a tie."""
    target = name.lower().replace(" ", "").replace("_", "")
    candidates = [d for d in image_dir.iterdir() if d.is_dir()
                  and target in d.name.lower().replace(" ", "").replace("_", "")]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def _latest(d: Path) -> float:
        try:
            return max(f.stat().st_mtime for f in d.rglob("*.fits"))
        except ValueError:
            return 0.0
    return max(candidates, key=_latest)


def _with_inset(img, patch, box, zoom: int):
    """Paste *patch* magnified *zoom* times (nearest, so pixels stay pixels) in
    the lower-right corner with an accent border, and outline *box*, its
    source, on the field."""
    from PIL import Image, ImageDraw

    w, h = img.size
    pw, ph = patch.size
    zoom = max(1, int(zoom))
    # never let the inset cover more than 40% of the frame's short side
    while zoom > 1 and max(pw, ph) * zoom > 0.4 * min(w, h):
        zoom -= 1
    big = patch.resize((pw * zoom, ph * zoom), Image.NEAREST)
    bw = 3
    pad = int(0.012 * w)
    x = w - big.width - pad
    y = h - big.height - pad
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle((x - bw, y - bw, x + big.width + bw - 1, y + big.height + bw - 1), fill=ACCENT)
    out.paste(big, (x, y))
    if box:
        d.rectangle(box, outline=ACCENT, width=2)
        # a leader from the box to the inset, so the eye connects them
        d.line((box[2], box[3], x, y + big.height // 2), fill=ACCENT, width=1)
    return out


def pick_inset(frame: np.ndarray, size: int, margin: int) -> tuple[int, int]:
    """Centre of the patch the inset should show, from one calibrated sub.

    The first M33 movie put the inset at the field centre, which is the
    galaxy's core: bright enough that the noise the inset exists to show was
    invisible there. Noise is plainest on faint extended signal, so tile the
    frame in *size* blocks, rank the blocks by median, and take the one at
    the 75th percentile: above sky, below the bright core or shell. One sub
    is noisy per pixel but a 480x480 block median is not.
    """
    H, W = frame.shape
    ys = list(range(margin, H - margin - size + 1, size))
    xs = list(range(margin, W - margin - size + 1, size))
    if not ys or not xs:
        return W // 2, H // 2
    meds = []
    for y in ys:
        for x in xs:
            meds.append((float(np.nanmedian(frame[y:y + size, x:x + size])), x, y))
    meds.sort()
    _, x, y = meds[int(0.75 * (len(meds) - 1))]
    return x + size // 2, y + size // 2


def _footer(img, dso: str, recipe: str, shown: dict[str, int], total: dict[str, int],
            order: list[str], inset_px: int = 0):
    """Return *img* with a footer: sub counts on the left, per filter on the
    right, and a progress bar along the bottom edge."""
    from PIL import Image, ImageDraw

    from stacking.color_process import _label_font

    w, h = img.size
    fh = max(40, int(w * FOOTER_FRACTION))
    out = Image.new("RGB", (w, h + fh), (13, 15, 20))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    font = _label_font(max(12, int(fh * 0.42)))
    small = _label_font(max(10, int(fh * 0.34)))
    n_shown, n_total = sum(shown.values()), sum(total.values())
    left = f"{dso}  ·  {recipe}     {n_shown} of {n_total} subs"
    right = "   ".join(f"{f} {shown[f]}/{total[f]}" for f in order)
    if inset_px:
        right = f"inset {inset_px} px at 1:1     " + right
    pad = int(fh * 0.3)
    ty = h + (fh - font.size) // 2 - int(fh * 0.06)
    d.text((pad, ty), left, fill=(235, 238, 245), font=font)
    rw = d.textlength(right, font=small)
    d.text((w - pad - rw, h + (fh - small.size) // 2 - int(fh * 0.06)), right,
           fill=(170, 178, 194), font=small)
    bar = max(3, fh // 12)
    d.rectangle((0, h + fh - bar, w, h + fh), fill=(32, 36, 48))
    d.rectangle((0, h + fh - bar, int(w * n_shown / max(n_total, 1)), h + fh), fill=ACCENT)
    return out


def _encode_mp4(frames: list, durations: list[float], path: Path, fps: int = FPS) -> None:
    """Pipe RGB frames to ffmpeg; *durations* are seconds each frame stays."""
    import imageio_ffmpeg

    w, h = frames[0].size
    w -= w % 2
    h -= h % 2
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for img, secs in zip(frames, durations):
            raw = np.asarray(img.convert("RGB"))[:h, :w].tobytes()
            for _ in range(max(1, int(round(secs * fps)))):
                proc.stdin.write(raw)
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}): {err.strip()[-500:]}")


def make_stacking_movie(
    dso_dir: Path,
    recipe: str,
    steps: int,
    out_path: Path,
    use_flats: bool = True,
    width: int = WIDTH,
    hold_s: float = HOLD_SECONDS,
    fade_s: float = FADE_SECONDS,
    inset: Optional[object] = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    **compose_kw,
) -> tuple[Path, dict]:
    """Build the movie; returns (mp4 path, info). A poster JPEG lands beside it.

    *inset*: True for a INSET_PX patch placed by pick_inset(), False/None for
    none, an int for a patch that size placed the same way, or (x, y, size)
    in reference pixels to place it yourself.
    """
    from PIL import Image

    from fits_processing.frame_cache import load_precomputed_fwhm_stars
    from stacking import color_process, stacker
    from stacking.color_process import RECIPES, resolve_filter

    def say(m: str) -> None:
        if progress_cb:
            progress_cb(m)

    def ckpt() -> None:
        stacker._ckpt(cancel_cb)

    recipe = recipe.upper()
    steps = int(steps)
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe '{recipe}'. Choose one of {', '.join(sorted(RECIPES))}.")

    lights = sorted((f for f in dso_dir.rglob("*.fits") if f.parent.name.upper() == "LIGHT"),
                    key=lambda f: f.stat().st_mtime)
    if not lights:
        raise ValueError(f"No LIGHT frames under {dso_dir}")
    by_filter = stacker.group_by_filter(lights)

    mapping = RECIPES[recipe]
    resolved: dict[str, str] = {}
    for chan, wanted in mapping.items():
        found = resolve_filter(wanted, list(by_filter))
        if found:
            resolved[chan] = found
    if not {"R", "G", "B"} <= set(resolved):
        missing = [f"{c}={mapping[c]}" for c in mapping if c not in resolved]
        raise ValueError(f"{recipe} needs {', '.join(mapping[c] for c in ('R', 'G', 'B'))}; "
                         f"this target has {', '.join(sorted(by_filter))}. Missing: {', '.join(missing)}")
    filters: list[str] = []
    for chan in ("L", "R", "G", "B"):
        f = resolved.get(chan)
        if f and f not in filters:
            filters.append(f)

    # Shared reference, chosen as process_dso chooses it.
    ref_filter = max(filters, key=lambda f: len(by_filter[f]))
    ref_paths = by_filter[ref_filter]
    arcsec = stacker._get_arcsec_per_pixel()
    ref_pre = load_precomputed_fwhm_stars(dso_dir, ref_paths, arcsec)
    ref_idx = stacker._reference_index_by_fwhm(
        [ref_pre.get(p, (0.0, 0))[0] for p in ref_paths]) or 0
    ref_path = ref_paths[ref_idx]
    say(f"shared reference: {ref_path.name} ({ref_filter})")
    ref_cal = stacker.calibration_from_config(ref_filter if use_flats else None)
    reference = stacker._load_calibrated(ref_path, ref_cal)
    ref_shape = reference.shape
    det_target = stacker._reference_control_points(stacker._despike(reference))
    if det_target is None:
        raise ValueError("Could not extract control points from the reference frame")

    # Registration downscales by an integer chosen so the short side is at least
    # this; the frames are resized to *width* at the end.
    downscale_to = max(256, int(width * 2 / 3))
    reg_scale = max(1, min(ref_shape) // downscale_to)

    crop = None
    if inset:
        H, W = ref_shape
        margin = int(color_process.EDGE_CROP * min(H, W)) + 8
        if isinstance(inset, (tuple, list)) and len(inset) == 3:
            cx, cy, size = int(inset[0]), int(inset[1]), int(inset[2])
            size = max(64, min(size, min(H, W) // 2))
        else:
            size = max(64, min(INSET_PX if inset is True else int(inset), min(H, W) // 2))
            cx, cy = pick_inset(reference, size, margin)
        cx = min(max(cx, margin + size // 2), W - margin - size // 2)
        cy = min(max(cy, margin + size // 2), H - margin - size // 2)
        crop = (cy - size // 2, cy - size // 2 + size, cx - size // 2, cx - size // 2 + size)
        say(f"inset: {size} px at 1:1, centred ({cx}, {cy})")
    reference = None

    cubes: dict[str, tuple[np.ndarray, np.ndarray, float, list[Path]]] = {}
    inset_cubes: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for filt in filters:
        ckpt()
        paths = by_filter[filt]
        say(f"{filt}: registering {len(paths)} frames…")
        cal = stacker.calibration_from_config(filt if use_flats else None)
        pre = load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
        frames, accepted, fwhm = stacker._prepare_for_convergence(
            paths, register=True, downscale_to=downscale_to,
            progress_cb=lambda m, _f=filt: say(f"{_f}: {m}"), cancel_cb=cancel_cb,
            precomputed_fwhm_stars=pre, calibration=cal,
            shared_reference=(det_target, ref_shape), crop=crop)
        if not frames:
            raise ValueError(f"{filt}: no frames survived registration")
        if crop is not None:
            patches = np.stack([f[1] for f in frames], axis=0).astype(np.float32)
            frames = [f[0] for f in frames]
        else:
            patches = None
        arr = np.stack(frames, axis=0).astype(np.float32)
        frames = None
        w = stacker._fwhm_weights(fwhm, accepted)
        levels = np.array([float(np.nanmedian(f)) for f in arr], dtype=np.float64)
        arr -= levels[:, None, None].astype(np.float32)
        if patches is not None:
            patches -= levels[:, None, None].astype(np.float32)
        cubes[filt] = (arr, w, float(levels.mean()), accepted)
        if patches is not None:
            inset_cubes[filt] = patches
        say(f"{filt}: {len(accepted)} of {len(paths)} frames aligned "
            f"({time.perf_counter() - t0:.0f}s)")

    shapes = {c[0].shape[1:] for c in cubes.values()}
    if len(shapes) != 1:
        raise ValueError(f"channels are on different grids: {shapes}")
    h, w = shapes.pop()
    m = int(color_process.EDGE_CROP * min(h, w))

    totals = {f: cubes[f][0].shape[0] for f in filters}
    plan = increments(totals, steps)

    # A black percentile given explicitly wins; otherwise anchor to the sky.
    black_sigma = 0.0 if "black_pct" in compose_kw else compose_kw.pop("black_sigma", BLACK_SIGMA)
    compose_kw.setdefault("white_pct", WHITE_PCT)
    opts = color_process.effective_options(**compose_kw)

    def combine(filt: str, n: int, patch: bool = False) -> np.ndarray:
        arr, wts, level, _ = cubes[filt]
        sub = (inset_cubes[filt] if patch else arr)[:n]
        sw = wts[:n] / max(float(wts[:n].sum()), 1e-12)
        img = stacker._combine_tile(sub, stacker.StackMethod.SIGMA_CLIP_FWHM, sw, sigma=3.0)
        img = img + level
        bad = ~np.isfinite(img)
        if bad.any():
            img[bad] = float(np.nanmedian(img))
        if patch:
            return img
        return img[m:-m, m:-m] if m > 0 else img

    # The stretch comes from the full stack and is applied to every depth.
    say(f"stacking {steps} depths…")
    stacks_at: dict[tuple[str, int], np.ndarray] = {}

    def channels_for(counts: dict[str, int], patch: bool = False) -> dict[str, np.ndarray]:
        out = {}
        for chan in ("R", "G", "B", "L"):
            f = resolved.get(chan)
            if not f:
                continue
            key = (f, counts[f], patch)
            if key not in stacks_at:
                stacks_at[key] = combine(f, counts[f], patch)
            out[chan] = stacks_at[key]
        return out

    ckpt()
    full = channels_for(plan[-1])
    subbed_full, white = color_process._prepare(
        full, opts["subtract_background"], opts["mesh"], opts["white_pct"])
    blacks = color_process.sky_anchored_blacks(subbed_full, black_sigma, opts["black_pct"],
                                               log=lambda m: _logger.info(m.strip()))
    _logger.info("Movie stretch: white %.2f ADU, blacks %s", white,
                 {c: round(b, 3) for c, b in blacks.items()})

    # The inset cannot go through the gradient mesh — on a 480 px patch the
    # mesh would eat the nebulosity — so it gets the offset the full-frame
    # background model has over its footprint, read off the downscaled frame.
    inset_offset: dict[str, float] = {}
    inset_box_ds = None
    if crop is not None:
        y0, y1, x0, x1 = crop
        ys, ye = (y0 // reg_scale) - m, (y1 // reg_scale) - m
        xs, xe = (x0 // reg_scale) - m, (x1 // reg_scale) - m
        inset_box_ds = (xs, ys, xe, ye)
        for c in full:
            region_raw = full[c][ys:ye, xs:xe]
            region_sub = subbed_full[c][ys:ye, xs:xe]
            inset_offset[c] = float(np.nanmedian(region_raw - region_sub))

    dso_label = dso_dir.name
    stills: list = []
    for k, counts in enumerate(plan, 1):
        ckpt()
        chans = channels_for(counts)
        subbed, _ = color_process._prepare(
            chans, opts["subtract_background"], opts["mesh"], opts["white_pct"])
        rgb = color_process.compose_prepared(subbed, blacks, white,
                                             softening=opts["softening"], scnr=opts["scnr"])
        arr8 = stacker.sky_parity((rgb * 255.0 + 0.5).astype(np.uint8))
        img = Image.fromarray(arr8, mode="RGB")
        box_on_movie = None
        if inset_box_ds is not None:
            # sky_parity keeps the array as stored, so the box maps straight across
            sx = width / img.width
            xs, ys, xe, ye = inset_box_ds
            box_on_movie = (int(xs * sx), int(ys * sx), int(xe * sx), int(ye * sx))
        if img.width != width:
            img = img.resize((width, max(2, int(round(img.height * width / img.width)))),
                             Image.LANCZOS)
        if crop is not None:
            pch = channels_for(counts, patch=True)
            psub = {c: pch[c] - inset_offset[c] for c in pch}
            prgb = color_process.compose_prepared(psub, blacks, white,
                                                  softening=opts["softening"], scnr=opts["scnr"])
            pimg = Image.fromarray(stacker.sky_parity((prgb * 255.0 + 0.5).astype(np.uint8)), mode="RGB")
            img = _with_inset(img, pimg, box_on_movie, INSET_ZOOM)
        stills.append(_footer(img, dso_label, recipe, counts, totals, filters,
                              inset_px=(crop[1] - crop[0]) if crop else 0))
        say(f"frame {k}/{steps}: " + ", ".join(f"{f} {counts[f]}" for f in filters))
        # Every prefix except the full one is used exactly once.
        for f in filters:
            if counts[f] != totals[f]:
                stacks_at.pop((f, counts[f], False), None)
                stacks_at.pop((f, counts[f], True), None)

    # Sequence: hold, crossfade, hold, … and a crossfade from the last back to
    # the first so the loop has no jump.
    n_fade = max(2, int(round(fade_s * FPS)))
    frames: list = []
    durations: list[float] = []
    for i, still in enumerate(stills):
        frames.append(still)
        durations.append(hold_s)
        nxt = stills[(i + 1) % len(stills)]
        for j in range(1, n_fade):
            t = j / n_fade
            frames.append(Image.blend(still, nxt, t))
            durations.append(1.0 / FPS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    say("encoding MP4…")
    _encode_mp4(frames, durations, out_path)
    poster = out_path.with_name(out_path.stem + "_poster.jpg")
    stills[-1].save(poster, quality=90, optimize=True)

    info = {
        "recipe": recipe, "steps": steps, "flats": use_flats,
        "channels": {c: resolved[c] for c in resolved},
        "frames": totals, "plan": plan, "reference": ref_path.name,
        "size": stills[0].size, "seconds": round(len(stills) * (hold_s + fade_s), 1),
        "stretch": {"white": white, "blacks": blacks, "softening": opts["softening"]},
        "poster": poster, "bytes": out_path.stat().st_size, "inset": crop,
        "elapsed": round(time.perf_counter() - t0, 1),
    }
    return out_path, info


def main() -> int:
    import argparse

    from configs import config
    from stacking import stacker

    ap = argparse.ArgumentParser(description="Stacking-progression movie")
    ap.add_argument("dso")
    ap.add_argument("recipe")
    ap.add_argument("steps", type=int)
    ap.add_argument("--noflat", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--hold", type=float, default=HOLD_SECONDS)
    ap.add_argument("--out", default="")
    ap.add_argument("--inset", default="on",
                    help="off, a size in px, or x,y,size in reference pixels")
    args = ap.parse_args()
    inset: object = True
    if args.inset.lower() in ("off", "no", "none", "0"):
        inset = False
    elif "," in args.inset:
        inset = tuple(int(v) for v in args.inset.split(","))
    elif args.inset.lower() != "on":
        inset = int(args.inset)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    image_dir = Path(config.data()["nina"]["image_dir"])
    dso_dir = find_dso_dir(image_dir, args.dso)
    if dso_dir is None:
        print(f"no image directory for '{args.dso}' under {image_dir}")
        return 1
    out = Path(args.out) if args.out else (
        stacker.results_dir(dso_dir.name) / f"movie_{dso_dir.name}_{args.recipe.upper()}_{args.steps}.mp4")
    path, info = make_stacking_movie(
        dso_dir, args.recipe, args.steps, out, use_flats=not args.noflat,
        width=args.width, hold_s=args.hold, inset=inset,
        progress_cb=lambda m: print(m, flush=True))
    print(f"wrote {path} ({info['bytes'] / 1e6:.1f} MB, {info['seconds']} s, "
          f"{info['size'][0]}x{info['size'][1]}) poster {info['poster'].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
