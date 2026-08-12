#!/usr/bin/env python3
"""Denoise all LIGHT frames for a given filter and exposure time.

Usage:
    python scripts/n2n_denoise.py <filter> <seconds> [--compare] [--no-register] [--force]

Examples:
    python scripts/n2n_denoise.py L 300
    python scripts/n2n_denoise.py Ha 300 --compare

Walks every DSO directory under subs_dir and denoises all LIGHT frames
whose FITS FILTER header matches <filter> and EXPTIME rounds to <seconds>.
The model local/models/n2n_{filter}_{seconds}s.pt must already exist
(train with: python scripts/n2n_train.py <filter> <seconds>).

After denoising, each frame is registered to the first frame of its own DSO
using SEP star centroid matching (same method as training prep).  This
corrects dithering within a target; cross-target pointing is left to the
stacker, which registers properly rather than by a capped translation fit.

Flags:
  --no-register  Skip registration; write denoised frames in their original
                 pixel coordinates.
  --compare      Save a side-by-side JPEG of the first denoised frame per DSO.
  --force        Overwrite existing denoised output files.
"""

import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from configs import config
from nn import denoiser, registration as _reg


def _save_comparison(raw: np.ndarray, denoised: np.ndarray, out_path: Path, title: str) -> None:
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(raw)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0d0d1a")
    for ax, data, label in zip(axes, [raw, denoised], ["Raw", "Denoised (N2N)"]):
        ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(label, color="white", fontsize=14)
        ax.axis("off")
        ax.set_facecolor("#0d0d1a")
    fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Comparison saved → {out_path}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/n2n_denoise.py <filter> <seconds> [--compare] [--no-register] [--force]")
        sys.exit(1)

    args = sys.argv[1:]
    compare     = "--compare"     in args
    do_register = "--no-register" not in args
    force       = "--force"       in args
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    filter_name = args[0].strip()
    try:
        exptime_s = int(args[1])
    except (IndexError, ValueError):
        print(f"Error: seconds must be an integer, got '{args[1] if len(args) > 1 else ''}'")
        sys.exit(1)

    cfg = config.data()
    hostname = socket.gethostname()
    machine_cfg = cfg.get("machine", {}).get(hostname)
    if machine_cfg is None:
        print(f"Error: no 'machine.{hostname}' entry in config_public.py")
        sys.exit(1)

    if "subs_dir" not in machine_cfg:
        print(f"Error: machine.{hostname} has no 'subs_dir' in config_public.py")
        print("  N2N reads frames from subs_dir; only hosts that define it can run this.")
        sys.exit(1)
    subs_dir = Path(machine_cfg["subs_dir"])
    if not subs_dir.exists():
        print(f"Error: subs_dir does not exist: {subs_dir}")
        sys.exit(1)

    nn_cfg    = cfg.get("nn", {})
    tile_size = int(nn_cfg.get("tile_size", 512))
    overlap   = int(nn_cfg.get("tile_overlap", 64))

    model_path = denoiser.get_model_path(filter_name, exptime_s)
    if not model_path.exists():
        print(f"Error: no model at {model_path}")
        print(f"  Train first: python scripts/n2n_train.py {filter_name} {exptime_s}")
        sys.exit(1)
    print(f"Loading model {model_path.name}…")
    model = denoiser.load_model(model_path)
    device = denoiser.best_device()
    print(f"Inference device: {device}")

    from astropy.io import fits as _fits

    # Collect all matching LIGHT frames
    dso_dirs = sorted(d for d in subs_dir.iterdir() if d.is_dir())
    all_frames: list[tuple[Path, Path]] = []  # (fits_path, dso_dir)
    for dso_dir in dso_dirs:
        for fp in sorted(dso_dir.rglob("*.fits")):
            if fp.parent.name.upper() != "LIGHT":
                continue
            try:
                with _fits.open(fp) as hdul:
                    hdr = hdul[0].header
                    if str(hdr.get("FILTER", "")).strip() != filter_name:
                        continue
                    if round(float(hdr.get("EXPTIME", 0))) != exptime_s:
                        continue
                    all_frames.append((fp, dso_dir))
            except Exception:
                pass

    if not all_frames:
        print(f"No {filter_name} {exptime_s}s LIGHT frames found under {subs_dir}")
        sys.exit(0)

    n_dsos = len(set(d.name for _, d in all_frames))
    print(f"Found {len(all_frames)} {filter_name} {exptime_s}s frames across {n_dsos} DSO(s)")

    # One reference per DSO, from that DSO's own first frame.
    #
    # This used to take a single global reference — the first frame of the first
    # DSO — and register every frame of every target to it, on the reasoning that
    # it would also absorb cross-session pointing. It cannot: _match_translation
    # is a translation-only fit capped at max_dist=200 px, and two targets are
    # degrees apart with star fields that have nothing in common. At ~50 stars
    # over this sensor the mean separation is ~1100 px, so frames of every DSO
    # but the first either found fewer than three matches and were silently left
    # unregistered, or found three spurious ones and were shifted by a
    # meaningless median — and nothing in the output told the two apart.
    # Training was always per-group (nn/registration.py); inference now agrees.
    ref_stars_by_dso: dict[str, np.ndarray] = {}
    ref_fp_by_dso: dict[str, Path] = {}
    if do_register:
        for fp, dso_dir in all_frames:
            if dso_dir.name in ref_fp_by_dso:
                continue
            ref_fp_by_dso[dso_dir.name] = fp
            with _fits.open(fp) as hdul:
                ref_raw = np.squeeze(hdul[0].data).astype(np.float32)
            stars = _reg._find_stars(ref_raw)
            if len(stars) < 3:
                print(f"Warning: too few stars in the reference for {dso_dir.name} "
                      f"({fp.name}) — that DSO will not be registered")
                continue
            ref_stars_by_dso[dso_dir.name] = stars
            print(f"  reference [{dso_dir.name}]: {len(stars)} stars from {fp.name}")

    if do_register and ref_stars_by_dso:
        print("Registration: ON  (per-DSO reference, max_dist=200 px)")
    else:
        print("Registration: OFF")

    t_total = time.time()
    done = skipped = 0
    compare_saved: set[str] = set()

    for fp, dso_dir in all_frames:
        out_dir = subs_dir / "denoised" / f"{filter_name}_{exptime_s}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fp.name

        if out_path.exists() and not force:
            print(f"  skip (exists): {dso_dir.name}/{fp.name}")
            skipped += 1
            continue

        t0 = time.time()
        with _fits.open(fp) as hdul:
            header = hdul[0].header.copy()
            raw = np.squeeze(hdul[0].data).astype(np.float32)

        denoised_arr = denoiser.denoise_frame(raw, model, device=device,
                                              tile_size=tile_size, overlap=overlap)

        # Apply this DSO's registration shift (computed from raw, applied to the
        # denoised array). Using raw for centroid detection avoids confusion from
        # N2N star halos.
        shift_applied = False
        ref_stars = ref_stars_by_dso.get(dso_dir.name)
        if do_register and ref_stars is not None and fp != ref_fp_by_dso.get(dso_dir.name):
            src_stars = _reg._find_stars(raw)
            if len(src_stars) >= 3:
                shift = _reg._match_translation(ref_stars, src_stars)
                if shift is not None:
                    from scipy.ndimage import shift as nd_shift
                    fill = float(np.median(denoised_arr))
                    denoised_arr = nd_shift(
                        denoised_arr, shift, mode="constant", cval=fill
                    ).astype(denoised_arr.dtype)
                    shift_applied = True

        hdu = _fits.PrimaryHDU(data=denoised_arr, header=header)
        hdu.header["HISTORY"] = "Noise2Noise denoised"
        if shift_applied:
            hdu.header["HISTORY"] = "Registered (SEP centroid, global)"
        _fits.HDUList([hdu]).writeto(out_path, overwrite=True)

        elapsed = time.time() - t0
        rel = fp.relative_to(dso_dir)
        reg_tag = " [reg]" if shift_applied else ""
        print(f"  [{done + skipped + 1}/{len(all_frames)}] {dso_dir.name}/{rel} → {fp.name}  ({elapsed:.1f}s){reg_tag}")
        done += 1

        if compare and dso_dir.name not in compare_saved:
            cmp_path = out_dir / f"compare_{dso_dir.name}.jpg"
            _save_comparison(raw, denoised_arr, cmp_path,
                             f"{dso_dir.name}  {filter_name} {exptime_s}s")
            compare_saved.add(dso_dir.name)

    total = time.time() - t_total
    print(f"\nDone — {done} denoised, {skipped} skipped  ({total:.0f}s)")


if __name__ == "__main__":
    main()
