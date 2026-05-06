#!/usr/bin/env python3
"""Denoise LIGHT frames for a DSO using trained Noise2Noise models.

Usage:
    python scripts/n2n_denoise.py <filter> <dso> [--compare]

Examples:
    python scripts/n2n_denoise.py Ha m31
    python scripts/n2n_denoise.py R "NGC 1499" --compare

Each frame's EXPTIME header is read to select the correct model
(e.g. n2n_R_90s.pt or n2n_R_300s.pt). Frames whose model is missing
are skipped with a warning.

Denoised frames are written to {dso_dir}/denoised/ with the same filenames
as the originals. Existing files in denoised/ are skipped (re-run is safe).

--compare  Save a side-by-side JPEG of the first denoised frame to
           {dso_dir}/denoised/compare_{filter}.jpg for quick inspection.
"""

import os
import socket
import sys
import time
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
from nn import denoiser


def _save_comparison(raw: np.ndarray, denoised: np.ndarray, out_path: Path, title: str) -> None:
    """Save a side-by-side ZScale JPEG of raw vs denoised."""
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


def _find_dso_dir(subs_dir: Path, dso_name: str) -> Path | None:
    target = dso_name.lower().replace(" ", "").replace("_", "")
    candidates = [
        d for d in subs_dir.iterdir()
        if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple matches — pick the one with the most recent LIGHT frame
    def _latest(d: Path) -> float:
        try:
            return max(
                f.stat().st_mtime for f in d.rglob("*.fits")
                if f.parent.name.upper() == "LIGHT"
            )
        except ValueError:
            return 0.0
    return max(candidates, key=_latest)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/n2n_denoise.py <filter> <dso>")
        print("  e.g. python scripts/n2n_denoise.py Ha m31")
        sys.exit(1)

    args = sys.argv[1:]
    compare = "--compare" in args
    args = [a for a in args if a != "--compare"]

    filter_name = args[0].strip()
    dso_name    = " ".join(args[1:]).strip()

    cfg = config.data()
    hostname = socket.gethostname()
    machine_cfg = cfg.get("machine", {}).get(hostname)
    if machine_cfg is None:
        print(f"Error: no 'machine.{hostname}' entry in config_public.py")
        sys.exit(1)

    subs_dir = Path(machine_cfg["subs_dir"])
    if not subs_dir.exists():
        print(f"Error: subs_dir does not exist: {subs_dir}")
        sys.exit(1)

    nn_cfg    = cfg.get("nn", {})
    tile_size = int(nn_cfg.get("tile_size", 512))
    overlap   = int(nn_cfg.get("tile_overlap", 64))

    # Locate DSO directory
    dso_dir = _find_dso_dir(subs_dir, dso_name)
    if dso_dir is None:
        print(f"Error: no directory found for DSO '{dso_name}' under {subs_dir}")
        sys.exit(1)
    print(f"DSO directory: {dso_dir}")

    from astropy.io import fits as _fits

    # Collect matching LIGHT frames in this DSO
    light_frames = sorted(
        (f for f in dso_dir.rglob("*.fits") if f.parent.name.upper() == "LIGHT"),
        key=lambda f: f.stat().st_mtime,
    )

    matching = []  # list of (path, exptime_s)
    for fp in light_frames:
        try:
            with _fits.open(fp) as hdul:
                hdr = hdul[0].header
                if str(hdr.get("FILTER", "")).strip() != filter_name:
                    continue
                et = round(float(hdr.get("EXPTIME", 0)))
                matching.append((fp, et))
        except Exception:
            pass

    if not matching:
        print(f"No LIGHT frames with FILTER={filter_name!r} found in {dso_dir}")
        sys.exit(0)

    print(f"Found {len(matching)} {filter_name} frames to denoise")

    device = denoiser.best_device()
    print(f"Inference device: {device}")

    # Cache loaded models by exptime so we don't reload for every frame
    model_cache: dict[int, object] = {}
    missing_models: set[int] = set()

    out_dir = dso_dir / "denoised"
    out_dir.mkdir(exist_ok=True)

    t_total = time.time()
    done = skipped = 0
    for i, (fp, et) in enumerate(matching, 1):
        out_path = out_dir / fp.name
        if out_path.exists():
            print(f"  [{i}/{len(matching)}] skip (exists): {out_path.name}")
            skipped += 1
            continue

        # Load model for this exptime if not already cached
        if et not in model_cache and et not in missing_models:
            model_path = denoiser.get_model_path(filter_name, et)
            if not model_path.exists():
                print(f"  [{i}/{len(matching)}] SKIP — no model for {filter_name} {et}s "
                      f"(train with: python scripts/n2n_train.py {filter_name} {et})")
                missing_models.add(et)
            else:
                print(f"  Loading model {model_path.name}…")
                model_cache[et] = denoiser.load_model(model_path)

        if et in missing_models:
            continue

        model = model_cache[et]
        t0 = time.time()
        with _fits.open(fp) as hdul:
            header = hdul[0].header.copy()
            raw = np.squeeze(hdul[0].data).astype(np.float32)

        denoised_arr = denoiser.denoise_frame(raw, model, device=device, tile_size=tile_size, overlap=overlap)

        hdu = _fits.PrimaryHDU(data=denoised_arr, header=header)
        hdu.header["HISTORY"] = "Noise2Noise denoised"
        _fits.HDUList([hdu]).writeto(out_path, overwrite=True)

        elapsed = time.time() - t0
        rel = fp.relative_to(dso_dir)
        print(f"  [{i}/{len(matching)}] {rel} → denoised/{fp.name}  ({et}s, {elapsed:.1f}s)")
        done += 1

        if compare and done == 1:
            cmp_path = out_dir / f"compare_{filter_name}.jpg"
            _save_comparison(raw, denoised_arr, cmp_path, f"{dso_dir.name}  {filter_name} {et}s")

    total = time.time() - t_total
    print(f"\nDone — {done} denoised, {skipped} skipped  ({total:.0f}s) → {out_dir}")


if __name__ == "__main__":
    main()
