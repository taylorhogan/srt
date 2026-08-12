#!/usr/bin/env python3
"""Does the denoiser buy frames, or does it launder noise into smoothness?

This is the test that has to pass before anything denoised goes near the
science pipeline. It is deliberately not a visual comparison: a denoiser that
makes an image prettier and a denoiser that preserves photometry are different
things, and only the second one is allowed near a transit depth or a
colour-magnitude diagram.

Two measurements, because they fail in different ways.

1. CONVERGENCE. stacking.stacker.convergence_curve measures RMSE against the
   all-frames stack as the frame count grows, and fits_processing.convergence.
   decay_ratio says how far the tail sits above what *independent* noise
   predicts. ~1.0 is textbook 1/sqrt(N); well above 1 means a correlated
   component that more frames will never remove.

   That is exactly the signature of a biased denoiser. Every frame goes through
   the same network with the same learned prior, so whatever the network gets
   wrong it gets wrong the same way on all of them. Independent noise averages
   down; a shared bias survives stacking untouched. So:

       denoised decay_ratio ~ raw decay_ratio, lower RMSE  -> genuinely bought frames
       denoised decay_ratio >> raw decay_ratio             -> laundering noise
       curve flattens to a floor raw does not have         -> laundering noise

2. PHOTOMETRY. Aperture flux on the same stars, raw stack vs denoised stack,
   binned by brightness. This catches the failure the convergence curve cannot
   see, and it is a specific, predictable one: the training loss is L1, whose
   optimum is the conditional *median* of the posterior. For a source near the
   detection limit the posterior mass sits at background, so the median sits at
   background too -- an L1-trained denoiser is structurally inclined to erase
   faint sources while leaving bright ones alone. That shows up as a flux ratio
   that falls below 1.0 in the faint bins and nowhere else, which is why the
   readout is binned rather than a single number.

Run it on a DSO the model was NOT trained on. n2n_train holds out whole targets
(see nn/trainer.py:_holdout_frames) and prints which ones -- use one of those.
Measuring on a training target answers a question nobody asked.

Usage:
    python scripts/n2n_evaluate.py <dso> <filter> <seconds> [--out DIR]

Example:
    python scripts/n2n_evaluate.py m13 R 300
"""
import os
import socket
import sys
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs import config
from fits_processing.convergence import decay_ratio

# Bin edges in instrumental magnitudes relative to the brightest matched star,
# so the readout does not depend on a photometric zero point this observatory
# has no reason to carry. 0 = brightest.
MAG_BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 99.0)]
MAG_LABELS = ["bright", "mid", "faint", "very faint"]


def _light_frames(root: Path, filter_name: str, exptime_s: int) -> list[Path]:
    """LIGHT frames under *root* matching filter and exposure."""
    from astropy.io import fits
    out = []
    for fp in sorted(root.rglob("*.fits")):
        if fp.parent.name.upper() != "LIGHT":
            continue
        try:
            hdr = fits.getheader(fp)
            if str(hdr.get("FILTER", "")).strip() != filter_name:
                continue
            if round(float(hdr.get("EXPTIME", 0))) != exptime_s:
                continue
            out.append(fp)
        except Exception:
            continue
    return out


def _run_curve(paths: list[Path], filter_name: str, label: str, out_dir: Path) -> dict:
    """One convergence curve, plus the decay ratio that interprets it."""
    from stacking import stacker

    calibration = stacker.calibration_from_config(filter_name)
    if calibration is None:
        print("  WARNING: no calibration masters — RMSE%% is a fraction of the bias\n"
              "           pedestal, not of sky, and is ~100x compressed. The raw-vs-\n"
              "           denoised comparison still holds; the absolute numbers do not.")
    counts, residuals, slope_pct, final_rmse_pct = stacker.convergence_curve(
        paths,
        filter_name=f"{filter_name} ({label})",
        output_path=out_dir / f"convergence_{label}.jpg",
        calibration=calibration,
        progress_cb=lambda m: print(f"    {m}"),
    )
    return {
        "label": label,
        "counts": counts,
        "residuals": residuals,
        "slope_pct": slope_pct,
        "final_rmse_pct": final_rmse_pct,
        "decay_ratio": decay_ratio(counts, residuals),
        "calibrated": calibration is not None,
    }


def _stack_for_photometry(paths: list[Path], filter_name: str) -> np.ndarray:
    """A full stack of *paths*, as the photometry test's input.

    downscale_to=None is essential and not a detail: the convergence path
    downsamples to ~512 px for speed, which at this image scale would smear
    every star into a pixel or two and destroy the aperture measurement before
    it started. Photometry runs at native resolution.
    """
    from stacking import stacker
    calibration = stacker.calibration_from_config(filter_name)
    frames, accepted, fwhm_values = stacker._prepare_for_convergence(
        paths, register=True, calibration=calibration, downscale_to=None,
        progress_cb=lambda m: print(f"    {m}"),
    )
    arr = np.stack(frames, axis=0)
    weights = stacker._fwhm_weights(fwhm_values, accepted)
    stacked = stacker._combine_tile(arr, stacker.StackMethod.SIGMA_CLIP_FWHM,
                                    weights, sigma=3.0)
    bad = np.isnan(stacked)
    if bad.any():
        stacked[bad] = float(np.nanmedian(stacked))
    return stacked


def _photometry(raw_stack: np.ndarray, den_stack: np.ndarray) -> list[dict]:
    """Aperture flux on stars matched between the two stacks, binned by mag.

    Sources are detected on the RAW stack only and the same apertures are
    measured on both. Detecting separately would compare different star lists
    and quietly hide the very effect being looked for -- a source the denoiser
    erased would simply not appear in its list.
    """
    import sep
    from scipy.spatial import cKDTree

    raw = np.ascontiguousarray(raw_stack, dtype=np.float64)
    den = np.ascontiguousarray(den_stack, dtype=np.float64)
    bkg = sep.Background(raw)
    raw_sub = raw - bkg
    den_sub = den - np.array(sep.Background(den))

    sources = sep.extract(raw_sub, thresh=5.0, err=bkg.globalrms)
    if len(sources) < 20:
        return []

    # Aperture scaled to the stack's own stars rather than fixed: this camera's
    # FWHM moves with seeing, and a radius that is generous on a sharp night
    # clips the wings on a soft one -- which the denoiser also touches, so a
    # fixed radius would blend a seeing effect into the thing being measured.
    r = float(max(3.0, 2.5 * np.median(sources["a"])))

    # Drop blends: a neighbour inside 3 aperture radii contaminates both
    # measurements, and unequally, since the denoiser reshapes wings.
    xy = np.column_stack([sources["x"], sources["y"]])
    tree = cKDTree(xy)
    isolated = np.array([len(tree.query_ball_point(p, 3 * r)) == 1 for p in xy])
    sources, xy = sources[isolated], xy[isolated]
    if len(sources) < 20:
        return []

    print(f"  aperture r={r:.1f} px, {len(sources)} isolated sources")
    f_raw, _, flag_r = sep.sum_circle(raw_sub, xy[:, 0], xy[:, 1], r,
                                      err=bkg.globalrms)
    f_den, _, flag_d = sep.sum_circle(den_sub, xy[:, 0], xy[:, 1], r,
                                      err=bkg.globalrms)
    ok = (flag_r == 0) & (flag_d == 0) & (f_raw > 0)
    f_raw, f_den = f_raw[ok], f_den[ok]
    if len(f_raw) < 20:
        return []

    mag = -2.5 * np.log10(f_raw / f_raw.max())
    rows = []
    for (lo, hi), name in zip(MAG_BINS, MAG_LABELS):
        m = (mag >= lo) & (mag < hi)
        if m.sum() < 5:
            continue
        ratio = f_den[m] / f_raw[m]
        rows.append({
            "bin": name,
            "n": int(m.sum()),
            "median_ratio": float(np.median(ratio)),
            "scatter": float(np.percentile(ratio, 84) - np.percentile(ratio, 16)) / 2,
        })
    return rows


def _verdict(raw: dict, den: dict, phot: list[dict]) -> list[str]:
    """Read the two measurements. Says "inconclusive" rather than guessing."""
    out = []
    rr, dr = raw["decay_ratio"], den["decay_ratio"]
    if rr is None or dr is None:
        out.append("CONVERGENCE: inconclusive — too few points to fit a decay ratio.")
    else:
        rmse_gain = 100.0 * (1 - den["final_rmse_pct"] / max(raw["final_rmse_pct"], 1e-9))
        if dr > rr * 1.5 and dr > 1.5:
            out.append(
                f"CONVERGENCE: FAIL — denoised decay ratio {dr:.2f} against raw {rr:.2f}. "
                f"The denoised residual is falling well short of independent noise, which "
                f"is what a bias shared across frames looks like. More frames will not "
                f"remove it.")
        elif rmse_gain > 5:
            out.append(
                f"CONVERGENCE: PASS — decay ratio {dr:.2f} vs raw {rr:.2f} (still averaging "
                f"like independent noise) and final RMSE {rmse_gain:.0f}% lower. On this "
                f"evidence the denoiser is genuinely buying frames.")
        else:
            out.append(
                f"CONVERGENCE: NEUTRAL — decay ratio {dr:.2f} vs raw {rr:.2f}, RMSE change "
                f"{rmse_gain:+.0f}%. No bias signature, but no real gain either.")

    if not phot:
        out.append("PHOTOMETRY: inconclusive — too few isolated, unflagged stars matched.")
        return out
    worst = min(phot, key=lambda r: r["median_ratio"])
    faint = [r for r in phot if r["bin"] in ("faint", "very faint")]
    if worst["median_ratio"] < 0.97:
        out.append(
            f"PHOTOMETRY: FAIL — the {worst['bin']} bin loses "
            f"{100 * (1 - worst['median_ratio']):.1f}% of its flux (n={worst['n']}). "
            f"That is a systematic brightness error, and every result on this site that "
            f"rests on measured brightness would inherit it.")
    elif faint and min(r["median_ratio"] for r in faint) < 0.99:
        out.append("PHOTOMETRY: MARGINAL — faint-end flux is suppressed by ~1%. Tolerable "
                   "for display, not for a 2% transit.")
    else:
        out.append("PHOTOMETRY: PASS — flux preserved to better than 1% in every bin.")
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 3:
        print(__doc__)
        return 1
    dso, filter_name = args[0].strip().lower(), args[1].strip()
    try:
        exptime_s = int(args[2])
    except ValueError:
        print(f"Error: seconds must be an integer, got '{args[2]}'")
        return 1

    cfg = config.data()
    hostname = socket.gethostname()
    machine_cfg = cfg.get("machine", {}).get(hostname) or {}
    if "subs_dir" not in machine_cfg:
        print(f"Error: machine.{hostname} has no 'subs_dir' in config_public.py")
        return 1
    subs_dir = Path(machine_cfg["subs_dir"])

    dso_dir = next((d for d in sorted(subs_dir.iterdir())
                    if d.is_dir() and d.name.lower().replace(" ", "") == dso), None)
    if dso_dir is None:
        print(f"Error: no directory for '{dso}' under {subs_dir}")
        return 1

    raw_paths = _light_frames(dso_dir, filter_name, exptime_s)
    den_root = subs_dir / "denoised" / f"{filter_name}_{exptime_s}s"
    # The denoised tree is flat and holds every DSO, so it is filtered down to
    # the frames whose names came from this target.
    raw_names = {p.name for p in raw_paths}
    den_paths = sorted(p for p in den_root.glob("*.fits") if p.name in raw_names) \
        if den_root.exists() else []

    print(f"{dso}  {filter_name} {exptime_s}s")
    print(f"  raw:      {len(raw_paths)} frames under {dso_dir}")
    print(f"  denoised: {len(den_paths)} frames under {den_root}")
    if len(raw_paths) < 8 or len(den_paths) < 8:
        print("\nError: need at least 8 frames on both sides for a meaningful curve.")
        print("  Denoise this target first: "
              f"python scripts/n2n_denoise.py {filter_name} {exptime_s}")
        return 1
    if len(den_paths) != len(raw_paths):
        print(f"  NOTE: counts differ ({len(raw_paths)} vs {len(den_paths)}); the "
              f"comparison is over different frame sets and is weaker for it.")

    out_dir = Path(_root) / "local" / "n2n_eval" / f"{dso}_{filter_name}_{exptime_s}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Convergence — raw")
    raw_res = _run_curve(raw_paths, filter_name, "raw", out_dir)
    print("\n[2/3] Convergence — denoised")
    den_res = _run_curve(den_paths, filter_name, "denoised", out_dir)

    print("\n[3/3] Aperture photometry, raw stack vs denoised stack")
    phot = []
    try:
        raw_stack = _stack_for_photometry(raw_paths, filter_name)
        den_stack = _stack_for_photometry(den_paths, filter_name)
        if raw_stack.shape != den_stack.shape:
            print("  Skipped: stacks differ in shape "
                  f"({raw_stack.shape} vs {den_stack.shape}).")
        else:
            phot = _photometry(raw_stack, den_stack)
    except Exception as exc:  # noqa: BLE001
        print(f"  Photometry failed: {exc}")

    print("\n" + "=" * 72)
    print(f"RESULT — {dso} {filter_name} {exptime_s}s")
    print("=" * 72)
    print(f"{'':10s} {'frames':>7s} {'RMSE%':>9s} {'slope%/fr':>11s} {'decay ratio':>12s}")
    for r in (raw_res, den_res):
        dr = "n/a" if r["decay_ratio"] is None else f"{r['decay_ratio']:.2f}"
        print(f"{r['label']:10s} {r['counts'][-1]:7d} {r['final_rmse_pct']:9.2f} "
              f"{r['slope_pct']:+11.3f} {dr:>12s}")
    if not raw_res["calibrated"]:
        print("  (uncalibrated: RMSE% is on the pedestal scale — relative only)")

    if phot:
        print(f"\n{'bin':12s} {'n':>5s} {'flux den/raw':>13s} {'scatter':>9s}")
        for r in phot:
            print(f"{r['bin']:12s} {r['n']:5d} {r['median_ratio']:13.4f} "
                  f"{r['scatter']:9.4f}")

    print()
    for line in _verdict(raw_res, den_res, phot):
        print(line)
    print(f"\nPlots: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
