"""Judge whether closed-roof darks are real darks or carry a light leak.

The question behind it: the QHY600M has no shutter, so autonomous darks mean
trusting the closed, unlit observatory to BE the shutter. That is a claim, so
it gets measured, two ways:

  python scripts/dark_leak_check.py <candidate_dir>
      Candidates vs the trusted reference library (cfg["calibration"]),
      per exposure time: dark-rate excess and large-scale spatial structure.
      The July 2026 library was shot capped, so it defines "no leak".

  python scripts/dark_leak_check.py --compare <lights_off_dir> <lights_on_dir>
      The injection test, and the stronger claim: shoot the SAME dark
      sequence twice in one session, once with the observatory interior
      lights deliberately ON. Lights-on is orders of magnitude brighter than
      any real night; if on-minus-off shows nothing above noise, the light
      path is sealed with margin to spare, and no blank filter or shutter is
      needed for calibration. If it shows structure, the tile map says which
      corner the leak enters from.

METHOD NOTES.
Frames are analysed 8x8-binned (means): a leak enters through the optical
path and is spatially SMOOTH, so binning keeps every leak photon while
averaging hot pixels away, and it keeps 61 MP frames from needing gigabytes.
All comparisons are per-exposure-time and quote the measured frame-to-frame
noise as the yardstick, so "clean" means "clean at N-frame depth", not clean
by eye. Match the reference's sensor temperature (-20 C) or the dark-current
term is comparing apples to oranges -- the report prints both temps rather
than assuming.
"""
import glob
import os
import sys

import numpy as np

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

BIN = 8
TILES = 4              # NxN tile grid for the structure map
SIGMA_BAR = 5.0        # structure above this many sigma = leak
MIN_FRAMES = 3         # below this a group has no measurable noise floor:
                       # N=1 gave MAD=0 and a zero difference read as an
                       # infinite-sigma "leak" -- judged UNJUDGEABLE instead


def _load_binned(path):
    """(binned_mean_frame float32, exposure_s, ccd_temp) for one FITS."""
    from astropy.io import fits
    # memmap=False: these frames carry BZERO (unsigned 16-bit convention),
    # which astropy refuses to scale through a memory map.
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = np.asarray(hdu.data, dtype=np.float32)
        hdr = hdu.header
    exp = float(hdr.get("EXPOSURE", hdr.get("EXPTIME", 0.0)))
    temp = hdr.get("CCD-TEMP", hdr.get("CCD_TEMP"))
    h, w = data.shape
    h2, w2 = h - h % BIN, w - w % BIN
    binned = data[:h2, :w2].reshape(h2 // BIN, BIN, w2 // BIN, BIN).mean(axis=(1, 3))
    return binned, exp, temp


def _group_by_exposure(paths):
    groups = {}
    temps = []
    for p in paths:
        frame, exp, temp = _load_binned(p)
        groups.setdefault(round(exp, 2), []).append(frame)
        if temp is not None:
            temps.append(float(temp))
    return groups, temps


def _stack_stats(frames):
    """(median_frame, per_binned_pixel_noise_of_the_median)."""
    cube = np.stack(frames)
    med = np.median(cube, axis=0)
    # Frame-to-frame scatter, robustly, then reduced by sqrt(N) for the median
    # (1.25 factor for median vs mean efficiency -- close enough at this N).
    mad = np.median(np.abs(cube - med[None]), axis=0) * 1.4826
    noise = float(np.median(mad)) / max(1.0, np.sqrt(len(frames))) * 1.25
    return med, noise


def _tile_map(diff, tiles=TILES):
    """Median of each tile of the difference image, as a tiles x tiles array."""
    h, w = diff.shape
    out = np.zeros((tiles, tiles))
    for i in range(tiles):
        for j in range(tiles):
            out[i, j] = np.median(diff[i * h // tiles:(i + 1) * h // tiles,
                                       j * w // tiles:(j + 1) * w // tiles])
    return out


def _report_structure(diff, noise, label):
    tm = _tile_map(diff)
    spread = tm.max() - tm.min()
    sig = spread / noise if noise > 0 else float("inf")
    print("  %s: tile spread %.4f ADU (%.1f sigma of the %d-frame noise floor "
          "%.4f ADU)" % (label, spread, sig, TILES, noise))
    if sig > SIGMA_BAR:
        i, j = np.unravel_index(tm.argmax(), tm.shape)
        print("    LEAK-LIKE STRUCTURE: brightest tile row %d col %d "
              "(0,0 = top-left of the frame)" % (i, j))
        for row in tm:
            print("      " + "  ".join("%+.4f" % v for v in row))
        return False
    print("    clean: no large-scale structure above %.0f sigma" % SIGMA_BAR)
    return True


def _fits_in(d):
    fs = sorted(glob.glob(os.path.join(d, "*.fits")))
    if not fs:
        sys.exit("no .fits in %s" % d)
    return fs


def compare_mode(dir_off, dir_on):
    print("Injection test: lights-ON minus lights-OFF (same exposures)")
    g_off, t_off = _group_by_exposure(_fits_in(dir_off))
    g_on, t_on = _group_by_exposure(_fits_in(dir_on))
    if t_off and t_on:
        print("  sensor temp: off %.1fC / on %.1fC" %
              (np.median(t_off), np.median(t_on)))
    ok = True
    for exp in sorted(set(g_off) & set(g_on)):
        if len(g_off[exp]) < MIN_FRAMES or len(g_on[exp]) < MIN_FRAMES:
            print("%gs: only %d/%d frames -- too few to establish a noise "
                  "floor, skipped" % (exp, len(g_off[exp]), len(g_on[exp])))
            continue
        m_off, n_off = _stack_stats(g_off[exp])
        m_on, n_on = _stack_stats(g_on[exp])
        noise = float(np.hypot(n_off, n_on))
        diff = m_on - m_off
        print("%gs: %d off / %d on frames, mean(on-off) %+.4f ADU"
              % (exp, len(g_off[exp]), len(g_on[exp]), float(np.median(diff))))
        ok &= _report_structure(diff, noise, "on-off structure")
    only = (set(g_off) ^ set(g_on))
    if only:
        print("  (exposures without a partner, skipped: %s)" % sorted(only))
    print("\nVERDICT: %s" % (
        "SEALED -- interior lights full-on produced nothing above noise; the "
        "closed observatory is a shutter, with margin" if ok else
        "LEAK -- structure appeared under lights-on; see tile maps for where"))
    return 0 if ok else 1


def reference_mode(cand_dir):
    from configs import config
    cal = config.data().get("calibration", {})
    ref_dir = cal.get("dark_dir")
    print("Candidates: %s\nReference : %s (the capped library)" % (cand_dir, ref_dir))
    g_cand, t_cand = _group_by_exposure(_fits_in(cand_dir))
    g_ref, t_ref = _group_by_exposure(_fits_in(ref_dir)) if ref_dir and os.path.isdir(ref_dir) else ({}, [])
    if t_cand:
        print("  candidate sensor temp %.1fC%s" % (
            np.median(t_cand),
            (" / reference %.1fC" % np.median(t_ref)) if t_ref else ""))
    ok = True
    for exp, frames in sorted(g_cand.items()):
        med, noise = _stack_stats(frames)
        print("%gs: %d candidate frames, median level %.3f ADU" %
              (exp, len(frames), float(np.median(med))))
        if len(frames) < MIN_FRAMES or len(g_ref.get(exp, [])) < MIN_FRAMES:
            print("  too few frames on one side to establish a noise floor -- "
                  "not judged (shoot at least %d)" % MIN_FRAMES)
            continue
        if exp in g_ref:
            ref_med, ref_noise = _stack_stats(g_ref[exp])
            diff = med - ref_med
            print("  vs reference (%d frames): level excess %+.4f ADU"
                  % (len(g_ref[exp]), float(np.median(diff))))
            ok &= _report_structure(diff, float(np.hypot(noise, ref_noise)),
                                    "candidate-reference structure")
        else:
            ok &= _report_structure(med - float(np.median(med)), noise,
                                    "internal structure (no reference at this exposure)")
    print("\nVERDICT: %s" % (
        "MATCHES the capped reference -- these are real darks" if ok else
        "DIFFERS from the capped reference -- see structure above"))
    return 0 if ok else 1


def main():
    args = [a for a in sys.argv[1:] if a != "--compare"]
    if "--compare" in sys.argv:
        if len(args) != 2:
            sys.exit("usage: dark_leak_check.py --compare <lights_off_dir> <lights_on_dir>")
        return compare_mode(args[0], args[1])
    if len(args) != 1:
        sys.exit("usage: dark_leak_check.py <candidate_dir> | --compare <off> <on>")
    return reference_mode(args[0])


if __name__ == "__main__":
    sys.exit(main())
