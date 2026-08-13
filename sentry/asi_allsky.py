#!/usr/bin/env python3
"""Capture the sky with the ASI120MC-S fisheye, and find the exposure that
shows the most of it.

This camera sits under the roof, so it photographs sky only while the roof is
open. Everything here is written for that: a capture is cheap and repeatable,
and the settings that make it worth looking at are MEASURED rather than
assumed.

Auto-exposure here does not mean metering the brightness. Metering optimises the
picture, and the thing being optimised is the star count, which is not the same
quantity -- a longer exposure raises the sky as fast as it raises the stars, so
past some point every extra second buys background and nothing else. So the
sweep runs the real detector on every rung and keeps whichever setting actually
delivers the most trustworthy detections. The winner is cached, because the
answer only changes when the focus or the sky does.

Three things about this sensor that shape the code:

  * It is a COLOUR camera. The raw frame is a Bayer mosaic, and left as-is the
    mosaic itself reads as fine detail -- the detector would measure the pattern
    rather than the sky. Frames are debayered and reduced to luminance first.

  * It is UNCOOLED, and every uncooled sensor has hot pixels: single-pixel
    spikes far brighter than any star, sitting in the same place every frame.
    They are exactly what a point-source detector is built to find. The Kasa
    camera keeps them out with a 6-pixel minimum blob size, but this lens puts a
    focused star into fewer pixels than that, so the floor has to come down and
    something else has to remove the spikes. That something is a master dark,
    which the roof being SHUT makes easy to collect -- see build_darks().

  * Frames are stored as FITS, not JPEG. The faint end of the completeness table
    lives in the bottom few percent of the range, and an 8-bit JPEG does not
    have it to give. A stretched JPEG is written alongside, for looking at.

Usage:
    python sentry/asi_allsky.py --capture [--exposure S] [--gain G]
    python sentry/asi_allsky.py --autoexpose [--save]
    python sentry/asi_allsky.py --dark            # roof CLOSED; builds the darks
    python sentry/asi_allsky.py --settings        # show what is cached
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from sentry import star_count

PROFILE = "allsky camera"
FULL = 65535.0          # RAW16 full scale
ROOT = Path(__file__).resolve().parents[1]


def cfg(profile=PROFILE):
    return config.data().get(profile, {})


def _path(key, default, profile=PROFILE):
    return ROOT / cfg(profile).get(key, default)


# ----------------------------------------------------------------- camera

class _Camera:
    """Open the all-sky camera for as long as the block runs, then close it.

    Closing matters more than it looks: the SDK hands out one handle per camera
    and a process that exits without releasing it can leave the device
    unavailable to the next one, which on a scheduled job means the feed dies
    until someone unplugs the camera.
    """

    def __init__(self, profile=PROFILE):
        self.profile = profile
        self.cam = None
        self.prop = None
        self.name = None

    def __enter__(self):
        import zwoasi as asi
        c = cfg(self.profile)
        asi.init(c.get("sdk"))
        names = asi.list_cameras()
        match = str(c.get("camera_match", "120"))
        idx = [i for i, n in enumerate(names) if match in n]
        if not idx:
            raise RuntimeError("no camera matching %r; found %s" % (match, names))
        self.name = names[idx[0]]
        self.cam = asi.Camera(idx[0])
        self.cam.set_image_type(asi.ASI_IMG_RAW16)
        self.prop = self.cam.get_camera_property()
        self.cam.set_roi(0, 0, self.prop["MaxWidth"], self.prop["MaxHeight"])
        return self

    def __exit__(self, *exc):
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
        return False

    def gain_limits(self):
        """(min, max) gain this camera accepts. Asked, never assumed."""
        import zwoasi as asi
        for ctrl in self.cam.get_controls().values():
            if ctrl.get("ControlType") == asi.ASI_GAIN:
                return int(ctrl["MinValue"]), int(ctrl["MaxValue"])
        return 0, 100

    def raw(self, exposure_s, gain):
        """One RAW16 frame at these settings, as the sensor delivered it."""
        import zwoasi as asi
        lo, hi = self.gain_limits()
        gain = int(np.clip(gain, lo, hi))
        self.cam.set_control_value(asi.ASI_EXPOSURE, int(round(exposure_s * 1e6)))
        self.cam.set_control_value(asi.ASI_GAIN, gain)
        # capture() polls the exposure status in a loop. Sleeping through the
        # exposure first, and then polling slowly, keeps a 30-second frame from
        # costing three thousand round trips to the driver for an answer that
        # cannot change until the shutter time is up.
        return np.asarray(self.cam.capture(initial_sleep=max(exposure_s - 0.05, 0.01),
                                           poll=0.05))


def to_luminance(raw, prop):
    """A Bayer mosaic as a single luminance plane; a mono frame unchanged.

    Debayering before anything else is not cosmetic. On a colour sensor the raw
    frame alternates filters pixel by pixel, so a flat grey sky arrives as a
    checkerboard with an amplitude set by the filters rather than the scene. A
    high-pass point-source detector reads that as structure everywhere.
    """
    if not prop.get("IsColorCam"):
        return raw.astype(np.float32)
    import cv2
    pattern = {0: cv2.COLOR_BAYER_RG2RGB, 1: cv2.COLOR_BAYER_BG2RGB,
               2: cv2.COLOR_BAYER_GR2RGB, 3: cv2.COLOR_BAYER_GB2RGB}
    rgb = cv2.cvtColor(raw, pattern.get(prop.get("BayerPattern", 0),
                                        cv2.COLOR_BAYER_RG2RGB))
    # Plain mean, not the Rec.601 luma weights. Those are tuned for how bright a
    # colour looks to a human eye, and weighting blue at 0.07 would throw away
    # most of the signal from exactly the hot blue stars this is trying to
    # detect. Every channel carries photons, so every channel counts equally.
    return rgb.astype(np.float32).mean(axis=2)


def clipped_pct(a):
    """Percent of the frame pinned at full scale.

    A saturated star is a flat-topped blob, not a point, so clipping does not
    merely lose brightness -- it removes the star from the count entirely, and
    does it to the brightest stars first.
    """
    return 100.0 * float(np.mean(a >= 0.98 * FULL))


# ------------------------------------------------------------------ darks

def dark_path(exposure_s, gain, profile=PROFILE):
    """Where the master dark for exactly these settings lives.

    Named in whole microseconds rather than seconds. A one-decimal second is
    lossy below 100 ms -- 20 ms and 40 ms both round to "0.0s" and would share a
    file, so one exposure's dark would be subtracted from another's frames.
    """
    d = _path("dark_dir", "local/allsky_darks", profile)
    return d / ("dark_%dus_g%d.fits" % (round(exposure_s * 1e6), gain))


def load_dark(exposure_s, gain, profile=PROFILE):
    """The master dark for these settings, or None if it was never taken.

    Matched EXACTLY on exposure and gain rather than scaled from a nearby one.
    Dark current scales with time but amp glow does not scale with anything
    simple, and a mis-scaled dark subtracts a pattern that was never there.
    """
    p = dark_path(exposure_s, gain, profile)
    if not p.exists():
        return None
    from astropy.io import fits
    with fits.open(str(p)) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


MIN_EXPOSURE_S = 64e-6      # the sensor's floor; a frame this short is bias
LIT_FRACTION = 0.05         # above this, the "dark" is a picture of something


def lit_fraction(master, bias_level):
    """Fraction of a supposed dark sitting well clear of the bias level.

    The discriminator between a dark and a photograph. In a genuinely dark
    enclosure the only things above bias are hot pixels and amp glow -- a
    percent or two of the frame. With light getting in, most of the frame is
    above bias, because most of the frame is a picture of the thing the light is
    falling on.
    """
    sigma = _mad(master)
    return float(np.mean(master > bias_level + 10.0 * sigma))


def build_darks(exposures=None, gains=None, frames=3, profile=PROFILE,
                verbose=True, force=False):
    """Take master darks over the sweep grid. THE ROOF MUST BE SHUT, AND DARK.

    A dark is a frame with no light in it, and a closed roof supplies that
    without anyone having to cap a lens the camera does not have. But a shut
    roof is not the same as a dark one: in daylight enough gets past the seals
    to give a clearly exposed picture of the roof's own underside, and adopting
    that as a dark would subtract the roof out of every night frame afterwards.
    So the light level is measured against the sensor's bias and the job refuses
    what is plainly a photograph.

    The right moment is after dusk with the roof still shut, before it opens.

    Each master is the per-pixel MEDIAN of *frames* exposures. The median is
    what makes it a dark rather than a noise map: cosmic rays and the random
    half of the read noise average out, while the hot pixels -- which are in the
    same place every time -- survive at full strength, which is precisely the
    part that needs subtracting.
    """
    from astropy.io import fits
    c = cfg(profile)
    exposures = exposures or [float(e) for e in c.get("sweep_exposures_s", (1.0,))]
    gains = gains or [int(g) for g in c.get("sweep_gains", (100,))]
    out_dir = _path("dark_dir", "local/allsky_darks", profile)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, refused = [], []
    with _Camera(profile) as cam:
        if verbose:
            print("%s  %dx%d" % (cam.name, cam.prop["MaxWidth"],
                                 cam.prop["MaxHeight"]))
        for gain in gains:
            # Bias per gain, not once overall: the offset the sensor reads out
            # at zero exposure moves with gain, so one bias level compared
            # against every gain's darks would misjudge most of them.
            bias = to_luminance(cam.raw(MIN_EXPOSURE_S, gain), cam.prop)
            bias_level = float(np.median(bias))
            if verbose:
                print("  gain %3d: bias %.1f ADU at the %.0fus floor"
                      % (gain, bias_level, MIN_EXPOSURE_S * 1e6))
            for exp in exposures:
                stack = [to_luminance(cam.raw(exp, gain), cam.prop)
                         for _ in range(frames)]
                master = np.median(np.stack(stack), axis=0).astype(np.float32)
                lit = lit_fraction(master, bias_level)
                hot = int(np.sum(master > np.median(master) + 20 * _mad(master)))
                if lit > LIT_FRACTION and not force:
                    refused.append((exp, gain, lit))
                    if verbose:
                        print("  %6gs gain %3d -> REFUSED: %.0f%% of the frame "
                              "is lit; this is a picture, not a dark"
                              % (exp, gain, 100 * lit))
                    continue
                p = dark_path(exp, gain, profile)
                hdu = fits.PrimaryHDU(master)
                hdu.header["EXPTIME"] = exp
                hdu.header["GAIN"] = gain
                hdu.header["NFRAMES"] = frames
                hdu.header["BIAS"] = bias_level
                hdu.header["LITFRAC"] = round(lit, 4)
                hdu.header["DATE"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds")
                hdu.header["COMMENT"] = "master dark, roof closed; median of NFRAMES"
                hdu.writeto(str(p), overwrite=True)
                written.append(p)
                if verbose:
                    print("  %6gs gain %3d -> median %7.1f  lit %4.1f%%  "
                          "hot pixels %5d  %s"
                          % (exp, gain, float(np.median(master)), 100 * lit,
                             hot, p.name))
    if refused and verbose:
        print("\n%d rung(s) refused. If the roof really is shut, this is "
              "daylight getting past the seals -- wait until after dark and "
              "run it again." % len(refused))
    return written


def _mad(a):
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) or 1.0


def hot_pixel_mask(dark, sigma=20.0):
    """Pixels a master dark says are spikes, not sky.

    Cut on the dark's own scatter rather than an absolute level, so it holds at
    any exposure and gain without a second number to keep in step.
    """
    return dark > (np.median(dark) + sigma * _mad(dark))


def _repair(a, mask):
    """Replace masked pixels with their surroundings.

    Set to the local median rather than zero. A zero is a hole, and the
    high-pass stage the detector starts with turns a hole into a strong NEGATIVE
    spike -- which the negative-image control then counts as a false positive,
    quietly wrecking the purity measurement that the whole threshold rests on.
    """
    if not mask.any():
        return a
    from scipy import ndimage
    out = a.copy()
    out[mask] = ndimage.median_filter(a, 5)[mask]
    return out


# ---------------------------------------------------------------- capture

def capture(exposure_s=None, gain=None, profile=PROFILE, out_path=None,
            jpeg=True, verbose=False):
    """One calibrated luminance frame. Returns (array, meta).

    Falls back to the cached auto-exposure settings when none are given, and
    refuses to invent them: an all-sky frame at the wrong exposure is not a
    worse reading of the sky, it is a reading of nothing.
    """
    st = load_settings(profile) or {}
    if exposure_s is None:
        exposure_s = st.get("exposure_s")
    if gain is None:
        gain = st.get("gain")
    if exposure_s is None or gain is None:
        raise RuntimeError(
            "no exposure cached for %r; run "
            "`python sentry/asi_allsky.py --autoexpose --save` under a dark sky"
            % profile)
    exposure_s, gain = float(exposure_s), int(gain)

    with _Camera(profile) as cam:
        a = to_luminance(cam.raw(exposure_s, gain), cam.prop)
        cam_name, prop = cam.name, cam.prop

    meta = {"exposure_s": exposure_s, "gain": gain, "camera": cam_name,
            "clip_pct": round(clipped_pct(a), 3),
            "level_adu": round(float(np.median(a)), 1),
            "captured": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    dark = load_dark(exposure_s, gain, profile)
    if dark is None:
        # Loud, because the consequence is silent: with no dark the hot pixels
        # stay in, and at this profile's minimum blob size they are counted as
        # stars. The number would look plausible and be wrong.
        meta["dark"] = None
        meta["warning"] = ("no master dark for %gs gain %d -- hot pixels will "
                           "be counted as stars; run --dark after dark with the "
                           "roof still shut" % (exposure_s, gain))
        if verbose:
            print("WARNING:", meta["warning"])
    else:
        hot = hot_pixel_mask(dark)
        a = _repair(a - dark, hot)
        meta["dark"] = dark_path(exposure_s, gain, profile).name
        meta["hot_pixels"] = int(hot.sum())

    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = _path("capture_dir", "local/allsky_frames", profile)
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / ("allsky_" + stamp + ".fits")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fits(a, out_path, meta, prop)
    meta["path"] = str(out_path)
    if jpeg:
        meta["jpeg"] = str(save_jpeg(a, out_path.with_suffix(".jpg")))
    return a, meta


def save_fits(a, path, meta, prop=None):
    from astropy.io import fits
    hdu = fits.PrimaryHDU(a.astype(np.float32))
    hdu.header["EXPTIME"] = float(meta["exposure_s"])
    hdu.header["GAIN"] = int(meta["gain"])
    hdu.header["DATE-OBS"] = meta["captured"]
    hdu.header["INSTRUME"] = str(meta.get("camera", ""))[:60]
    if meta.get("dark"):
        hdu.header["DARK"] = meta["dark"]
    hdu.writeto(str(path), overwrite=True)
    return path


def save_jpeg(a, path, percentiles=(20.0, 99.8)):
    """A viewable stretch of the frame. For looking at, never for measuring.

    The low point is a percentile rather than the minimum because an all-sky
    frame is mostly sky: anchoring at the true minimum spends the whole range on
    the darkest corner and leaves the stars in the top few levels.
    """
    from PIL import Image
    lo, hi = np.percentile(a, percentiles[0]), np.percentile(a, percentiles[1])
    b = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1)
    # Square root, so the faint stars survive the 8 bits. A linear map puts
    # almost everything in the bottom few levels of a sky-dominated frame.
    b = (np.sqrt(b) * 255).astype(np.uint8)
    Image.fromarray(b).save(str(path), quality=90)
    return Path(path)


# ------------------------------------------------------------ autoexposure

def autoexpose(exposures=None, gains=None, profile=PROFILE, verbose=True,
               annotate_dir=None):
    """Sweep exposure and gain; keep whatever shows the most stars.

    Scored with the same detector that publishes the number, so the winner is
    the setting that genuinely reveals the most sky rather than the one that
    looks best exposed. Rungs that clip beyond the profile's limit are
    disqualified outright, and so are rungs the detector does not trust: on a
    frame full of cloud or hot pixels the raw count goes UP, so an unguarded
    "most detections" objective would reliably choose the worst frame of the
    sweep.

    Returns (best, rows). *best* is None when nothing qualified.
    """
    c = cfg(profile)
    exposures = exposures or [float(e) for e in c.get("sweep_exposures_s", (1.0,))]
    gains = gains or [int(g) for g in c.get("sweep_gains", (100,))]
    clip_limit = float(c.get("clip_limit_pct", 0.5))

    rows = []
    with _Camera(profile) as cam:
        if verbose:
            lo, hi = cam.gain_limits()
            print("%s  %dx%d  gain %d-%d  colour=%s"
                  % (cam.name, cam.prop["MaxWidth"], cam.prop["MaxHeight"],
                     lo, hi, bool(cam.prop.get("IsColorCam"))))
            print("%8s %5s %8s %7s %7s %7s %7s  %s"
                  % ("exp(s)", "gain", "level", "clip%", "stars", "purity",
                     "fwhm", "verdict"))
        for gain in gains:
            for exp in exposures:
                a = to_luminance(cam.raw(exp, gain), cam.prop)
                clip = clipped_pct(a)
                dark = load_dark(exp, gain, profile)
                if dark is not None:
                    a = _repair(a - dark, hot_pixel_mask(dark))
                res = star_count.count_stars(a, profile=profile)
                row = {"exposure_s": exp, "gain": gain,
                       "level_adu": round(float(np.median(a)), 1),
                       "clip_pct": round(clip, 3),
                       "stars": res["stars"], "purity": res["purity"],
                       "trustworthy": res["trustworthy"],
                       "false_positives": res["false_positives"],
                       "median_fwhm_px": res["median_fwhm_px"],
                       "noise_adu": res["noise_adu"],
                       "dark": dark is not None}
                # Disqualified, not merely down-ranked. A clipped or untrusted
                # rung has no meaningful star count to compare at all.
                if clip > clip_limit:
                    row["rejected"] = "clipped %.2f%% > %.2f%%" % (clip, clip_limit)
                elif not res["trustworthy"]:
                    row["rejected"] = ("purity %.2f -- %d of %d detections false"
                                       % (res["purity"], res["false_positives"],
                                          res["stars"]))
                rows.append(row)
                if annotate_dir:
                    d = Path(annotate_dir)
                    d.mkdir(parents=True, exist_ok=True)
                    save_jpeg(a, d / ("sweep_%.1fs_g%d.jpg" % (exp, gain)))
                if verbose:
                    print("%8.3f %5d %8.1f %7.3f %7d %7.3f %7s  %s"
                          % (exp, gain, row["level_adu"], clip, res["stars"],
                             res["purity"],
                             "-" if res["median_fwhm_px"] is None
                             else "%.2f" % res["median_fwhm_px"],
                             row.get("rejected", "ok")))

    ok = [r for r in rows if "rejected" not in r]
    best = max(ok, key=lambda r: r["stars"]) if ok else None
    if verbose:
        if best is None:
            print("\nnothing qualified -- every rung clipped or was untrusted.")
            if all("purity" in r.get("rejected", "") for r in rows):
                # The likeliest cause by far on a camera whose threshold has not
                # been set yet: a detection threshold in the wrong place makes
                # every rung fail purity at once, which looks like bad sky and
                # is not. Point at the measurement that fixes it rather than
                # leaving someone to re-run the sweep at different exposures.
                print("every rung failed on PURITY, not on exposure. That is "
                      "what a detection threshold in the wrong place looks "
                      "like -- take one frame and set it from the negative "
                      "control:\n"
                      '  python sentry/asi_allsky.py --capture --exposure 8 --gain 100\n'
                      '  python sentry/star_count.py <that .fits> --profile '
                      '"allsky camera" --sweep\n'
                      "then put the chosen value in star_threshold_adu and run "
                      "this again.")
        else:
            print("\nbest: %.1fs at gain %d -- %d stars, purity %.3f, FWHM %s px"
                  % (best["exposure_s"], best["gain"], best["stars"],
                     best["purity"],
                     "-" if best["median_fwhm_px"] is None
                     else "%.2f" % best["median_fwhm_px"]))
            _focus_advice(best, rows)
    return best, rows


def _focus_advice(best, rows):
    """Say whether the focus, rather than the exposure, is what is limiting.

    Worth saying out loud because the two failures look identical from the star
    count alone. A star focused into fewer pixels than the detector's blob floor
    is simply not counted, so a perfectly focused lens can score WORSE than a
    slightly soft one -- and the fix is to turn the ring, not to change the
    exposure this sweep is searching over.
    """
    f = best.get("median_fwhm_px")
    if f is None:
        print("no stars measured, so nothing to say about focus.")
        return
    if f < 1.8:
        print("stars are %.2f px across, which is at or under this detector's "
              "floor -- some are being missed for being too small, not too "
              "faint. Defocusing very slightly should RAISE the count; check "
              "with scripts/asi_focus.py." % f)
    elif f > 5.0:
        print("stars are %.2f px across, which is soft. Focusing should raise "
              "both the count and the limiting magnitude; scripts/asi_focus.py "
              "gives a live meter." % f)
    else:
        print("star width %.2f px is in a good range; focus is not what is "
              "limiting this." % f)


# --------------------------------------------------------------- settings

def settings_path(profile=PROFILE):
    return _path("settings_file", "local/allsky_settings.json", profile)


def load_settings(profile=PROFILE):
    p = settings_path(profile)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def save_settings(best, rows=None, profile=PROFILE):
    """Store the winning setting, and the sweep that chose it.

    The whole table is kept, not just the winner. A single number gives no way
    to tell a clear choice from a coin toss between two rungs one star apart,
    and the next person to look at this will want to know which it was.
    """
    p = settings_path(profile)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(best)
    out["chosen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if rows is not None:
        out["sweep"] = rows
    p.write_text(json.dumps(out, indent=1))
    return p


# -------------------------------------------------------------------- cli

def main(argv):
    argv = list(argv)
    prof = star_count.pop_arg(argv, "--profile", PROFILE)
    exp = star_count.pop_arg(argv, "--exposure")
    gain = star_count.pop_arg(argv, "--gain")

    if "--settings" in argv:
        st = load_settings(prof)
        print(json.dumps(st, indent=1) if st else "no cached settings for %r" % prof)
        return 0 if st else 1

    if "--dark" in argv:
        print("Building master darks. THE ROOF MUST BE SHUT, AND IT MUST BE "
              "DARK OUTSIDE -- anything these frames can see is subtracted out "
              "of every capture from now on.\n")
        written = build_darks(profile=prof,
                              exposures=[float(exp)] if exp else None,
                              gains=[int(gain)] if gain else None,
                              force="--force" in argv)
        print("\nwrote %d master dark(s) to %s"
              % (len(written), _path("dark_dir", "local/allsky_darks", prof)))
        return 0

    if "--autoexpose" in argv:
        best, rows = autoexpose(
            profile=prof,
            exposures=[float(exp)] if exp else None,
            gains=[int(gain)] if gain else None,
            annotate_dir=star_count.pop_arg(argv, "--annotate-dir"))
        if best is None:
            return 1
        if "--save" in argv:
            print("saved ->", save_settings(best, rows, prof))
        else:
            print("(not saved; pass --save to keep it)")
        return 0

    if "--capture" in argv:
        a, meta = capture(float(exp) if exp else None,
                          int(gain) if gain else None,
                          profile=prof, verbose=True)
        print(json.dumps({k: v for k, v in meta.items()}, indent=1))
        res = star_count.count_stars(a, profile=prof)
        print("%d stars, purity %.3f, FWHM %s px"
              % (res["stars"], res["purity"], res["median_fwhm_px"]))
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
