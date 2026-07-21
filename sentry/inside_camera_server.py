import contextlib
import shutil

import cv2 as cv
import numpy as np
import os, sys
import asyncio
import threading
import time



if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from cmd_processing import super_user_commands
from hardware_control import kasa_utils as ku
from utils import utils




_loger = utils.set_logger()

def best_exposure_score(img):
    if len(img.shape) == 3:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)  # Luminance channel (0-255)
    else:
        L = img.astype(np.float32)

    L_norm = L / 255.0

    # Count clipped pixels — use lenient thresholds since observatory scenes
    # legitimately have dark shadow regions (telescope body, corners)
    under = np.sum(L < 16) / L.size   # < ~6% brightness
    over = np.sum(L > 240) / L.size   # > ~94% brightness

    mean_lum = np.mean(L_norm)
    std_lum = np.std(L_norm)

    # Additive penalty only kicks in when clipping is heavy (>30% under, >5% over).
    # Previously the penalty was multiplicative and went negative for any clipping
    # above 12.5%, causing max() to pick the darkest (nearly black) frame.
    clip_penalty = max(0.0, under - 0.30) * 3.0 + max(0.0, over - 0.05) * 3.0

    # Reward frames close to target brightness (0.45) with high contrast
    mean_reward = np.exp(-8.0 * (mean_lum - 0.45) ** 2)

    score = std_lum * mean_reward - clip_penalty
    return score


def dark_sky_score(img):
    """Exposure score for a NO-LIGHT view of the night sky (the `live` command).

    Unlike best_exposure_score, which targets a well-lit indoor scene (mean ~0.45),
    the sky is intrinsically dark: the goal is to gather as much light and detail
    as possible (sky glow, clouds, the moon, bright stars) without blowing out
    highlights. So we reward brightness + contrast and penalize only heavy
    over-exposure — this naturally picks the longest exposure that isn't clipped
    (and backs off when e.g. the moon is up).
    """
    if len(img.shape) == 3:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
    else:
        L = img.astype(np.float32)

    L_norm = L / 255.0
    over = np.sum(L > 240) / L.size   # blown highlights (moon / stray light)
    mean_lum = np.mean(L_norm)
    std_lum = np.std(L_norm)

    # Brightness + contrast reward; steep penalty once >2% of the frame clips.
    return (mean_lum + std_lum) - 5.0 * max(0.0, over - 0.02)


def gamma_correction(img, gamma=1.0):
    # Build a lookup table (fastest method)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)


# Process-wide camera mutex. There is a single USB camera (VideoCapture(0)) and
# a single inside-light state that take_snapshot saves/restores. Two threads
# running this at once (e.g. an in-flight roof-open confirm loop plus a manual
# status/close) opened the camera concurrently — OpenCV throws "Unknown C++
# exception" from vid.set(...) — and raced the light save/restore, leaving the
# scene dark so vision misclassified the roof. Serialize every snapshot.
#
# Reentrant so a caller can hold the lock across a snapshot AND its own read of
# the shared scope_view.jpg (see camera_session) while take_snapshot re-acquires
# it internally.
_camera_lock = threading.RLock()


@contextlib.contextmanager
def camera_session():
    """Hold the camera mutex across a snapshot and the caller's subsequent read
    of scope_view.jpg.

    take_snapshot()'s first act is to clobber scope_view.jpg with a placeholder,
    so a concurrent snapshot can overwrite the file between another caller's
    snapshot and its imread — yielding a None/placeholder frame or a
    misclassified state. Wrapping snapshot+read in this session makes the pair
    atomic; the RLock lets the nested take_snapshot re-acquire the same lock.
    """
    with _camera_lock:
        yield


def _capture_burst(exposure, n, light):
    """Reopen the USB camera at a fixed exposure and grab up to ``n`` frames.

    Mirrors the dark-sky setup (max gain, lifted brightness) when ``light`` is
    False, matching the sweep. Used for the final robust capture / multi-frame
    stack. Returns a list of frames (possibly shorter than ``n``).
    """
    frames = []
    vid = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not vid.isOpened():
        _loger.warning("burst capture: camera failed to open")
        return frames
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    time.sleep(0.5)
    if not light:
        vid.set(cv.CAP_PROP_GAIN, 100)
        vid.set(cv.CAP_PROP_BRIGHTNESS, 0)
    vid.set(cv.CAP_PROP_EXPOSURE, exposure)
    time.sleep(0.5)
    for _ in range(10):
        vid.read()  # discard settle frames at the chosen exposure
    for _ in range(n):
        ret, frame = vid.read()
        if ret:
            frames.append(frame)
    vid.release()
    return frames


def _combine_stable(frames, stack_frames=1):
    """Reject torn/smeared outlier frames, then combine.

    A frame corrupted by camera starvation (e.g. a concurrent snr/stack job
    hogging the CPU) differs grossly from the per-pixel median of the burst.
    Reject frames whose RMS deviation from that median is a MAD outlier, then:
    a single-frame request returns the cleanest surviving *real* frame; a stack
    averages the survivors (noise ~ 1/sqrt(N)). Returns None for an empty list.
    """
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    arr = np.stack([f.astype(np.float32) for f in frames], axis=0)
    med = np.median(arr, axis=0)
    devs = np.sqrt(np.mean((arr - med) ** 2, axis=(1, 2, 3)))
    d_med = float(np.median(devs))
    mad = float(np.median(np.abs(devs - d_med)))
    # +1 keeps an all-clean burst intact (devs ~ noise, mad ~ 0); torn frames
    # (RMS dev tens of ADU) sit far above and get dropped.
    thresh = d_med + 3.0 * 1.4826 * mad + 1.0
    keep = devs <= thresh
    if not keep.any():
        keep[int(np.argmin(devs))] = True
    dropped = int((~keep).sum())
    if dropped:
        _loger.info("Rejected %d/%d torn frame(s) (RMS devs %s, thresh %.1f)",
                    dropped, len(frames), np.round(devs, 1).tolist(), thresh)
    kept = arr[keep]
    if stack_frames > 1:
        out = kept.mean(axis=0)
    else:
        surv_idx = np.where(keep)[0]              # cleanest surviving real frame
        out = arr[surv_idx[int(np.argmin(devs[keep]))]]
    return out.round().astype(frames[0].dtype)


def take_snapshot(test_path=None, light=True, out_path=None, scorer=None, stack_frames=1):
    """Serialize all camera access, then delegate to :func:`_take_snapshot`.

    The lock makes each snapshot atomic with respect to both the USB camera and
    the inside-light save/restore, so concurrent callers queue instead of
    corrupting each other (see ``_camera_lock``).

    light:        when False, the inside light is left untouched (never turned on)
                  — used by the `live` sky view so imaging isn't lit up.
    out_path:     write the chosen frame here instead of the shared scope_view.jpg
                  (so a dark sky frame never clobbers the lit safety snapshot).
    scorer:       exposure-scoring function for the sweep (defaults to the
                  lit-scene best_exposure_score; the sky view passes dark_sky_score).
    stack_frames: average this many frames at the chosen exposure (default 1) to
                  beat read/shot noise down on a dark sky view.
    """
    with _camera_lock:
        return _take_snapshot(test_path, light=light, out_path=out_path,
                              scorer=scorer, stack_frames=stack_frames)


def _take_snapshot(test_path=None, light=True, out_path=None, scorer=None, stack_frames=1):
    _loger.info("Starting camera snapshot (light=%s)", light)
    cfg = config.data()
    print (utils.set_install_dir())
    if scorer is None:
        scorer = best_exposure_score

    to_path = out_path or cfg["camera safety"]["scope_view"]

    if test_path is not None:
        shutil.copyfile(test_path, to_path)
        return True

    print("taking picture")

    no_image = cfg["camera safety"]["no_image"]
    shutil.copyfile(no_image, to_path)

    # Open camera with DirectShow backend (best for exposure on Windows)
     # Change 0 if you have multiple cameras

    # Optional: set resolution/FPS first (helps some cameras)
    #vid.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
    #vid.set(cv.CAP_PROP_FRAME_HEIGHT, 2160)
    # vid.set(cv.CAP_PROP_FPS, 30)
    # vid.set(cv.CAP_PROP_AUTOFOCUS, 1)
    #
    # # --- Set manual exposure here ---
    # exposure_value = -1  # Try values from -1 (bright) to -11 (dark/short)
    # vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)

    # Sometimes helps to also explicitly disable auto exposure (0.25 or 0.75 works on MSMF)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    # Light control is skipped entirely for a no-light capture (the `live` sky
    # view): we neither read nor change the inside light, so imaging stays dark.
    dev_map = None
    incoming_inside_light_status = None
    if light:
        dev_map = asyncio.run(ku.make_discovery_map())
        incoming_inside_light_status = super_user_commands.is_inside_light_on(dev_map)

    for attempt in range (1):
        if light:
            super_user_commands.turn_inside_light_on (dev_map)
            time.sleep(5)


        # Open and immediately release to flush any state left by another app (e.g. Windows Camera)
        _flush = cv.VideoCapture(0, cv.CAP_DSHOW)
        _flush.release()
        time.sleep(1.0)

        vid = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not vid.isOpened():
            _loger.error("Failed to open camera")
            return False

        # Disable auto-exposure before warm-up so the driver starts in manual mode.
        # For DirectShow (CAP_DSHOW): 1 = manual, 3 = auto.
        vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
        time.sleep(0.5)  # Give the driver time to switch to manual mode

        # Warm-up: read a few frames so the sensor is fully initialized
        for _ in range(10):
            vid.read()

        # Verify auto-exposure was disabled
        ae_val = vid.get(cv.CAP_PROP_AUTO_EXPOSURE)
        _loger.info("Auto-exposure value after set: %s", ae_val)

        # Exposure/gain plan differs by scene:
        #  - Lit indoor scene (safety snapshot): default gain, sweep -1..-11
        #    (CAP_DSHOW exposure is log2 seconds; -1 ~= 0.5s is already the longest
        #    of that set). This is a well-lit target, so gain is left alone.
        #  - No-light sky view (`live`): the frame is intrinsically dark and
        #    exposure alone bottoms out black (max ~0.5-1s on this webcam), so we
        #    crank sensor gain to the max and probe the LONGEST exposures (0..-6).
        #    dark_sky_score backs off if the boosted gain blows out highlights
        #    (moon / stray light), so max() still lands on a usable frame.
        if light:
            exposure_values = list(range(-1, -12, -1))
        else:
            # This webcam (DSHOW) caps exposure at -1 (~0.5s), so exposure alone
            # can't brighten a dark sky — the real levers are GAIN and BRIGHTNESS.
            # Probed ranges on this camera: gain 0..100 (default 0), brightness
            # -64..64 (default -64, i.e. pinned to its floor, which crushes the
            # night sky to black). Push both toward the top so faint sky glow /
            # clouds register, then sweep the longest exposures (-1..-6). Give the
            # driver a beat to apply gain/brightness before reading them back —
            # setting and immediately get()-ing returns the stale value.
            exposure_values = list(range(-1, -7, -1))
            vid.set(cv.CAP_PROP_GAIN, 100)       # max analog/digital gain
            vid.set(cv.CAP_PROP_BRIGHTNESS, 0)   # lift off the -64 floor to neutral
            time.sleep(0.5)
            for _ in range(5):
                vid.read()
            _loger.info("Dark-sky gain->%s brightness->%s",
                        vid.get(cv.CAP_PROP_GAIN), vid.get(cv.CAP_PROP_BRIGHTNESS))

        pictures = []
        scores = []
        for exposure_value in exposure_values:
            vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)
            actual = vid.get(cv.CAP_PROP_EXPOSURE)
            if actual != exposure_value:
                _loger.warning("Exposure set to %s but camera reports %s", exposure_value, actual)
            # Wait for the driver to apply the new exposure before discarding frames
            time.sleep(0.5)
            # Discard settle frames so the sensor adjusts to the new exposure
            for _ in range(10):
                vid.read()
            ret, frame = vid.read()
            if not ret:
                _loger.warning("Camera read failed at exposure %s", exposure_value)
                continue
            score = scorer(frame)
            _loger.info("Exposure: %s (actual: %s) Score: %.4f", exposure_value, actual, score)
            pictures.append(frame)
            scores.append(score)

        vid.release()

    def _restore_light():
        # Restore the inside light to its prior state (only if we changed it).
        if light:
            if not incoming_inside_light_status:
                super_user_commands.turn_inside_light_off(dev_map)
            else:
                super_user_commands.turn_inside_light_on(dev_map)

    if not scores:
        _loger.error("No valid frames captured at any exposure")
        _restore_light()
        return False

    best_score = max(scores)
    best_index = scores.index(best_score)
    best_exposure = exposure_values[best_index]
    _loger.info("Selected exposure index %s (exp %s) with score %.4f",
                best_index, best_exposure, best_score)

    # Robust final capture (see _combine_stable): a concurrent CPU-heavy job
    # (snr/stack) can starve the USB capture and hand back torn/smeared frames.
    # Grab a short burst at the chosen exposure and reject the torn ones, so
    # neither a `live` stack nor a roof/park safety snapshot is ever built on a
    # corrupt frame. The light stays on through the burst (needed for the lit
    # safety scene) and is restored afterwards; n >= 3 gives the outlier
    # rejection power even for a plain single-frame snapshot. stack_frames > 1
    # averages the survivors so faint sky detail lifts out of the noise.
    burst = _capture_burst(best_exposure, max(3, stack_frames), light)
    burst.append(pictures[best_index])  # the sweep already captured one good sample
    best_picture = _combine_stable(burst, stack_frames)

    _restore_light()

    if best_picture is None:
        _loger.error("No usable frames after rejecting torn captures")
        return False
    if stack_frames > 1:
        _loger.info("Stacked %d frames at exposure %s", stack_frames, best_exposure)

    cv.imwrite(to_path, best_picture)
    print(f"best score:  {best_score} of: {scores}")
    return True





if __name__ == '__main__':
    cfg = config.data()
    take_snapshot(None)
