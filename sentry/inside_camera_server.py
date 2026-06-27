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
_camera_lock = threading.Lock()


def take_snapshot(test_path=None):
    """Serialize all camera access, then delegate to :func:`_take_snapshot`.

    The lock makes each snapshot atomic with respect to both the USB camera and
    the inside-light save/restore, so concurrent callers queue instead of
    corrupting each other (see ``_camera_lock``).
    """
    with _camera_lock:
        return _take_snapshot(test_path)


def _take_snapshot(test_path=None):
    _loger.info("Starting camera snapshot")
    cfg = config.data()
    print (utils.set_install_dir())

    if test_path is not None:
        to_path = cfg["camera safety"]["scope_view"]
        shutil.copyfile(test_path, to_path)
        return True

    print("taking picture")

    no_image = cfg["camera safety"]["no_image"]
    to_path = cfg["camera safety"]["scope_view"]
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

    dev_map = asyncio.run(ku.make_discovery_map())
    incoming_inside_light_status = super_user_commands.is_inside_light_on(dev_map)

    for attempt in range (1):
        if attempt == 0:
            super_user_commands.turn_inside_light_on (dev_map)
            time.sleep(5)

        else:
            super_user_commands.turn_inside_light_off(dev_map)


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

        pictures = []
        scores = []
        for exposure_value in range(-1, -12, -1):
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
            score = best_exposure_score(frame)
            _loger.info("Exposure: %s (actual: %s) Score: %.4f", exposure_value, actual, score)
            pictures.append(frame)
            scores.append(score)

        vid.release()

    if not incoming_inside_light_status:
        super_user_commands.turn_inside_light_off(dev_map)
    else:
        super_user_commands.turn_inside_light_on(dev_map)

    if not scores:
        _loger.error("No valid frames captured at any exposure")
        return False

    best_score = max(scores)
    best_index = scores.index(best_score)
    best_picture = pictures[best_index]
    _loger.info("Selected exposure index %s with score %.4f", best_index, best_score)



    picture = []
    scores = []
    # for gamma_val in np.arange(0.1, 4.5, 0.1):
    #     print(f"gamma: {gamma_val}")
    #     result = gamma_correction(best_picture, gamma=gamma_val)
    #     scores.append(best_exposure_score(result))
    #     picture.append(result)
    #
    # best_score = max(scores)
    # best_index = scores.index(best_score)
    # best_picture = picture[best_index]
    cv.imwrite(to_path, best_picture)
    #cv.imshow('Image Window Title', best_picture)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    print(f"best score:  {best_score} of: {scores}")
    return True





if __name__ == '__main__':
    cfg = config.data()
    take_snapshot(None)
