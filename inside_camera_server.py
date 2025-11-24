import shutil

import cv2 as cv
import numpy as np

import config


def best_exposure_score(img):
    if len(img.shape) == 3:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)  # Luminance channel (0-255)
    else:
        L = img.astype(np.float32)

    L_norm = L / 255.0

    # Count badly clipped pixels
    under = np.sum(L < 8) / L.size  # < ~3%
    over = np.sum(L > 248) / L.size  # > ~97%

    # Luminance mean (in log space is better, but linear works fine)
    mean_lum = np.mean(L_norm)

    # Standard deviation of luminance (more contrast = better)
    std_lum = np.std(L_norm)

    # Final score
    score = std_lum * (1 - 8 * (under + over)) * np.exp(-15 * (mean_lum - 0.5) ** 2)
    return score


def gamma_correction(img, gamma=1.0):
    # Build a lookup table (fastest method)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)


def take_snapshot(test_path=None):
    cfg = config.data()
    if test_path is not None:
        to_path = cfg["camera safety"]["scope_view"]
        shutil.copyfile(test_path, to_path)
        return True

    print("taking picture")

    no_image = cfg["camera safety"]["no_image"]
    to_path = cfg["camera safety"]["scope_view"]
    shutil.copyfile(no_image, to_path)

    # Open camera with DirectShow backend (best for exposure on Windows)
    vid = cv.VideoCapture(0, cv.CAP_DSHOW)  # Change 0 if you have multiple cameras

    # Optional: set resolution/FPS first (helps some cameras)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv.CAP_PROP_FPS, 30)

    # --- Set manual exposure here ---
    exposure_value = -6  # Try values from -1 (bright) to -11 (dark/short)
    vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)

    # Sometimes helps to also explicitly disable auto exposure (0.25 or 0.75 works on MSMF)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    print(f"Exposure set to: {vid.get(cv.CAP_PROP_EXPOSURE)}")  # May return -1 if not supported
    pictures = []
    scores = []
    for exposure_value in range(-1, -11, -1):
        ret, frame = vid.read()
        if not ret:
            return False
        vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)
        # cv2.imshow('Camera - Manual Exposure', frame)
        score = best_exposure_score(frame)
        print(f"Exposure: {exposure_value} Score: {score}")
        pictures.append(frame)
        scores.append(score)

    best_score = max(scores)
    best_index = scores.index(best_score)
    best_picture = pictures[best_index]



    picture = []
    scores = []
    for gamma_val in np.arange(0.1, 4.5, 0.1):
        print(f"gamma: {gamma_val}")
        result = gamma_correction(best_picture, gamma=gamma_val)
        scores.append(best_exposure_score(result))
        picture.append(result)

    best_score = max(scores)
    best_index = scores.index(best_score)
    best_picture = picture[best_index]
    cv.imwrite(to_path, best_picture)
    cv.imshow('Image Window Title', best_picture)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print(f"best score:  {best_score} of: {scores}")
    return True




if __name__ == '__main__':
    cfg = config.data()
    take_snapshot(None)
