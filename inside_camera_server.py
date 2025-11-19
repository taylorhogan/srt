
import cv2 as cv
import config
import shutil
import numpy as np


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

    vid = cv.VideoCapture(0,cv.CAP_DSHOW)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv.CAP_PROP_FPS, 30)
    vid.set(cv.CAP_PROP_EXPOSURE, -6)


    ret, frame = vid.read()
    vid.set(cv.CAP_PROP_EXPOSURE, -12)
    ret, frame = vid.read()
    cv.imshow("", frame)

    if ret:
        picture = []
        scores = []
        for gamma_val in np.arange(0.1, 4.5, 0.1):
            print (f"gamma: {gamma_val}")
            result = gamma_correction(frame, gamma=gamma_val)
            scores.append(best_exposure_score(result))
            picture.append(result)

        best_score = max(scores)
        best_index = scores.index(best_score)
        best_picture = picture[best_index]
        cv.imwrite(to_path, best_picture)
        print (f"best score:  {best_score} of: {scores}")
        return True

    else:
        print("no Image")
        return False



if __name__ == '__main__':
    cfg = config.data()
    take_snapshot(None)

