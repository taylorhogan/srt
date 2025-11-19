
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

    vid = cv.VideoCapture(0)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    vid.set(cv.CAP_PROP_EXPOSURE, 12)

    ret, frame = vid.read()
    if ret:
        img_src = frame
        cv.imwrite(to_path, img_src)
        score = best_exposure_score(img_src)
        print (f"best exposure score: {score}")
        return True
    else:
        print("no Image")
        return False



if __name__ == '__main__':
    cfg = config.data()
    take_snapshot(None)

