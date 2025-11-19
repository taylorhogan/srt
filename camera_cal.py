import cv2
import numpy as np


def best_exposure_score(img):
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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

# Open camera with DirectShow backend (best for exposure on Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change 0 if you have multiple cameras

# Optional: set resolution/FPS first (helps some cameras)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- Set manual exposure here ---
exposure_value = -6  # Try values from -1 (bright) to -11 (dark/short)
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

# Sometimes helps to also explicitly disable auto exposure (0.25 or 0.75 works on MSMF)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

print(f"Exposure set to: {cap.get(cv2.CAP_PROP_EXPOSURE)}")  # May return -1 if not supported
pictures=[]
scores = []
for exposure_value in range(-1, -13, -1):
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    #cv2.imshow('Camera - Manual Exposure', frame)
    score = best_exposure_score(frame)
    pictures.append(frame)
    scores.append(score)

best_score = max(scores)
best_index = scores.index(best_score)
best_picture = pictures[best_index]
cv2.imshow(str(best_score), best_picture)

print (f"best score:  {best_score} of: {scores}")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


