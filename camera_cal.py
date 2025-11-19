import cv2
import inside_camera_server as ics

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
    score = ics.best_exposure_score(frame)
    pictures.append(frame)
    scores.append(score)

best_score = max(scores)
best_index = scores.index(best_score)
best_picture = pictures[best_index]
cv2.imshow(str(best_score), best_picture)
print (f"best score:  {best_score} of: {scores}")



