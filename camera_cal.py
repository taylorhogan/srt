import cv2

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera - Manual Exposure', frame)

    # Press 'q' to quit, or '+' / '-' to adjust exposure live
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        exposure_value = max(exposure_value - 1, -13)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        print(f"Exposure increased to {exposure_value}")
    elif key == ord('-'):
        exposure_value = min(exposure_value + 1, -1)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        print(f"Exposure decreased to {exposure_value}")

cap.release()
cv2.destroyAllWindows()
