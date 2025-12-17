
import cv2 as cv
import inside_camera_server
import config
import pushover
import inside_camera_server

def count_stars ():

    cfg = config.data()

    print("taking picture")


    to_path = cfg["camera safety"]["scope_view"]


    # Open camera with DirectShow backend (best for exposure on Windows)
    #vid = cv.VideoCapture(0)  # Change 0 if you have multiple cameras
    vid = cv.VideoCapture(0, cv.CAP_DSHOW)
    # Optional: set resolution/FPS first (helps some cameras)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 2160)
    vid.set(cv.CAP_PROP_FPS, 30)
    vid.set(cv.CAP_PROP_AUTOFOCUS, 1)

    # --- Set manual exposure here ---
    exposure_value = -6  # Try values from -1 (bright) to -11 (dark/short)
    vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)

    # Sometimes helps to also explicitly disable auto exposure (0.25 or 0.75 works on MSMF)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)






    pictures = []
    scores = []
    for exposure_value in range(-1, -11, -1):
        ret, frame = vid.read()
        if not ret:
            print("failed to read frame")
            return False
        vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)

        score = inside_camera_server.best_exposure_score(frame)
        print(f"Exposure: {exposure_value} Score: {score}")
        pictures.append(frame)
        scores.append(score)



    best_score = max(scores)
    best_index = scores.index(best_score)
    best_picture = pictures[best_index]
    pushover.push_message_with_picture("picture", best_picture)


    print(f"best score:  {best_score} of: {scores}")
    return True