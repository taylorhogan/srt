#!/usr/bin/env python3
import time
import statistics
from gpiozero import DistanceSensor
import pushover
import cv2 as cv
import os
import time
import math
import cv2 as cv
import inside_camera_server
import config
import pushover
import inside_camera_server


# ========================= CONFIGURATION =========================

# --- Ultrasonic sensor pins (BCM numbering) ---
TRIGGER_PIN = 20
ECHO_PIN = 21

# --- Kasa Smart Plug ---
PLUG_IP = "192.168.1.XXX"  # Change to your plug's IP (highly recommended to use static IP)

# --- Pushover credentials ---
PUSHOVER_USER = "your_user_key_here"
PUSHOVER_TOKEN = "your_api_token_here"

# --- Thresholds ---
ROOF_CLOSED_DISTANCE_CM = 30  # < 12 inches ≈ 30 cm → roof closed
ROOF_OPEN_MIN_CM = 100  # if we can't see anything closer than this → roof open
SAMPLES_PER_READING = 5  # take 5 measurements and median filter
READ_INTERVAL_SECONDS = 2  # how often we check the sensor
DEBOUNCE_READINGS = 3  # need this many consecutive same state before we accept change

# ================================================================

# Initialize components

echo = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN, max_distance=4)  # 4 meters max



# State tracking



def get_distance_cm():
    """Return median distance in cm from several samples, or None on failure"""
    distances = []
    for _ in range(SAMPLES_PER_READING):
        try:
            # gpiozero sometimes throws ValueError on bad readings
            dist = echo.distance * 100  # meters → cm
            if dist < 400:  # sanity check, HC-SR04 max ~400cm
                distances.append(dist)
        except Exception:
            pass
        time.sleep(0.06)

    if len(distances) == 0:
        return None
    return statistics.median(distances)


def send_notification(message):
    try:
        pushover.push_message(message)
        print(f"Pushover sent:{message}")
    except Exception as e:
        print(f"Failed to send Pushover: {e}")


def set_mount_power(on: bool):
    try:
        if on:
            #plug.turn_on()
            print("Mount power: ON")
        else:
            #plug.turn_off()
            print("Mount power: OFF")
    except Exception as e:
        print(f"Kasa plug error: {e}")


def determine_roof_state(distance_cm):
    if distance_cm is None:
        # No echo = roof fully open (or sensor blocked, but in observatory this means open)
        return "OPEN"
    elif distance_cm < ROOF_CLOSED_DISTANCE_CM:
        return "CLOSED"
    elif distance_cm > ROOF_OPEN_MIN_CM:
        return "OPEN"
    else:
        # In between → ambiguous, we keep previous known state
        return None


print("Observatory roof monitor started...")
print("Waiting for first valid reading...")


def count_stars ():

    cfg = config.data()

    print("taking picture")


    to_path = cfg["camera safety"]["scope_view"]


    # Open camera with DirectShow backend (best for exposure on Windows)
    vid = cv.VideoCapture(0)  # Change 0 if you have multiple cameras

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
            return False
        vid.set(cv.CAP_PROP_EXPOSURE, exposure_value)

        score = inside_camera_server.best_exposure_score(frame)
        print(f"Exposure: {exposure_value} Score: {score}")
        pictures.append(frame)
        scores.append(score)



    best_score = max(scores)
    best_index = scores.index(best_score)
    best_picture = pictures[best_index]


    print(f"best score:  {best_score} of: {scores}")
    return True

def roof_state ():

    last_state = None
    current_roof_state = None
    consecutive_count = 0
    while True:
        try:
            dist = get_distance_cm()

            if dist is None:
                print("No echo received")
            else:
                print(f"Distance: {dist:.1f} cm")

            test_state = determine_roof_state(dist)


            if test_state is None:
                # Ambiguous reading → don't change state, reset debounce
                consecutive_count = 0
            elif test_state == last_state:
                # Same as before → reinforce
                consecutive_count += 1
            else:
                # Potential state change
                consecutive_count = 1
            last_state = test_state
            print(
                f"current state {current_roof_state} last state {last_state} current state {current_roof_state} consecutive_count {consecutive_count}")
            # Only act after debounce threshold
            if consecutive_count >= DEBOUNCE_READINGS and last_state != current_roof_state:

                current_roof_state = last_state



                if current_roof_state == "CLOSED":
                    set_mount_power(False)
                    send_notification("🔴 Roof CLOSED Telescope mount power has been turned OFF for safety.")
                elif current_roof_state == "OPEN":
                    set_mount_power(True)
                    send_notification("🟢 Roof OPEN  Telescope mount power has been turned ON.")

            time.sleep(READ_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(READ_INTERVAL_SECONDS)

if __name__ == "__main__":
    #roof_state()
    count_stars()