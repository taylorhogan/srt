#!/usr/bin/env python3
import time
import statistics
from gpiozero import DistanceSensor
import pushover


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
current_roof_state = None  # None, "OPEN", or "CLOSED"
consecutive_count = 0


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


def send_notification(title, message):
    try:
        pushover.push_message(message, title=title, sound="intermission")
        print(f"Pushover sent: {title} - {message}")
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

while True:
    try:
        dist = get_distance_cm()

        if dist is None:
            print("No echo received (roof likely open)")
        else:
            print(f"Distance: {dist:.1f} cm")

        new_state = determine_roof_state(dist)
        print (f"current state {current_roof_state} new state {new_state} current state {current_roof_state} consecutive_count {consecutive_count}")
        print (str (consecutive_count))
        if new_state is None:
            # Ambiguous reading → don't change state, reset debounce
            consecutive_count = 0
        elif new_state == current_roof_state:
            # Same as before → reinforce
            consecutive_count += 1
        else:
            # Potential state change
            consecutive_count = 1
            current_roof_state = new_state  # we'll confirm after debounce

        # Only act after debounce threshold
        if consecutive_count >= DEBOUNCE_READINGS and new_state != current_roof_state:
            old_state = current_roof_state
            current_roof_state = new_state

            print(f"ROOF STATE CHANGED: {old_state or 'unknown'} → {current_roof_state}")

            if current_roof_state == "CLOSED":
                set_mount_power(False)
                send_notification("🔴 Roof CLOSED", "Telescope mount power has been turned OFF for safety.")
            elif current_roof_state == "OPEN":
                set_mount_power(True)
                send_notification("🟢 Roof OPEN", "Telescope mount power has been turned ON.")

        time.sleep(READ_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        break
    except Exception as e:
        print(f"Unexpected error: {e}")
        time.sleep(READ_INTERVAL_SECONDS)

# Final safety: turn mount off when script exits
#set_mount_power(False)