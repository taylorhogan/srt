#!/usr/bin/env python3
import statistics
from gpiozero import DistanceSensor
import os,sys
import requests
import time
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils import pushover
from configs import config

cfg = config.data()

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
ROOF_CLOSED_DISTANCE_CM = 70  # < 12 inches ≈ 30 cm → roof closed
ROOF_OPEN_MIN_CM = 100  # if we can't see anything closer than this → roof open
SAMPLES_PER_READING = 5  # take 5 measurements and median filter
READ_INTERVAL_SECONDS = 2  # how often we check the sensor
DEBOUNCE_READINGS = 3  # need this many consecutive same state before we accept change

# --- Temperature compensation ---
DEFAULT_SPEED_OF_SOUND = 343.21  # m/s at ~20°C (gpiozero default assumption)
TEMPERATURE_CACHE_SECONDS = 300  # refresh temperature every 5 minutes

# ================================================================

# Temperature caching
_cached_temperature = None
_temperature_last_fetched = 0


def get_current_temperature():
    """Fetch current temperature from Open-Meteo API. Returns temperature in Celsius or None on failure."""
    global _cached_temperature, _temperature_last_fetched

    current_time = time.time()
    if _cached_temperature is not None and (current_time - _temperature_last_fetched) < TEMPERATURE_CACHE_SECONDS:
        return _cached_temperature

    try:
        lat = cfg["location"]["latitude"]
        lon = cfg["location"]["longitude"]
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        _cached_temperature = data["current_weather"]["temperature"]
        _temperature_last_fetched = current_time
        print(f"Temperature updated: {_cached_temperature}°C")
        return _cached_temperature
    except Exception as e:
        print(f"Failed to fetch temperature: {e}")
        return _cached_temperature  # return cached value if available


def calculate_speed_of_sound(temperature_celsius):
    """Calculate speed of sound in air based on temperature.

    Formula: v = 331.3 + 0.606 * T (m/s)
    where T is temperature in Celsius.
    """
    if temperature_celsius is None:
        return DEFAULT_SPEED_OF_SOUND
    return 331.3 + 0.606 * temperature_celsius


def get_temperature_correction_factor(temperature_celsius):
    """Get the correction factor to apply to distance measurements.

    gpiozero assumes speed of sound is ~343.21 m/s (20°C).
    We correct by: actual_speed / assumed_speed
    """
    actual_speed = calculate_speed_of_sound(temperature_celsius)
    return actual_speed / DEFAULT_SPEED_OF_SOUND

# Initialize components

echo = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN, max_distance=4)  # 4 meters max



# State tracking



def get_distance_cm(temperature_celsius=None):
    """Return median distance in cm from several samples, with temperature compensation.

    Args:
        temperature_celsius: Current temperature for speed of sound correction.
                            If None, will attempt to fetch from weather API.

    Returns:
        Temperature-corrected distance in cm, or None on failure.
    """
    if temperature_celsius is None:
        temperature_celsius = get_current_temperature()

    correction_factor = get_temperature_correction_factor(temperature_celsius)

    distances = []
    for _ in range(SAMPLES_PER_READING):
        try:
            # gpiozero sometimes throws ValueError on bad readings
            dist = echo.distance * 100  # meters → cm
            if dist < 400:  # sanity check, HC-SR04 max ~400cm
                # Apply temperature correction
                corrected_dist = dist * correction_factor
                distances.append(corrected_dist)
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




def roof_state ():

    last_state = None
    current_roof_state = None
    consecutive_count = 0
    while True:
        try:
            dist = get_distance_cm()

            temp = get_current_temperature()
            if dist is None:
                print(f"No echo received (temp: {temp}°C)" if temp else "No echo received")
            else:
                correction = get_temperature_correction_factor(temp)
                print(f"Distance: {dist:.1f} cm (temp: {temp}°C, correction: {correction:.4f})")

            test_state = determine_roof_state(dist)


            if test_state is None:
                # Ambiguous reading → don't change state, reset debounce
                consecutive_count = 0
            elif test_state == last_state:
                # Same as before → reinforce
                consecutive_count += 1
                consecutive_count = min (consecutive_count, DEBOUNCE_READINGS+1)

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
    roof_state()
