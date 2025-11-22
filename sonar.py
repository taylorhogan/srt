# modern_hcsr04.py – works on Bookworm 64-bit without RPi.GPIO
from gpiozero import DistanceSensor
from time import sleep

sensor = DistanceSensor(echo=24, trigger=23, max_distance=4)  # pins in BCM

while True:
    print(f"Distance: {sensor.distance * 100:.1f} cm")
    sleep(0.1)
