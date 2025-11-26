# modern_hcsr04.py – works on Bookworm 64-bit without RPi.GPIO
from gpiozero import DistanceSensor
from time import sleep

sensor = DistanceSensor(echo=21, trigger=20)  # pins in BCM

while True:
    cm = sensor.distance * 100
    inches = cm / 2.54
    print(f"Distance: {inches:.2f} inches")

    sleep(0.1)
