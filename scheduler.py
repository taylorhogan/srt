import enum
from datetime import datetime, timedelta

import social_server
import weather
import time
import inspect
import obs_calendar

class ObsState(enum.Enum):
    WaitingForNoon = 0
    Noon = 1
    WaitingForSunset= 2
    Sunset = 3
    WaitingForSunrise= 4
    Sunrise =5
    DetermineWillImage = 6
    RunningActiveSequence=7
    RunningTestSequence=8


def simple_machine ():
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()
    print(sunrise)
    print(sunset)
    print(now)

    while now.hour < 12:
        now = datetime.now().time()
        time.sleep(10)
    print("noon")
    announce_plans_before_sunset()


def determine_state ()->ObsState:
    now = datetime.now()
    if now.hour < 12:
        return ObsState.WaitingForNoon
    sunrise, sunset = weather.get_sunrise_sunset()
    if now.hour < sunset.hour:
        return ObsState.WaitingForSunrise
    return ObsState.WaitingForSunset


def wait_for_sunrise ():
    print(inspect.stack()[0][3])
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()
    print(sunrise)
    print(sunset)
    print(now)

    while now < sunrise:
        now = datetime.now().time()
        time.sleep(1)
    print("sunrise")
    determine_will_image()

def determine_will_image():
    print(inspect.stack()[0][3])
    description, weather_ok = weather.get_current_weather(False)
    if weather_ok:
        print ("weather ok")
    else:
        print ("weather not ok")
        wait_for_sunrise ()

def waiting_for_sunset ():
    print(inspect.stack()[0][3])
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()
    print (sunrise)
    print (sunset)
    print(now)

    while now < sunset:
        now = datetime.now().time()
        time.sleep(1)
    print ("sunset")
    determine_will_image()

def do_state (state)->ObsState:
    return ObsState.WaitingForSunrise

def announce_plans_before_sunset ():
    description, weather_ok = weather.get_current_weather(False)
    if weather_ok:
        social_server.post_social_message("Will image tonight")
        obs_calendar.set_today_stat('image', 'research')
    else:
        social_server.post_social_message ("Will NOT image tonight")
        obs_calendar.set_today_stat('weather', '')


if __name__ == '__main__':
  simple_machine()

