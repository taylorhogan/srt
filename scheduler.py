import enum
import inspect
import time
from datetime import datetime

import instructions
import obs_calendar
import social_server
import weather
import config
import os
import logging
import social_server


class ObsState(enum.Enum):
    WaitingForNoon = 0
    Noon = 1
    WaitingForSunset = 2
    Sunset = 3
    WaitingForSunrise = 4
    Sunrise = 5
    DetermineWillImage = 6
    RunningActiveSequence = 7
    RunningTestSequence = 8


def simple_machine():

    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()


    while now.hour < 12:
        now = datetime.now().time()
        time.sleep(10)

    announce_plans_before_sunset()


def determine_state() -> ObsState:
    now = datetime.now()
    if now.hour < 12:
        return ObsState.WaitingForNoon
    sunrise, sunset = weather.get_sunrise_sunset()
    if now.hour < sunset.hour:
        return ObsState.WaitingForSunrise
    return ObsState.WaitingForSunset


def wait_for_sunrise():
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
        print("weather ok")
    else:
        print("weather not ok")
        wait_for_sunrise()


def waiting_for_sunset():
    print(inspect.stack()[0][3])
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()
    print(sunrise)
    print(sunset)
    print(now)

    while now < sunset:
        now = datetime.now().time()
        time.sleep(1)
    print("sunset")
    determine_will_image()


def do_state(state) -> ObsState:
    return ObsState.WaitingForSunrise


def announce_plans_before_sunset():
    description, weather_ok = weather.get_current_weather(False)
    best_instruction = instructions.get_dso_object_tonight()
    dso = best_instruction["dso"]
    requestor = best_instruction["requestor"]

    if weather_ok:
        social_server.post_social_message("Will image " + dso + " requested by " + requestor + " tonight")
        obs_calendar.set_today_stat('image', dso)
    else:
        social_server.post_social_message(
            "Will NOT image " + dso + " requested by " + requestor + " tonight because of weather")
        obs_calendar.set_today_stat('weather', dso)



if __name__ == '__main__':
    cfg = config.data()
    path = os.path.join(cfg["Install"], 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger
    logger.info('Start Scheduler')
    try:
        simple_machine()
    except:
        logger.info('Problem')
        logger.exception("Exception")
        social_server.get_mastodon_instance().status_post("Oops I had a problem with server")


