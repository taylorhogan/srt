
import time
from datetime import datetime
import instructions
import obs_calendar
import weather
import config
import os
import logging
import social_server
import utils



def set_state (state, dso = "Unknown"):
    cfg = config.data()
    cfg["Globals"]["Observatory State"] = state
    cfg["Globals"]["Imaging Tonight"] = dso

    print ("State: " + state)
    social_server.post_social_message("Scheduler State: " + state)


def waiting_for_boot ():
    set_state ("Waiting For Boot")
    while True:
        time.sleep(60)

def waiting_for_noon ():
    set_state("Waiting For Noon")
    instructions.calc_and_store_hours_above_horizon()

    now = datetime.now().time()

    while now.hour < 12:
        now = datetime.now().time()
        time.sleep(60)

    announce_plans_before_sunset()
    waiting_for_sunset()




def imaging ():
    set_state ("Imaging")
    time.sleep (5 * 60 * 60)
    waiting_for_sunrise()

def waiting_for_imaging ():
    set_state ("Waiting For Imaging")
    description, weather_ok = weather.get_current_weather(False)
    if weather_ok:
        imaging ()
    else:
        waiting_for_sunrise()

def waiting_for_sunset():

        set_state ("Waiting For Sunset")
        sunrise, sunset = weather.get_sunrise_sunset()
        now = datetime.now().time()

        while now < sunset:
            now = datetime.now().time()
            time.sleep(1)

        waiting_for_imaging ()


def waiting_for_sunrise():
    set_state("Waiting For Sunrise")
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()

    while now < sunrise:
        now = datetime.now().time()
        time.sleep(1)
    waiting_for_noon()

def start_state_machine():
    waiting_for_sunrise ()

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



def main ():
    cfg = config.data()
    path = utils.set_install_dir()
    path = os.path.join(path, 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger
    logger.info('Start Scheduler')
    utils.set_install_dir()
    try:
       waiting_for_sunrise()
    except:

        logger.info('Problem')
        logger.exception("Exception")
        social_server.get_mastodon_instance().status_post("Oops I had a problem with server")
        waiting_for_boot()

if __name__ == '__main__':
    main()


