import asyncio
import json
import logging
import os
import time
from datetime import datetime

import astro_dso_visibility
import config
import instructions
import obs_calendar
import social_server
import utils
import weather

client = None
observatory_state = {
    "state": "Unknown",
    "dso": "Unknown",
    "will image tonight": "Unknown"
}


def message_handling(client, userdata, msg):
    message = msg.payload.decode("utf-8")
    if msg.topic == utils.topic_to_sched:
        print("incoming message", msg)
    json_payload = json.dumps(observatory_state)

    topic = "iris/from_sched"
    cfg = config.data()
    logger = cfg["logger"]["logging"]
    logger.info(f"Send `{json_payload}` to topic `{topic}`")

    result = client.publish(topic, json_payload)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")


def set_state(state, dso=None, will_image_tonight=None):
    global observatory_state
    observatory_state["state"] = state
    if dso is not None:
        observatory_state["dso"] = dso
    if will_image_tonight is not None:
        observatory_state["will image tonight"] = will_image_tonight

    print("State: " + state)
    # social_server.post_social_message("Scheduler State: " + state)


def waiting_for_boot():
    set_state("Waiting For Boot")
    while True:
        time.sleep(60)


def waiting_for_noon():
    set_state("Waiting For Noon")
    instructions.calc_and_store_hours_above_horizon()

    now = datetime.now().time()

    while now.hour < 12:
        now = datetime.now().time()
        asyncio.run(wait_a_bit())

    instructions.calc_and_store_hours_above_horizon()
    image, dso = announce_plans_before_sunset()

    set_state(observatory_state["state"], dso)
    if image:
        waiting_for_sunset()
    else:
        waiting_for_sunrise()


def imaging():
    set_state("Imaging")

    asyncio.run(wait_a_bit())
    waiting_for_sunrise()


def wait_for_tomorrow():
    now = datetime.now().time()

    while now.hour > 0:
        now = datetime.now().time()
        asyncio.run(wait_a_bit())


def waiting_for_imaging():
    set_state("Waiting For Imaging")
    description, weather_ok = weather.get_current_weather(False)
    if weather_ok:
        imaging()
    else:
        social_server.post_social_message("Will NOT image tonight because of weather")
        waiting_for_sunrise()


async def wait_a_bit():
    await asyncio.sleep(10)


def waiting_for_sunset():
    set_state("Waiting For Sunset")
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()

    while now < sunset:
        now = datetime.now().time()
        asyncio.run(wait_a_bit())

    waiting_for_imaging()


def waiting_for_sunrise():
    wait_for_tomorrow()
    set_state("Waiting For Sunrise")
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now().time()

    while now < sunrise:
        now = datetime.now().time()
        asyncio.run(wait_a_bit())

    waiting_for_noon()


def announce_plans_before_sunset():
    global observatory_state

    weather_ok = social_server.tonight_cmd("tonight dso", 1, "", "")

    best_instruction = instructions.get_dso_object_tonight()
    dso = best_instruction["dso"]
    requestor = best_instruction["requestor"]

    if weather_ok:

        social_server.post_social_message("Will image " + dso + " requested by " + requestor + " tonight")
        obs_calendar.set_today_stat('image', dso)
        set_state(observatory_state["state"], dso, True)
        return True, dso

    else:
        social_server.post_social_message(
            "Will NOT image " + dso + " requested by " + requestor + " tonight because of weather")
        obs_calendar.set_today_stat('weather', dso)
        set_state(observatory_state["state"], dso, False)
        return False, dso


def main():
    print("Starting Scheduler Server")
    cfg = config.data()
    path = utils.set_install_dir()
    path = os.path.join(path, 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger
    logger.info('Start Scheduler')
    utils.set_install_dir()
    client = utils.connect_mqtt()
    client.subscribe(utils.topic_to_sched)
    client.on_message = message_handling
    client.loop_start()

    try:
        while True:
            asyncio.run(wait_a_bit())
            waiting_for_noon()
    except:

        logger.info('Problem')
        logger.exception("Exception")
        social_server.get_mastodon_instance().status_post("Oops I had a problem with Scheduler server")
        waiting_for_boot()


if __name__ == '__main__':
    main()
