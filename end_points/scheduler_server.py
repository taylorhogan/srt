import asyncio
import json
import logging
import os,sys
import time
from datetime import datetime


if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from iris_astronomy import astro_dso_visibility, obs_calendar, weather
from configs import config
from control import instructions
from cmd_processing import social_server
from utils import utils, pushover
from iris_astronomy.astro_dso_visibility import show_plots
from cmd_processing import super_user_commands

CFG = config.data()
LOGGER = utils.set_logger()
CFG["logger"]["logging"] = LOGGER

client = None
observatory_state = {
    "state": "Unknown",
    "dso": "Unknown",
    "will image tonight": "Unknown"
}


def message_handling(client, userdata, msg):
    # Bug 2 fixed: only publish a response when the message is on the expected topic
    if msg.topic == utils.topic_to_sched:
        print("incoming message", msg)
        json_payload = json.dumps(observatory_state)

        topic = "iris/from_sched"
        LOGGER.info(f"Send `{json_payload}` to topic `{topic}`")

        result = client.publish(topic, json_payload)
        status = result[0]
        if status == 0:
            print(f"Send `{json_payload}` to topic `{topic}`")
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
    # Bug 7 fixed: only calculate after noon, not redundantly before the wait loop
    now = datetime.now().time()

    while now.hour < 12:
        now = datetime.now().time()
        asyncio.run(wait_a_bit())

    instructions.calc_and_store_hours_above_horizon()
    print("announcing plans before sunset")
    image, dso = announce_plans_before_sunset()
    print("announced")

    # Bug 8 fixed: removed redundant set_state call; announce_plans_before_sunset already sets it
    if image:
        print("imaging, waiting for sunset")
        waiting_for_sunset()
    else:
        print("not imaging, waiting for sunrise")
        waiting_for_sunrise()


def imaging():
    set_state("Imaging")
    asyncio.run(wait_a_bit())
    # Bug 3 fixed: uncommented the actual imaging command
    super_user_commands.image_cmd("", "iris")
    waiting_for_sunrise()


def wait_for_tomorrow():
    # Bug 4 fixed: wait until midnight using a target datetime comparison instead
    # of checking hour > 0, which would loop for ~24h if called after midnight.
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now >= midnight:
        from datetime import timedelta
        midnight = midnight + timedelta(days=1)

    while datetime.now() < midnight:
        asyncio.run(wait_a_bit())


def waiting_for_imaging():
    set_state("Waiting For Imaging")
    best_instruction = instructions.get_dso_object_tonight()
    dso = best_instruction["dso"]
    obj = astro_dso_visibility.is_a_dso_object(dso)
    # Bug 1 fixed: show_plots returns 4 values; capture the bool from the 4th
    _, _, _, weather_ok = show_plots(obj)

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
    now = datetime.now(sunset.tzinfo)
    pushover.push_message("waiting for sunset at " + str(sunset))

    while now < sunset:
        now = datetime.now(sunset.tzinfo)
        asyncio.run(wait_a_bit())

    waiting_for_imaging()


def waiting_for_sunrise():
    wait_for_tomorrow()
    set_state("Waiting For Sunrise")
    sunrise, sunset = weather.get_sunrise_sunset()
    now = datetime.now(sunrise.tzinfo)

    while now < sunrise:
        now = datetime.now(sunrise.tzinfo)
        asyncio.run(wait_a_bit())

    waiting_for_noon()


def announce_plans_before_sunset():
    global observatory_state

    weather_ok = social_server.tonight_cmd(["me", "tonight"], 1, "", "")
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _instructions_path = os.path.join(_project_root, CFG["location"]["instructions"])
    best_name, best_start, best_good_hours = astro_dso_visibility.best_object_tonight(_instructions_path)
    best_instruction = instructions.get_dso_object_tonight()
    dso = best_name
    requestor = "taylor"
    social_server.post_social_message("Tonight's imaging grid", image=os.path.join(_project_root, CFG["location"]["image_grid"]))
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
    super_user_commands.safe_cmd(None, None)
    super_user_commands.imaging_state(False)
    LOGGER.info('Start Scheduler')
    utils.set_install_dir()
    client = utils.connect_mqtt()
    client.subscribe(utils.topic_to_sched)
    client.on_message = message_handling
    client.loop_start()

    # Bug 5 fixed: bare except -> except Exception
    try:
        asyncio.run(wait_a_bit())
        waiting_for_noon()
    except Exception:
        LOGGER.info('Problem')
        LOGGER.exception("Exception")
        # Bug 6 fixed: wrap Mastodon post in its own try so a Mastodon failure
        # doesn't hide the original exception already logged above
        try:
            social_server.get_mastodon_instance().status_post("Oops I had a problem with Scheduler server")
        except Exception:
            LOGGER.exception("Also failed to post exception notice to Mastodon")
        waiting_for_boot()


if __name__ == '__main__':
    main()
