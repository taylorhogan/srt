import asyncio
from pathlib import Path
import datetime
import logging
import os
from datetime import date
import json
import time
import sys
from typing import Any, Optional
from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener
from mastodon.streaming import CallbackStreamListener
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from iris_astronomy import astro_dso_visibility, obs_calendar
from configs import config
from end_points import end
from fits_processing import fitstojpg, fitsfwhm
from control import instructions
from cmd_processing import super_user_commands as su
from utils.utils import topic_to_sched
from utils import utils




_json_payload = None


def get_dso_object_name(words: list[str], index: int) -> Optional[str]:
    if len(words) > index + 1:
        dso = words[index + 1]
    else:
        return None
    if len(words) > index + 2:
        dso = dso + " " + words[index + 2]
    return dso


def image_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
    example image m 13
    """
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = astro_dso_visibility.is_a_dso_object(dso_name)
        if object is not None:

            now = datetime.datetime.now()
            instructions.add_dso_object_instruction(dso_name, "", account)
            post_social_message(dso_name + " Added to list of objects to image\n")
        else:
            post_social_message(dso_name + " Not a known object\n")


def best_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    """
       example best m 13
    """
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = astro_dso_visibility.is_a_dso_object(dso_name)
        if object is not None:
            best_date, best_time, max_altitude = astro_dso_visibility.best_day_for_dso(object)
            if best_date is not None:
                formatted_date = best_date.strftime("%Y-%m-%d")
                formatted_air_mass = "{:.2f}".format((astro_dso_visibility.air_mass(max_altitude)))
                post_social_message(dso_name + " is above horizon for " + str(best_time) + " on " + formatted_date  + " with air mass of "+formatted_air_mass )
                hours = best_time.seconds / 3600
                if hours > 3:
                    image_cmd(words, index, m, account)

            else:
                post_social_message(dso_name + " is never above horizon")
        else:
            post_social_message(dso_name + " Not a known object\n")


def tonight_cmd(words: list[str], index: int, m: Mastodon, account: str) -> bool:
    print("in tonight cmd", words, index)
    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    instructions_path = os.path.join(_project_root, cfg["location"]["instructions"])

    best_name, best_start, best_good_hours = astro_dso_visibility.best_object_tonight(instructions_path)
    image_grid_path = os.path.join(_project_root, cfg["location"]["image_grid"])
    post_social_message(
        f"Tonight's best object: {best_name} ({best_good_hours}h good imaging)",
        image_grid_path
    )

    if best_name:
        obj = astro_dso_visibility.is_a_dso_object(best_name)
        if obj is not None:
            horizon, image, sky, weather_ok = astro_dso_visibility.show_plots(obj)

            if horizon is not None:
                post_social_message(f"{best_name} — altitude & conditions", horizon)
            if sky is not None:
                post_social_message(f"{best_name} — sky chart", sky)
            if weather_ok:
                post_social_message("Weather ok tonight")
            else:
                post_social_message("Weather not ok tonight")
            return weather_ok
        else:
            post_social_message(f"{best_name} not a known object")
    return False



def version_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    # Observatory State
    cfg = config.data()

    reply = "Version: " + cfg["version"]["date"] + "\n"
    reply += "Observatory Status: " + cfg["Globals"]["Observatory State"]
    post_social_message(reply)

def wait_for_mqtt_message(client: Any, userdata: Any, msg: Any) -> None:
    global _json_payload
    print ("setting payload to ")
    _json_payload = json.loads(msg.payload.decode("utf-8"))
    _json_payload = json.dumps(_json_payload)
    logger = logging.getLogger(__name__)
    logger.info("Received message: " + _json_payload)
    print("to ", _json_payload)




def status_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    # Observatory State
    cfg = config.data()

    global _json_payload
    mqtt_client = cfg["globals"]["mqtt_client"]
    timeout = 60  # Timeout in seconds
    start_time = time.time()
    _json_payload = None
    print ("setting state to None")
    end.determine_roof_state_visually(account)
    print ("asking for status")
    mqtt_client.publish(topic_to_sched, "status?")
    while _json_payload is None and (time.time() - start_time < timeout) :
        asyncio.run(wait_a_bit())
    if _json_payload is None:
        post_social_message("Status could not be determined")
    else:
        post_social_message(_json_payload)




def db_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    instructions.create_instructions_table()




def calendar_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    # Observatory State
    cfg = config.data()
    today = date.today()

    obs_calendar.print_month(today.year, today.month, cfg)
    post_social_message("", "cal.png")


def help_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    reply = "Available commands are\n"
    for word in keywords:
        action = keywords.get(word.strip(), "no_key")
        example = action.__doc__
        if example is None:
            reply += word + "\n"
        else:
            stripped = example.replace("\n", "")
            reply += word + "  " + stripped + "\n"
    post_social_message(reply)
    su.print_help(account)


def latest_cmd(words: list[str], index: int, m: Mastodon, account: str) -> None:
    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    image_dir = cfg["nina"]["image_dir"]
    latest_fits = fitstojpg.get_latest_file(image_dir, "fits")

    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])
    output_path = Path(os.path.join(scratch_dir, cfg["scratch"]["latest_jpg"]))

    latest_jpg = fitsfwhm.save_fwhm(Path(str(latest_fits)), output_path)

    post_social_message("Latest", str(latest_jpg))


keywords = {
    "tonight": tonight_cmd,
    "best": best_cmd,
    "image": image_cmd,
    "db": db_cmd,
    "version": version_cmd,
    "status": status_cmd,
    "latest": latest_cmd,
    "calendar": calendar_cmd,
    "help": help_cmd,
    "?": help_cmd
}


def do_command(sentence: str, m: Mastodon, account: str) -> None:

    if not su.is_super_user(account):
        return

    cmd = sentence.lower()
    words = cmd.split(" ")
    seen_base_command = False
    seen_super_user_commands = False
    #todo does not seem to remove leading blanks

    logger = logging.getLogger(__name__)
    logger.info("Got Command: " + sentence)

    if len(words) < 2:
        post_social_message("Command not recognized, ? for help")
        return

    action = keywords.get(words[1].strip(), "no_key")
    logger.info("Action: " + str(action))

    if action != "no_key":
        action(words, 1, m, account)
        seen_base_command = True

    if seen_base_command is False:
        seen_super_user_commands = su.do_super_user_command(words, account)
    if seen_base_command is False and seen_super_user_commands is False:
        post_social_message("Command not recognized, ? for help")


def do_notification(notification: Any, m: Mastodon) -> None:
    cmd = ""
    try:
        print(notification['type'])
        account = notification.account.acct
        html = notification.status.content.lower()
        note_type = notification['type']
        if note_type == 'mention' or note_type == 'reblog':
            cmd = BeautifulSoup(html, 'html.parser').get_text()
            do_command(cmd, m, account)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.info('Problem')
        logger.exception("Exception")
        m.status_post("Oops I had a problem " + cmd)


class TheStreamListener(StreamListener):

    def on_update(self, status: dict) -> None:
        print(f"Got update: {status['content']}")

    def on_notification(self, notification: Any) -> None:
        cfg = config.data()
        logger = logging.getLogger(__name__)
        mastodon = cfg["globals"]["mastodon instance"]
        do_notification(notification, mastodon)


def get_mastodon_instance() -> Mastodon:
    cfg = config.data()
    logger = logging.getLogger(__name__)
    access_token = cfg["mastodon"]["access_token"]
    api_base_url = cfg["mastodon"]["api_base_url"]
    mastodon = Mastodon(access_token=access_token, api_base_url=api_base_url)
    return mastodon


def post_social_message(message: str, image: Optional[str] = None, vis: Optional[str] = None) -> None:
    cfg = config.data()
    logger = logging.getLogger(__name__)
    mastodon = cfg["globals"]["mastodon instance"]

    if mastodon is None:
        mastodon = get_mastodon_instance()
        cfg["globals"]["mastodon instance"] = mastodon


    if image is None:
        mastodon.status_post(message, visibility=vis)
    else:
        #       media_upload_mastodon = mastodon.media_post(image)
        #        mastodon.media_update(media_upload_mastodon, description="text")
        #       post = mastodon.status_post(message, media_ids=media_upload_mastodon)

        media = mastodon.media_post(image, "image/png")
        mastodon.status_post(message, media_ids=media, visibility=vis)


def handle_mention(notification: Any) -> None:
    cfg = config.data()
    if notification.type == "mention":
        print(notification.status.content)
        mastodon = cfg["globals"]["mastodon instance"]
        do_notification(notification, mastodon)


def start_interface() -> None:
    print("Starting Social Server")
    cfg = config.data()
    post_social_message("Starting Version " + cfg["version"]["date"])

    mastodon = get_mastodon_instance()

    listener = CallbackStreamListener(notification_handler=handle_mention)
    mastodon.stream_user(listener, run_async=True, reconnect_async=True, timeout=600)
    while True:
        asyncio.run(wait_a_bit())




async def wait_a_bit() -> None:
    await asyncio.sleep(10)


def main() -> None:
    utils.set_install_dir()
    cfg = config.data()
    logger = logging.getLogger(__name__)

    cfg["logger"]["logging"] = logger
    path = utils.set_install_dir()
    path = os.path.join(path, 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info('Started Social Server')


    cfg["globals"]["mastodon instance"] = get_mastodon_instance()

    mqtt_client = utils.connect_mqtt()
    cfg["globals"]["mqtt_client"] = mqtt_client

    mqtt_client.subscribe(utils.topic_from_sched)
    mqtt_client.on_message = wait_for_mqtt_message
    mqtt_client.loop_start()

    try:
        start_interface()
    except Exception:
        logger.info('Problem')
        logger.exception("Exception")
        get_mastodon_instance().status_post("Oops I had a problem with Social server")
        start_interface()

    print("stop")


if __name__ == '__main__':
    main()
