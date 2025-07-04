import asyncio
import datetime
import logging
import os
from datetime import date

from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener
from mastodon.streaming import CallbackStreamListener

import astro_dso_visibility
import config
import end
import fitstojpg
import instructions
import obs_calendar
import sun as s
import super_user_commands as su
import utils
import weather
from utils import topic_to_sched
import json
import time



_json_payload = None


def get_dso_object_name(words, index):
    if len(words) > index + 1:
        dso = words[index + 1]
    else:
        return None
    if len(words) > index + 2:
        dso = dso + " " + words[index + 2]
    return dso


def image_cmd(words, index, m, account):
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


def best_cmd(words, index, m, account):
    """
       example best m 13
    """
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = astro_dso_visibility.is_a_dso_object(dso_name)
        if object is not None:
            best_date, best_time = astro_dso_visibility.best_day_for_dso(object)
            if best_date is not None:
                formatted_date = best_date.strftime("%Y-%m-%d")
                post_social_message(dso_name + " is above horizon for " + str(best_time) + " on " + formatted_date)
            else:
                post_social_message(dso_name + " is never above horizon")
        else:
            post_social_message(dso_name + " Not a known object\n")


def show_cmd(words, index, m, account):
    """
           example show m 13
    """
    logger = logging.getLogger(__name__)
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        obj = astro_dso_visibility.is_a_dso_object(dso_name)
        if obj is not None:
            horizon, image, sky = astro_dso_visibility.show_plots(obj)
            if horizon is not None:
                post_social_message("altitude \n", horizon)
            if image is not None:
                post_social_message("image\n", image)
            if sky is not None:
                post_social_message("sky\n", sky)
        else:
            post_social_message(dso_name + " Not a known object\n")


def weather_cmd(words, index, m, account):
    is_night, angle = s.is_night()
    sun = "daytime"
    if is_night:
        sun = "night "
    reply = "\nMode: " + sun + "\n"
    reply += "Sun Angle: " + "{:10.2f}".format(angle) + "\n"
    description, weather_ok = weather.get_current_weather(True)
    reply += description
    post_social_message(reply)
    description, weather_ok = weather.get_current_weather(False)
    post_social_message(description)


def version_cmd(words, index, m, account):
    # Observatory State
    cfg = config.data()

    reply = "Version: " + cfg["version"]["date"] + "\n"
    reply += "Observatory Status: " + cfg["Globals"]["Observatory State"]
    post_social_message(reply)

def wait_for_mqtt_message (client, userdata, msg):
    global _json_payload
    print ("setting payload to ")
    _json_payload = json.loads(msg.payload.decode("utf-8"))
    _json_payload = json.dumps(_json_payload)
    logger = logging.getLogger(__name__)
    logger.info("Received message: " + _json_payload)
    print("to ", _json_payload)




def status_cmd(words, index, m, account):
    # Observatory State
    cfg = config.data()

    global _json_payload
    mqtt_client = cfg["globals"]["mqtt_client"]
    timeout = 60  # Timeout in seconds
    start_time = time.time()
    _json_payload = None
    print ("setting state to None")
    end.determine_roof_state_visually()
    print ("asking for status")
    mqtt_client.publish(topic_to_sched, "status?")
    while _json_payload is None and (time.time() - start_time < timeout) :
        asyncio.run(wait_a_bit())
    if _json_payload is None:
        post_social_message("Status could not be determined")
    else:
        post_social_message(_json_payload)




def db_cmd(words, index, m, account):
    instructions.create_instructions_table()


def dbr_cmd(words, index, m, account):
    """
    rehash db, example dbr
    """
    instructions.rehash_db()
    instructions.create_instructions_table()


def dbd_cmd(words, index, m, account):
    """
       delete a db entry, example dbd 12
    """
    instructions.delete_instruction_db(words[index + 1])

    instructions.create_instructions_table()


def dbc_cmd(words, index, m, account):
    """
       mark db entry as complete, example dbc 1
        """
    logger = logging.getLogger(__name__)
    logger.info("db_cmd", words)
    instructions.set_completed_instruction_db(words[index + 1])
    instructions.create_instructions_table()


def calendar_cmd(words, index, m, account):
    # Observatory State
    cfg = config.data()
    today = date.today()

    obs_calendar.print_month(today.year, today.month, cfg)
    post_social_message("", "cal.png")


def help_cmd(words, index, m, account):
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


def latest_cmd(words, index, m, account):
    cfg = config.data()

    logger = logging.getLogger(__name__)
    image_dir = cfg["nina"]["image_dir"]
    latest_fits = fitstojpg.get_latest_file(image_dir, "fits")

    latest_jpg = fitstojpg.convert_to_jpg(str(latest_fits))

    post_social_message("Latest", latest_jpg)


keywords = {
    "show": show_cmd,
    "best": best_cmd,
    "image": image_cmd,
    "dbs": db_cmd,
    "dbr": dbr_cmd,
    "dbd": dbd_cmd,
    "dbc": dbc_cmd,
    "version": version_cmd,
    "status": status_cmd,
    "weather": weather_cmd,
    "latest": latest_cmd,
    "calendar": calendar_cmd,
    "help": help_cmd,
    "?": help_cmd
}


def do_command(sentence, m, account):
    cmd = sentence.lower()
    words = cmd.split(" ")
    seen_base_command = False

    logger = logging.getLogger(__name__)
    logger.info("Got Command: " + sentence)

    action = keywords.get(words[1].strip(), "no_key")
    logger.info("Action: " + str(action))

    if action != "no_key":
        action(words, 1, m, account)
        seen_base_command = True

    if seen_base_command is False:
        seen_super_user_commands = su.do_super_user_command(words, account)
    if seen_base_command is False and seen_super_user_commands is False:
        post_social_message("Command not recognized, ? for help")


def do_notification(notification, m):
    try:
        print(notification['type'])
        account = notification.account.acct
        html = notification.status.content.lower()
        note_type = notification['type']
        if note_type == 'mention' or note_type == 'reblog':
            cmd = BeautifulSoup(html, 'html.parser').get_text()
            do_command(cmd, m, account)
    except:
        logger = logging.getLogger(__name__)
        logger.info('Problem')
        logger.exception("Exception")
        m.status_post("Oops I had a problem " + cmd)


class TheStreamListener(StreamListener):

    def on_update(self, status):
        print(f"Got update: {status['content']}")

    def on_notification(self, notification):
        cfg = config.data()
        logger = logging.getLogger(__name__)
        mastodon = cfg["globals"]["mastodon instance"]
        do_notification(notification, mastodon)


def get_mastodon_instance():
    cfg = config.data()
    logger = logging.getLogger(__name__)
    access_token = cfg["mastodon"]["access_token"]
    api_base_url = cfg["mastodon"]["api_base_url"]
    mastodon = Mastodon(access_token=access_token, api_base_url=api_base_url)
    return mastodon


def post_social_message(message, image=None):
    cfg = config.data()
    logger = logging.getLogger(__name__)
    mastodon = cfg["globals"]["mastodon instance"]

    if mastodon is None:
        mastodon = get_mastodon_instance()
        print(mastodon)

    if image is None:
        mastodon.status_post(message, visibility='private')
    else:
        #       media_upload_mastodon = mastodon.media_post(image)
        #        mastodon.media_update(media_upload_mastodon, description="text")
        #       post = mastodon.status_post(message, media_ids=media_upload_mastodon)

        media = mastodon.media_post(image, "image/png")
        mastodon.status_post(message, media_ids=media)


def handle_mention(notification):
    cfg = config.data()
    if notification.type == "mention":
        print(notification.status.content)
        mastodon = cfg["globals"]["mastodon instance"]
        do_notification(notification, mastodon)


def start_interface():
    print("Starting Social Server")
    cfg = config.data()
    post_social_message("Starting Version " + cfg["version"]["date"])

    mastodon = get_mastodon_instance()

    listener = CallbackStreamListener(notification_handler=handle_mention)
    mastodon.stream_user(listener, run_async=True, reconnect_async=True, timeout=600)
    while True:
        asyncio.run(wait_a_bit())




async def wait_a_bit ():
    await asyncio.sleep(10)


def main():
    utils.set_install_dir()
    cfg = config.data()
    logger = logging.getLogger(__name__)

    cfg["logger"]["logging"] = logger

    path = utils.set_install_dir()
    path = os.path.join(path, 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info('Started Social Server')


    mastodon = get_mastodon_instance()

    mastodon = cfg["globals"]["mastodon instance"]

    mqtt_client = utils.connect_mqtt()
    cfg["globals"]["mqtt_client"] = mqtt_client

    mqtt_client.subscribe(utils.topic_from_sched)
    mqtt_client.on_message = wait_for_mqtt_message
    mqtt_client.loop_start()

    try:
        start_interface()
    except:
        logger.info('Problem')
        logger.exception("Exception")
        get_mastodon_instance().status_post("Oops I had a problem with Social server")
        start_interface()

    print("stop")


if __name__ == '__main__':
    main()

