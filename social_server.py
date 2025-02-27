import datetime
import logging
import time

from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener
from mastodon.streaming import CallbackStreamListener

import fitstojpg

import instructions
import obs_calendar
from datetime import date
import config
import sortdsoobjects
import sun as s
import super_user_commands as su
import weather
import os


def get_dso_object_name(words, index):
    if len(words) > index + 1:
        dso = words[index + 1]
    else:
        return None
    if len(words) > index + 2:
        dso = dso + " " + words[index + 2]
    return dso


def image_cmd(words, index, m, account):
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = sortdsoobjects.is_a_dso_object(dso_name)
        if object is not None:

            now = datetime.datetime.now()
            instructions.add_dso_object_instruction(dso_name,"",account)
            post_social_message(dso_name + " Added to list of objects to image\n")
        else:
            post_social_message(dso_name + " Not a known object\n")

def best_cmd(words, index, m, account):
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = sortdsoobjects.is_a_dso_object(dso_name)
        if object is not None:
            best_date, best_time = sortdsoobjects.best_day_for_dso(object)
            if best_date is not None:
                post_social_message(dso_name + " is above horizon for " + best_date + " on "+ best_date)
            else:
                post_social_message(dso_name + " is never above horizon")
        else:
            post_social_message(dso_name + " Not a known object\n")

def show_cmd(words, index, m, account):
    logger = logging.getLogger(__name__)
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        obj = sortdsoobjects.is_a_dso_object(dso_name)
        if obj is not None:
            horizon, image, sky = sortdsoobjects.show_plots(obj)
            logger.info(horizon)
            post_social_message("altitude \n", horizon)
            #post_social_message("image\n", image)
           #post_social_message("sky\n", sky)
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


def status_cmd(words, index, m, account):
    # Observatory State
    cfg = config.data()

    print("status")
    reply = "Version: " + cfg["version"]["date"] + "\n"
    reply += "Observatory Status: " + cfg["Globals"]["Observatory State"]
    post_social_message(reply)


def instructions_cmd(words, index, m, account):
    instructions.create_instructions_table()
    post_social_message("", "instructions.png")

def calendar_cmd(words, index, m, account):
    # Observatory State
    cfg = config.data()
    today =date.today()

    obs_calendar.print_month (today.year, today.month, cfg)
    post_social_message("", "cal.png")


def help_cmd(words, index, m, account):
    reply = "Available commands are\n"
    for word in keywords:
        reply += word + "\n"
    post_social_message(reply)


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
    "requests": instructions_cmd,
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

    action = keywords.get(words[1].strip(), "no_key")
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
        mastodon = cfg["mastodon"]["instance"]
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
    mastodon = cfg["mastodon"]["instance"]

    if mastodon is None:
        mastodon = get_mastodon_instance()
        print(mastodon)

    if image is None:
        mastodon.status_post(message)
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
        mastodon = cfg["mastodon"]["instance"]
        do_notification(notification, mastodon)


def start_interface():
    cfg = config.data()
    post_social_message("Starting Version " + cfg["version"]["date"])

    mastodon = get_mastodon_instance()

    listener = CallbackStreamListener(notification_handler=handle_mention)
    mastodon.stream_user(listener, run_async=True, reconnect_async=True, timeout=600)
    while True:
        time.sleep(1)


def main():
    print ("Starting")
    cfg = config.data()

    logger = logging.getLogger(__name__)
    print ("1")
    cfg["logger"]["logging"] = logger
    print("2")
    if os.name == 'posix':
        path = cfg["InstallL"]
    else:
        path = cfg["Install"]
    print("3")
    if os.path.exists(path):
        os.chdir(path)
    print("5")
    logging.basicConfig(filename='iris.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info('Started Social Server')
    print("6")
    mastodon = get_mastodon_instance()
    print ("7")
    cfg["mastodon"]["instance"] = mastodon
    print(mastodon)
    try:
        start_interface()
    except:
        logger.info('Problem')
        logger.exception("Exception")
        get_mastodon_instance().status_post("Oops I had a problem with server")
        start_interface()

    print("stop")


if __name__ == '__main__':
    main()
