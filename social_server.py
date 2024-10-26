import datetime
import logging
import time

from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener

import baseconfig as cfg
import db as cdb
import inside_camera_server
import request_observatory_state
import sortdsoobjects
import sun as s
import super_user_commands as su
import weather
import vision_safety

config = cfg.FlowConfig().config

logger = logging.getLogger(__name__)
config["logger"]["logging"] = logger

def get_dso_object_name(words, index):
    if len(words) > index + 1:
        dso = words[index + 1]
    else:
        return None
    if len(words) > index + 2:
        dso = dso + " " + words[index + 2]
    return dso


def capture_cmd(words, index, m, account):
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        object = sortdsoobjects.is_a_dso_object(dso_name)
        if object is not None:
            capture_db = cdb.DB()
            now = datetime.datetime.now()
            capture_db.add(dso_name, now, now, 0, account, "default", "mastodon", 0, "")
            post_social_message(dso_name + " Added to list of objects to image\n")

        else:
            post_social_message(dso_name + " Not a known object\n")


def show_cmd(words, index, m, account):
    dso_name = get_dso_object_name(words, index)
    if dso_name is not None:
        obj = sortdsoobjects.is_a_dso_object(dso_name)
        if obj is not None:
            altitude, image, sky = sortdsoobjects.show_plots(obj)
            post_social_message("altitude \n", altitude)
            post_social_message("image\n", image)
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
    description, clouds, wind_speed, moon_phase = weather.get_weather()
    reply += description
    post_social_message(reply)


def wait_for_picture_received():
    while True:
        time.sleep(2)
        if config["camera safety"]["valid_data"] == True:
            break

def status_cmd(words, index, m, account):
    # Observatory State
    print ("status")
    reply = "Version: " + config["version"]["date"] + "\n"
    reply += "Observatory Status: " + config["Globals"]["Observatory State"]
    post_social_message(reply)
    print ("A")
    if inside_camera_server.take_snapshot():

        print ("B")
        print ("asking for state")

        print ("received")
        is_closed, is_open, is_parked, mod_date = vision_safety.analyse_safety(config["camera safety"]["out_picture"])
        reply = "Roof Closed: " + str(is_closed) + "\n"
        reply += "Roof Open: " + str(is_open) + "\n"
        reply += "Scope Parked:" + str(is_parked) + "\n"
        reply += "Copied Date:" + mod_date + "\n"
        post_social_message(reply, config["camera safety"]["out_picture"])
    else:
        post_social_message("Problem taking picture")




def help_cmd(words, index, m, account):
    reply = "Available commands are\n"
    for word in keywords:
        reply += word + "\n"
    post_social_message(reply)


keywords = {
    "show": show_cmd,
    "capture": capture_cmd,
    "status": status_cmd,
    "weather": weather_cmd,
    "help": help_cmd,
    "?": help_cmd
}


def do_command(sentence, m, account):
    cmd = sentence.lower()
    words = cmd.split(" ")
    seen_base_command = False

    action = keywords.get(words[1], "no_key")
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
        logger.info('Problem')
        logger.exception("Exception")
        m.status_post("I do not understand the command " + cmd)


class TheStreamListener(StreamListener):

    def on_update(self, status):
        print(f"Got update: {status['content']}")

    def on_notification(self, notification):
        mastodon = config["mastodon"]["instance"]
        do_notification(notification, mastodon)


def get_mastodon_instance():
    access_token = config["mastodon"]["access_token"]
    api_base_url = config["mastodon"]["api_base_url"]
    mastodon = Mastodon(access_token=access_token, api_base_url=api_base_url)
    return mastodon


def post_social_message(message, image=None):
    mastodon = config["mastodon"]["instance"]
    if image is None:
        mastodon.status_post(message)
    else:
        #       media_upload_mastodon = mastodon.media_post(image)
        #        mastodon.media_update(media_upload_mastodon, description="text")
        #       post = mastodon.status_post(message, media_ids=media_upload_mastodon)

        media = mastodon.media_post(image, "image/jpeg")
        mastodon.status_post(message, media_ids=media)


def start_interface():
    post_social_message("Starting Version " + config["version"]["date"])
    mastodon = config["mastodon"]["instance"]
    user = mastodon.stream_user(TheStreamListener(), run_async=True, reconnect_async=True)
    while True:
        time.sleep(100)


def main():
    logging.basicConfig(filename='iris.log', level=logging.INFO)
    logger.info('Started')
    print("start")
    mastodon = get_mastodon_instance()
    config["mastodon"]["instance"] = mastodon
    start_interface()
    print("stop")


if __name__ == '__main__':
    main()
