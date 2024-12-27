import datetime
import logging
import time

from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener

import fitstojpg
import db as cdb
import inside_camera_server
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
    description, weather_ok =  weather.get_current_weather()
    reply += description
    post_social_message(reply)
    link = '<a href="https://www.cleardarksky.com/c/CntnCTkey.html"> Cloud Cover</a>'
    post_social_message(link)



def status_cmd(words, index, m, account):
    # Observatory State
    cfg = config.data()

    print ("status")
    reply = "Version: " + cfg["version"]["date"] + "\n"
    reply += "Observatory Status: " + cfg["Globals"]["Observatory State"]
    post_social_message(reply)





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
    "capture": capture_cmd,
    "status": status_cmd,
    "weather": weather_cmd,
    "latest": latest_cmd,
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
        print (mastodon)

    if image is None:
        mastodon.status_post(message)
    else:
        #       media_upload_mastodon = mastodon.media_post(image)
        #        mastodon.media_update(media_upload_mastodon, description="text")
        #       post = mastodon.status_post(message, media_ids=media_upload_mastodon)

        media = mastodon.media_post(image, "image/jpeg")
        mastodon.status_post(message, media_ids=media)



def handle_mention(status):
    cfg = config.data()
    if '@tmhobservatory' in status.content:
        logger = logging.getLogger(__name__)
        mastodon = cfg["mastodon"]["instance"]
        html = status.content.lower()
        cmd = BeautifulSoup(html, 'html.parser').get_text()



def start_interface():
    cfg = config.data()
    post_social_message("Starting Version " + cfg["version"]["date"])
    mastodon = cfg["mastodon"]["instance"]
    mastodon = get_mastodon_instance()
    #user = mastodon.stream_user(TheStreamListener(), run_async=True, reconnect_async=True)

    # Start streaming for mentions
    mastodon.stream_user(handle_mention)
    while True:
        time.sleep(100)


def main():
    cfg = config.data()

    logger = logging.getLogger(__name__)

    cfg["logger"]["logging"] = logger
    path = cfg["Install"]
    if  os.path.exists(path):
        os.chdir(cfg["Install"])
    logging.basicConfig(filename='iris.log', level=logging.INFO,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.info('Started Social Server')

    mastodon = get_mastodon_instance()
    cfg["mastodon"]["instance"] = mastodon
    try:
        start_interface()
    except:
        logger.info('Problem')
        logger.exception("Exception")
        get_mastodon_instance().status_post("Oops I had a problem with server")

    print("stop")



if __name__ == '__main__':
    main()
