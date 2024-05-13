import datetime

from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener

import baseconfig as cfg
import db as cdb
import sun as s
import weather
import logging


config = cfg.FlowConfig().config


def capture_cmd(words, index, m, account):
    if len(words) > index + 1:
        dso = words[index + 1]
        capture_db = cdb.DB()
        now = datetime.datetime.now()
        capture_db.add(dso, now, now, 0, account, "default", "mastodon", 0, "")
        post_social_message("Added to list of objects to image\n")


def status_cmd(words, index, m, account):
# Observatory State
    reply = "Observatory Status: " + config["Globals"]["Observatory State"]



    is_night, angle = s.is_night()
    sun = "daytime"
    if is_night:
        sun = "night "
    reply += "\nMode: " + sun + "\n"
    reply += "Sun Angle: " + "{:10.2f}".format(angle) + "\n"
    description, clouds, wind_speed, moon_phase = weather.get_weather()
    reply += description

    capture_db = cdb.DB()
    rows, expo_time, requesters, dso_objects = capture_db.do_stats()
    reply += "\nRequests: " + str(rows) + "\n"
    reply += "Imaged Minutes: " + str(expo_time) + "\n"
    reply += "Unique Accounts: " + str(len(requesters)) + " \n"
    reply += "DSO Objects: " + str(len(dso_objects)) + "\n"
    post_social_message(reply)


def do_command(sentence, m, account):
    keywords = {
        "capture": capture_cmd,
        "status": status_cmd
    }

    cmd = sentence.lower()
    words = cmd.split(" ")
    for index, word in enumerate(words):
        action = keywords.get(word, "no_key")
        if action != "no_key":
            action(words, index, m, account)


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
        log = logging.getLogger()
        log.exception("Message for you, sir!")
        m.status_post("I do not understand the command " + cmd)


class TheStreamListener(StreamListener):

    def __init__(self, m):
        self.m = m

    def on_update(self, status):
        print(f"Got update: {status['content']}")

    def on_notification(self, notification):
        do_notification(notification, self.m)


def get_mastodon_instance():
    access_token = config["mastodon"]["access_token"]
    api_base_url = config["mastodon"]["api_base_url"]
    mastodon = Mastodon(access_token=access_token, api_base_url=api_base_url)
    return mastodon


def post_social_message(message, image=None):

    mastodon = get_mastodon_instance()
    if image is None:
        mastodon.status_post(message)
    else:
#       media_upload_mastodon = mastodon.media_post(image)
#        mastodon.media_update(media_upload_mastodon, description="text")
#       post = mastodon.status_post(message, media_ids=media_upload_mastodon)

        media = mastodon.media_post(image, "image/jpeg")
        mastodon.status_post(message, media_ids=media)


def start_interface():
    mastodon = get_mastodon_instance()
    user = mastodon.stream_user(TheStreamListener(mastodon), run_async=True)
    print("started mastodon")


