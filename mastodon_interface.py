import datetime

from bs4 import BeautifulSoup
from mastodon import Mastodon, StreamListener

import baseconfig as cfg
import db as cdb
import sun as s

config = cfg.FlowConfig().config


def capture_cmd(words, index, m, account):
    if len(words) > index + 1:
        dso = words[index + 1]
        capture_db = cdb.DB()
        now = datetime.datetime.now()
        capture_db.add(dso, now, now, 0, account, "default", "mastodon", 0, "")
        m.status_post(dso, " Added to list of objects to image")


def status_cmd(words, index, m, account):
    capture_db = cdb.DB()
    rows, expo_time, requesters, dso_objects = capture_db.do_stats()
    reply = "Requests: " + str(rows) + "\n"
    reply += "Imaged Minutes: " + str(expo_time) + "\n"
    reply += "Unique Accounts: " + str(len(requesters)) + " \n"
    reply += "DSO Objects: " + str(len(dso_objects)) + "\n"

    is_night, angle = s.is_night()
    sun = "daytime"
    if is_night:
        sun = "night "
    reply += "Mode: " + sun + "\n"
    reply += "Sun Angle: " + str(angle) + "\n"
    m.status_post(reply)


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
        m.status_post("I do not understand the command" + cmd)


class TheStreamListener(StreamListener):

    def __init__(self, m):
        self.m = m

    def on_update(self, status):
        print(f"Got update: {status['content']}")

    def on_notification(self, notification):
        do_notification(notification, self.m)





def post_mastodon_message(message):
    m = Mastodon(
        access_token=config["mastodon"]["access_token"],
        api_base_url=config["mastodon"]["api_base_url"]
    )
    m.status_post(message)

def start_interface():

    m = Mastodon(
    access_token=config["mastodon"]["access_token"],
    api_base_url=config["mastodon"]["api_base_url"])
    user = m.stream_user(TheStreamListener(m))



