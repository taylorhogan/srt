import asyncio
import subprocess
import time
import config
import requests
import end
import os

import utils


def close_roof_command ():
    return


def open_roof_command():
    return

def start_nina():
    print("Starting Nina")
    path = utils.set_install_dir()
    #fullpath = os.path.join (path, "runnina.bat")
    print (path)
    subprocess.run(["runnina.bat"])
    print("Done with Nina")


def shutdown():
    return

def get_super_user_commands():
    return {
        "close!'": close_roof_command,
        "open!'": close_roof_command,
         "nina!": start_nina,
        "reboot!": shutdown
    }


def do_super_user_command(words, account):
    cfg = config.data()
    su_commands = get_super_user_commands()
    print (str(su_commands))
    action = su_commands.get(words[1], "no_key")
    print("action is " + str (action) + " word " + str(words[1]) + ".")
    if action != "no_key":
        super_users = cfg["Super Users"]
        if account in super_users:
            print("Whoot")
            action()
            return True
        else:
            # social_server.post_social_message(account + " Is not authorized\n")
            print("no auth")
            return False
    else:
        return False


