import subprocess

import config
import social_server
import utils
import logging
import pwi4_utils
import os
import kasa_utils as ku
import asyncio
import shelly
import requests
import time





def park_and_close_cmd():
    if not pwi4_utils.park_scope():
        return False
    print ("true from park")
    parked = pwi4_utils.get_is_parked()
    if parked:
        social_server.post_social_message("Mount says Iris is parked")
        dev_map = asyncio.run(ku.make_discovery_map())
        instructions = (dict
            (
            {
                "Telescope mount": 'off',
                "Roof motor": 'on',
                "Iris inside light": 'on'
            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))
        r = requests.get('http://192.168.87.41/relay/0?turn=on')


    else:
        social_server.post_social_message("Mount says Iris is NOT parked")
        return False




def open_if_mount_off_cmd():
    dev_map = asyncio.run(ku.make_discovery_map())
    instructions = (dict
        (
        {
            "Telescope mount": 'isoff',
        }
    ))

    check_ok = asyncio.run(ku.kasa_check(dev_map, instructions))
    if check_ok:
        social_server.post_social_message("Mount is Off")

        instructions = (dict
            (
            {
                "Roof motor": 'on',
                "Iris inside light": 'off'
            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))
        r = requests.get('http://192.168.87.41/relay/0?turn=on')
        time.sleep(30)
        instructions = (dict
            (
            {
                "Roof motor": 'off',
            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))




    else:
        social_server.post_social_message("Mount is not Off")


    return


def start_nina():
    print("Starting Nina")
    path = utils.set_install_dir()
    # fullpath = os.path.join (path, "runnina.bat")
    print(path)
    subprocess.run(["runnina.bat"])
    print("Done with Nina")


def shutdown():
    return


def print_help(account):
    if not is_super_user(account):
        return
    reply = "Available SU commands are\n"
    keywords = get_super_user_commands()
    for word in keywords:
        reply += word + "\n"
    social_server.post_social_message(reply)


def get_super_user_commands():
    return {
        "pandc!": park_and_close_cmd,
        "oim0!": open_if_mount_off_cmd,
        "nina!": start_nina,
        "reboot!": shutdown
    }


def is_super_user(account):
    cfg = config.data()

    super_users = cfg["Super Users"]
    if account in super_users:
        return True
    else:
        return False


def do_super_user_command(words, account):
    if not is_super_user(account):
        print("no auth")
        return False

    su_commands = get_super_user_commands()
    action = su_commands.get(words[1], "no_key")
    print("action is " + str(action) + " word " + str(words[1]) + ".")
    if action != "no_key":
        action()
        return True
    else:
        return False

