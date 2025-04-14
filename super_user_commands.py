import subprocess

import configuration
import social_server
import kasa_utils as ku
import pwi4_utils
import utils
import os
import asyncio
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
        time.sleep(30)
        instructions = (dict
            (
            {

                "Roof motor": 'off',

            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))




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
                "Telescope mount": 'on',
            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))




    else:
        social_server.post_social_message("Mount is not Off")


    return


def on_nina():
    print("Starting Nina")
    path = utils.set_install_dir()
    os.chdir(path)
    os.startfile("on_nina.bat")
    print(path)
    #subprocess.run(["on_nina.bat"])
    print("Done with Nina")

def image_nina():
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.run(["image_nina.bat"])
    print("Done with Nina")

def image_nina_a():
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.run(["image_ninaA.bat"])
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
        "start!": open_if_mount_off_cmd,
        "stop!": park_and_close_cmd,
        "nina1!": on_nina,
        "nina2!!": image_nina,
        "nina2A!": image_nina_a,
        "reboot!": shutdown
    }


def is_super_user(account):
    cfg = configuration.data()

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

