import subprocess

import config
import social_server
import kasa_utils as ku
import utils
import os
import asyncio
import requests
import time
import instructions
import logging
import vision_safety




def toggle_roof (dev_map):

    instructions = (dict
        (
        {
            "Roof motor": 'on',
        }
    ))

    asyncio.run(ku.kasa_do(dev_map, instructions))
    time.sleep(10)
    r = requests.get('http://192.168.87.41/relay/0?turn=on')
    time.sleep(30)
    instructions = (dict
        (
        {
            "Roof motor": 'off',
        }
    ))
    asyncio.run(ku.kasa_do(dev_map, instructions))

def open_roof_with_option (check:bool):
    dev_map = asyncio.run(ku.make_discovery_map())
    if check:

        instructions = (dict
            (
            {
                "Iris inside light": 'on'
            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))

        parked, closed, open, mod_date = vision_safety.visual_status()

        instructions = (dict
            (
            {
                "Iris inside light": 'off'
            }
        ))

        asyncio.run(ku.kasa_do(dev_map, instructions))

        if parked:
            if closed:
                social_server.post_social_message("Vision Safety says roof is closed, opening roof")
                toggle_roof(dev_map)

            else:
                social_server.post_social_message("Vision Safety says roof is NOT closed, therefore will not open")
                return
        else:
            social_server.post_social_message("Vision Safety says Scope is NOT parked, therefore will not open")
            return
    else:
        toggle_roof(dev_map)






def open_roof_cmd_no_check(words, account):
    open_roof_with_option(False)



def open_roof_cmd (words, account):
    open_roof_with_option(True)



def park_and_close_cmd(words, account):
    social_server.post_social_message("User has stopped imaging")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:  # 'w' mode to write (overwrites if file exists)
        file.write("USER UNSAFE")




def open_if_mount_off_cmd(words, account):
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


def on_nina(words, account):
    print("Starting Nina")
    path = utils.set_install_dir()
    os.chdir(path)
    #os.startfile("on_nina.bat")
    print(path)
    subprocess.run(["on_nina.bat"],shell=True)
    print("Done with Nina")

def image_nina(words, account):
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.Popen(["image_nina.bat"],shell=True)
    print("Done with Nina")

def image_nina_a(words, account):
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.Popen(["image_ninaA.bat"])
    print("Done with Nina")


def shutdown(words, account):
    return


def print_help(account):
    if not is_super_user(account):
        return
    reply = "Available SU commands are\n"
    keywords = get_super_user_commands()
    for word in keywords:
        reply += word + "\n"
    social_server.post_social_message(reply)

def dbb_cmd (words, account):
    instructions.rehash_db()
    instructions.create_instructions_table(True)

def dbr_cmd(words, account):
    """
    rehash db, example dbr
    """
    instructions.rehash_db()
    instructions.create_instructions_table()


def dbd_cmd(words, account):
    """
       delete a db entry, example dbd 12
    """
    instructions.delete_instruction_db(words[2])

    instructions.create_instructions_table()


def dbc_cmd(words, account):
    """
       mark db entry as complete, example dbc 1
        """
    logger = logging.getLogger(__name__)
    logger.info("db_cmd", words)
    instructions.set_completed_instruction_db(words[2])
    instructions.create_instructions_table()

def get_super_user_commands():
    return {
        "dbr": dbr_cmd,
        "dbd": dbd_cmd,
        "dbc": dbc_cmd,
        "dbb": dbb_cmd,
        "stop!": park_and_close_cmd,
        "nina1!": on_nina,
        "nina2!": image_nina,
        "nina2A!": image_nina_a,
        "open!": open_roof_cmd,
        "open!!": open_roof_cmd_no_check
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
        action(words, account)
        return True
    else:
        return False

if __name__ == "__main__":
    print("Starting Nina")
    path = utils.set_install_dir()
    os.chdir(path)
    #os.startfile("on_nina.bat")
    print(path)
    subprocess.run(["on_nina.bat"])
    print("Done with Nina")