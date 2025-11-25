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
import pushover





def is_inside_light_on(dev_map):

    instructions = (dict
        (
        {
            "Iris inside light": "ison"
        }
    ))

    inside_light_on = asyncio.run(ku.kasa_check(dev_map, instructions))
    return inside_light_on


def turn_inside_light_on (dev_map):
    instructions = (dict
        (
        {
            "Iris inside light": 'on',
        }
    ))
    asyncio.run(ku.kasa_do(dev_map, instructions))
    time.sleep(2)
def turn_inside_light_off (dev_map):
    instructions = (dict
        (
        {
            "Iris inside light": 'off',
        }
    ))
    asyncio.run(ku.kasa_do(dev_map, instructions))
    time.sleep(2)

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
    time.sleep(45)
    instructions = (dict
        (
        {
            "Roof motor": 'off',
        }
    ))
    asyncio.run(ku.kasa_do(dev_map, instructions))


def get_status_with_lights():
    dev_map = asyncio.run(ku.make_discovery_map())

    instructions = (dict
        (
        {
            "Iris inside light": 'on',
            "Observatory strip": 'on'
        }
    ))

    asyncio.run(ku.kasa_do(dev_map, instructions))
    time.sleep(10)
    parked, closed, open, mod_date = vision_safety.visual_status()
    time.sleep(10)
    instructions = (dict
        (
        {
            "Iris inside light": 'off',
            "Observatory strip": 'off'
        }
    ))

    asyncio.run(ku.kasa_do(dev_map, instructions))

    return parked, closed, open, mod_date

def open_roof_with_option (check:bool)->bool:
    dev_map = asyncio.run(ku.make_discovery_map())
    if check:
        parked, closed, open, mod_date = get_status_with_lights()
        if parked:
            if closed:
                social_server.post_social_message("Vision Safety says roof is closed, opening roof")
                toggle_roof(dev_map)
                return True

            else:
                social_server.post_social_message("Vision Safety says roof is NOT closed, therefore will not open")
                return False
        else:
            social_server.post_social_message("Vision Safety says Scope is NOT parked, therefore will not open")
            return False
    else:
        toggle_roof(dev_map)
        return False






def open_roof_cmd_no_check(words, account):
    open_roof_with_option(False)



def open_roof_cmd (words, account):
    open_roof_with_option(True)



def park_and_close_cmd(words, account):
    social_server.post_social_message("User has stopped imaging")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:
        file.write("USER UNSAFE")


def safe_cmd(words, account):
    social_server.post_social_message("User has said imaging is safe")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:
        file.write("USER SAFE")

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
        "image!!":doit_cmd,
        "stop!": park_and_close_cmd,
        "safe!": safe_cmd,
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

def is_safe ():

    utils.set_install_dir()
    with open("safety.txt", "r") as file:
        first_line = file.readline()
    if first_line == "USER SAFE":
        return True
    else:
        return False

def doit_cmd (words, account):
    cfg = config.data()

    inside_view= cfg["camera safety"]["scope_view"]

    print (words, account)
    wait_time = 5 * 60
    utils.set_install_dir()
    parked, closed, open, mod_date = get_status_with_lights()
    if not closed:
        pushover.push_message_with_picture("roof is not closed, stopping", inside_view)
        return

    # the roof is closed, so we can start imaging
    pushover.push_message_with_picture("Roof is closed, starting run in 5 min", inside_view)
    if not is_safe():
        pushover.push_message("not safe 1, stopping")
        return


    time.sleep(wait_time)

    if not is_safe():
        pushover.push_message("not safe 2, stopping")
        return



    ok = open_roof_with_option(True)
    if not ok:
        pushover.push_message_with_picture("roof not open, stopping", inside_view)
        return

    pushover.push_message_with_picture("roof is open, starting imaging in 5 min", inside_view)
    time.sleep(wait_time)

    if not is_safe():
        pushover.push_message_with_picture("not safe 3, stopping", inside_view)
        return

    on_nina(None, None)

    # need to add a method to know if Nina is finished
    #write to file that prelude has finished
    time.sleep(wait_time)
    pushover.push_message_with_picture("prelude has finished", inside_view)


    if not is_safe():
        pushover.push_message("not safe 4, stopping")
        return
    # add in check to make sure mount is on

    parked, closed, open, mod_date = get_status_with_lights()
    if not parked:
        pushover.push_message_with_picture("scope is not parked, stopping", inside_view)
        return
    if closed:
        pushover.push_message_with_picture("roof is closed, stopping", inside_view)
        return
    if not open:
        pushover.push_message_with_picture("roof is not open, stopping", inside_view)
        return

    image_nina(None, None)
    pushover.push_message("imaging!")

if __name__ == "__main__":
    print("Starting Nina")
    path = utils.set_install_dir()
    os.chdir(path)
    #os.startfile("on_nina.bat")
    print(path)
    subprocess.run(["on_nina.bat"])
    print("Done with Nina")