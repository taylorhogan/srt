import asyncio
import logging
import os, sys
import subprocess
import time
from collections.abc import Callable
from typing import Any, Optional

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from control import instructions
from hardware_control import kasa_utils as ku
from cmd_processing import social_server
from utils import utils, pushover
from sentry import vision_safety
from end_points import end


_logger = utils.set_logger()

def is_inside_light_on(dev_map: dict) -> bool:
    inst = {"Iris inside light": "ison"}
    inside_light_on = asyncio.run(ku.kasa_check(dev_map, inst))
    return inside_light_on


def turn_inside_light_on(dev_map: dict) -> None:
    inst = {"Iris inside light": 'on'}
    asyncio.run(ku.kasa_do(dev_map, inst))
    time.sleep(2)


def turn_inside_light_off(dev_map: dict) -> None:
    inst = {"Iris inside light": 'off'}
    asyncio.run(ku.kasa_do(dev_map, inst))
    time.sleep(2)


def toggle_roof(dev_map: dict) -> None:
    inst = {"Roof motor": 'on'}
    asyncio.run(ku.kasa_do(dev_map, inst))
    time.sleep(10)
    try:
        r = requests.get('http://192.168.87.41/relay/0?turn=on', timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        _logger.error("Failed to trigger relay in toggle_roof: %s", e)
        raise
    time.sleep(45)
    inst = {"Roof motor": 'off'}
    asyncio.run(ku.kasa_do(dev_map, inst))



def get_status_with_lights() -> tuple[bool, bool, bool, Any]:

    parked, closed, open, mod_date = vision_safety.visual_status()

    return parked, closed, open, mod_date


def open_roof_with_option(check: bool) -> bool:
    dev_map = asyncio.run(ku.make_discovery_map())
    if check:
        parked, closed, open, mod_date = get_status_with_lights()
        if parked:
            if closed:
                social_server.post_social_message("Vision Safety says roof is closed, opening roof")
                toggle_roof(dev_map)
                time.sleep(30)
                parked, closed, open, mod_date = get_status_with_lights()
                return open and parked

            else:
                social_server.post_social_message("Vision Safety says roof is NOT closed, therefore will not open")
                return False
        else:
            social_server.post_social_message("Vision Safety says Scope is NOT parked, therefore will not open")
            return False
    else:
        toggle_roof(dev_map)
        return False


def open_roof_cmd_no_check(words: list[str], account: str) -> None:
    open_roof_with_option(False)


def open_roof_cmd(words: list[str], account: str) -> None:
    open_roof_with_option(True)


def unsafe_cmd(words: list[str], account: str) -> None:
    social_server.post_social_message("User has stopped imaging")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:
        file.write("USER UNSAFE")


def safe_cmd(words: list[str], account: str) -> None:
    social_server.post_social_message("User has said imaging is safe")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:
        file.write("USER SAFE")

def imaging_state(state: bool) -> None:

    utils.set_install_dir()
    with open("imaging.txt", "w") as file:
        if state is True:
            file.write("IMAGING TRUE")
        else:
            file.write("IMAGING FALSE")

def open_if_mount_off_cmd(words: list[str], account: str) -> None:
    dev_map = asyncio.run(ku.make_discovery_map())
    inst = {"Telescope mount": 'isoff'}

    check_ok = asyncio.run(ku.kasa_check(dev_map, inst))
    if check_ok:
        social_server.post_social_message("Mount is Off")

        inst = {"Roof motor": 'on', "Iris inside light": 'off'}
        asyncio.run(ku.kasa_do(dev_map, inst))
        try:
            r = requests.get('http://192.168.87.41/relay/0?turn=on', timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            _logger.error("Failed to trigger relay in open_if_mount_off_cmd: %s", e)
            return
        time.sleep(30)
        inst = {"Roof motor": 'off', "Telescope mount": 'on'}
        asyncio.run(ku.kasa_do(dev_map, inst))




    else:
        social_server.post_social_message("Mount is not Off")

    return


def on_nina(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    path = utils.set_install_dir()
    os.chdir(path)
    # os.startfile("on_nina.bat")
    print(path)
    subprocess.run(["on_nina.bat"], shell=True)
    print("Done with Nina")


def image_nina1(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.Popen(["image_nina1.bat"], shell=True)
    print("Done with Nina")

def image_nina2(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.Popen(["image_nina2.bat"], shell=True)
    print("Done with Nina")

def image_nina_a(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    path = utils.set_install_dir()
    print(path)
    subprocess.Popen(["image_ninaA.bat"])
    print("Done with Nina")


def shutdown(words: list[str], account: str) -> None:
    return


def print_help(account: str) -> None:
    if not is_super_user(account):
        return
    reply = "Available SU commands are\n"
    keywords = get_super_user_commands()
    for word in keywords:
        reply += word + "\n"
    social_server.post_social_message(reply)


def dbb_cmd(words: list[str], account: str) -> None:
    instructions.rehash_db()
    instructions.create_instructions_table(True)


def dbr_cmd(words: list[str], account: str) -> None:
    """
    rehash db, example dbr
    """
    instructions.rehash_db()
    instructions.create_instructions_table()


def dbd_cmd(words: list[str], account: str) -> None:
    """
       delete a db entry, example dbd 12
    """
    instructions.delete_instruction_db(words[2])

    instructions.create_instructions_table()


def dbc_cmd(words: list[str], account: str) -> None:
    """
       mark db entry as complete, example dbc 1
        """
    logger = logging.getLogger(__name__)
    logger.info("db_cmd %s", words)
    instructions.set_completed_instruction_db(words[2])
    instructions.create_instructions_table()


def get_super_user_commands() -> dict[str, Callable]:
    return {
        "dbr": dbr_cmd,
        "dbd": dbd_cmd,
        "dbc": dbc_cmd,
        "dbb": dbb_cmd,
        "image!!": image_cmd,
        "stop!": unsafe_cmd,
        "safe!": safe_cmd,
        "nina1!": on_nina,
        "nina2!": image_nina1,
        "nina2A!": image_nina_a,
        "open!": open_roof_cmd,
        "open!!": open_roof_cmd_no_check
    }


def is_super_user(account: str) -> bool:
    cfg = config.data()

    super_users = cfg["Super Users"]
    if account in super_users:
        return True
    else:
        return False


def do_super_user_command(words: list[str], account: str) -> bool:
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


def is_safe() -> bool:
    utils.set_install_dir()
    try:
        with open("safety.txt", "r") as file:
            first_line = file.readline()
    except FileNotFoundError:
        return False
    return first_line == "USER SAFE"

def is_imaging() -> bool:
    utils.set_install_dir()
    try:
        with open("imaging.txt", "r") as file:
            first_line = file.readline()
    except FileNotFoundError:
        return False
    print(first_line)
    return first_line == "IMAGING TRUE"

def image_cmd(words: list[str], account: str) -> None:
    if is_imaging():
        pushover.push_message("Already imaging, cannot restart")
    else:
        imaging_state(True)
        doit_cmd(words, account)
        imaging_state(False)



def doit_cmd(words: list[str], account: str) -> None:

    _logger.info("doit_cmd")
    cfg = config.data()

    inside_view = cfg["camera safety"]["scope_view"]


    operand = 1
    if len(words) > 2:
        operand = int(words[2])

    pushover.push_message(f"imaging! in mode {operand}")
    wait_time = 1 * 60
    utils.set_install_dir()
    parked, closed, open, mod_date = get_status_with_lights()
    if not closed:
        pushover.push_message("roof is not closed, stopping", inside_view)
        return

    # the roof is closed, so we can start imaging
    pushover.push_message("Roof is closed, starting run in 1 min", inside_view)
    if not is_safe():
        pushover.push_message("not safe 1, stopping")
        return

    time.sleep(wait_time)

    if not is_safe():
        pushover.push_message("not safe 2, stopping")
        return

    ok = open_roof_with_option(True)
    print ("ok=", str(ok))
    if not ok:
        pushover.push_message("problem opening roof, stopping", inside_view)
        return


    pushover.push_message("roof is open, starting imaging in 1 min", inside_view)
    time.sleep(wait_time)

    if not is_safe():
        pushover.push_message("not safe 3, stopping", inside_view)
        return

    if operand == 2 or operand == 1:
        print ("starting Nina")

        on_nina(None, None)

        # need to add a method to know if Nina is finished
        # write to file that prelude has finished
        time.sleep(5*60)
        pushover.push_message("prelude has finished", inside_view)

        if not is_safe():
            pushover.push_message("not safe 4, stopping")
            return
        # add in check to make sure mount is on

        parked, closed, open, mod_date = get_status_with_lights()
        if not parked:
            pushover.push_message("scope is not parked, stopping", inside_view)
            return
        if closed:
            pushover.push_message("roof is closed, stopping", inside_view)
            return
        if not open:
            pushover.push_message("roof is not open, stopping", inside_view)
            return
        if operand == 1:
            image_nina1(None, None)
        else:
            image_nina2(None, None)


        pushover.push_message("imaging!")
    else:
        print ("end started")
        pushover.push_message("just closing up, no imaging")
        time.sleep(60)
        end.do_main()



if __name__ == "__main__":
    print("Starting Nina")
    path = utils.set_install_dir()
    os.chdir(path)
    # os.startfile("on_nina.bat")
    print(path)
    subprocess.run(["on_nina.bat"])
    print("Done with Nina")
