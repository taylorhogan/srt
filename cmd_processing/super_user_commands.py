import asyncio
import logging
import os, sys
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from control import instructions
from hardware_control import kasa_utils as ku
from hardware_control import sonos_utils
from cmd_processing import social_server
from utils import utils, pushover
from sentry import vision_safety
from end_points import end
from iris_astronomy import astro_dso_visibility
from nina_gen import nina_sequence_gen


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



def announce_roof_movement(text: str, speaker_name: str = "Observatory", volume: int = 40) -> None:
    """Blink observatory lights for 30 sec and say text on Sonos."""
    light_names = ["Iris inside light", "Iris landscape lights", "Observatory strip", "Iris door light"]

    dev_map = asyncio.run(ku.make_discovery_map())
    original_states = asyncio.run(ku.kasa_get_states(dev_map, light_names))

    async def _blink_async():
        end_time = asyncio.get_event_loop().time() + 30
        state = True
        while asyncio.get_event_loop().time() < end_time:
            inst = {k: ("on" if state else "off") for k in light_names}
            try:
                await ku.kasa_do(dev_map, inst)
            except Exception as e:
                _logger.warning("Blink step failed: %s", e)
            state = not state
            await asyncio.sleep(3)
        try:
            await ku.kasa_do(dev_map, original_states)
        except Exception as e:
            _logger.warning("Blink restore failed: %s", e)

    def blink():
        asyncio.run(_blink_async())

    def announce():
        try:
            sonos_utils.sonos_say(text, speaker_name, volume)
        except Exception as e:
            _logger.error("Sonos announcement failed: %s", e)

    blink_thread = threading.Thread(target=blink, daemon=True)
    sonos_thread = threading.Thread(target=announce, daemon=True)
    blink_thread.start()
    sonos_thread.start()
    blink_thread.join()
    sonos_thread.join()


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


def set_mode(mode: str) -> None:
    utils.set_install_dir()
    with open("mode.txt", "w") as file:
        file.write(f"MODE {mode.upper()}")


def mode_cmd(words: list[str], account: str) -> None:
    if len(words) < 3 or words[2] not in ("auto", "manual"):
        social_server.post_social_message("Usage: mode auto|manual")
        return
    set_mode(words[2])
    social_server.post_social_message(f"Mode set to {words[2]}")


def get_mode() -> str:
    utils.set_install_dir()
    try:
        with open("mode.txt", "r") as file:
            line = file.readline().strip()
        if line == "MODE AUTO":
            return "auto"
    except FileNotFoundError:
        pass
    return "manual"

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


_SCRIPTS_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "scripts")


def on_nina(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    subprocess.run([os.path.join(_SCRIPTS_DIR, "on_nina.bat")], shell=True)
    print("Done with Nina")


def image_nina1(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    subprocess.Popen([os.path.join(_SCRIPTS_DIR, "image_nina1.bat")], shell=True)
    print("Done with Nina")

def image_nina2(words: Optional[list[str]], account: Optional[str]) -> None:
    print("Starting Nina")
    subprocess.Popen([os.path.join(_SCRIPTS_DIR, "image_nina2.bat")], shell=True)
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


def announce_cmd(words: list[str], account: str) -> None:
    """Say text on a Sonos speaker. Usage: announce <speaker_name> <text...>"""
    if len(words) < 4:
        social_server.post_social_message("Usage: announce <speaker_name> <text>")
        return
    speaker_name = words[2]
    text = " ".join(words[3:])
    try:
        sonos_utils.sonos_say(text, speaker_name)
    except Exception as e:
        social_server.post_social_message(f"Announce failed: {e}")


def prioritize_cmd(words: list[str], account: str) -> None:
    """
    Give a DSO top scheduling priority. Usage: prioritize <dso>  e.g. prioritize m 31
    """
    if len(words) < 3:
        social_server.post_social_message("Usage: prioritize <dso name>  e.g. prioritize m 31")
        return
    dso_name = words[2]
    if len(words) > 3:
        dso_name = dso_name + " " + words[3]
    matched = instructions.set_priority_instruction_db(dso_name, priority=100)
    if matched:
        social_server.post_social_message(f"{dso_name} set to top priority")
    else:
        social_server.post_social_message(f"{dso_name} not found in waiting instructions")


def sequence_cmd(words: list[str], account: str) -> None:
    """
    Generate a NINA sequence for a DSO. Usage: sequence <dso>  e.g. sequence m 31
    """
    cfg = config.data()

    # Accept one- or two-word DSO names: "sequence m 31" or "sequence ngc6888"
    if len(words) < 3:
        social_server.post_social_message("Usage: sequence <dso name>  e.g. sequence m 31")
        return

    dso_name = words[2]
    if len(words) > 3:
        dso_name = dso_name + " " + words[3]

    dso = astro_dso_visibility.is_a_dso_object(dso_name)
    if dso is None:
        social_server.post_social_message(f"{dso_name} is not a known object")
        return

    ra_hours = dso.coord.ra.hour
    dec_degrees = dso.coord.dec.deg

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    template_path = Path(os.path.join(_project_root, cfg["nina"]["sequence_input"]))
    output_path = Path(cfg["nina"]["sequence_output"])

    try:
        nina_sequence_gen.generate_sequence(
            template_path=template_path,
            dso_name=dso_name,
            ra_hours=ra_hours,
            dec_degrees=dec_degrees,
            output_path=output_path,
        )
        social_server.post_social_message(
            f"Sequence generated for {dso_name} "
            f"(RA {ra_hours:.4f}h  Dec {dec_degrees:+.4f}°) → {output_path.name}"
        )
        _logger.info("sequence_cmd: generated sequence for %s", dso_name)
    except Exception as e:
        _logger.exception("sequence_cmd: failed for %s", dso_name)
        social_server.post_social_message(f"Failed to generate sequence for {dso_name}: {e}")


def get_super_user_commands() -> dict[str, Callable]:
    return {
        "dbr": dbr_cmd,
        "dbd": dbd_cmd,
        "dbc": dbc_cmd,
        "dbb": dbb_cmd,
        "image!!": image_cmd,
        "stop!": unsafe_cmd,
        "safe!": safe_cmd,
        "announce": announce_cmd,
        "sequence": sequence_cmd,
        "mode": mode_cmd,
        "prioritize": prioritize_cmd,
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
    announce_roof_movement("The roof will be opening in 5 Minutes")