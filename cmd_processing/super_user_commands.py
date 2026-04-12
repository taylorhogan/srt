import asyncio
from datetime import datetime, timedelta
import logging
import os, sys
import subprocess
import threading
import time
from collections.abc import Callable
from enum import Enum
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
    """Check whether the observatory inside light is currently on via Kasa."""
    inst = {"Iris inside light": "ison"}
    inside_light_on = asyncio.run(ku.kasa_check(dev_map, inst))
    return inside_light_on


def turn_inside_light_on(dev_map: dict) -> None:
    """Turn on the observatory inside light and wait for the relay to settle."""
    inst = {"Iris inside light": 'on'}
    asyncio.run(ku.kasa_do(dev_map, inst))
    time.sleep(2)


def turn_inside_light_off(dev_map: dict) -> None:
    """Turn off the observatory inside light and wait for the relay to settle."""
    inst = {"Iris inside light": 'off'}
    asyncio.run(ku.kasa_do(dev_map, inst))
    time.sleep(2)


def toggle_roof(dev_map: dict) -> None:
    """Power the roof motor, trigger the Shelly relay to move the roof, then power off.

    The roof direction (open/close) depends on its current position — the relay
    simply toggles. Waits 45 seconds for the roof to complete its travel.
    """
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
            await asyncio.sleep(5)
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
    """Take a camera snapshot and return (parked, closed, open, mod_date) via vision safety."""
    parked, closed, open, mod_date = vision_safety.visual_status()

    return parked, closed, open, mod_date


def open_roof_with_option(check: bool) -> bool:
    """Open the observatory roof. If *check* is True, verify the scope is parked
    and the roof is closed via vision safety before toggling. Returns True only
    when the roof is confirmed open and the scope is still parked.
    """
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
    """Mark conditions as unsafe — writes USER UNSAFE to safety.txt.

    Any in-progress imaging run will abort at its next safety gate.
    Command: ``stop!``
    """
    social_server.post_social_message("User has stopped imaging")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:
        file.write("USER UNSAFE")


def safe_cmd(words: list[str], account: str) -> None:
    """Mark conditions as safe for imaging — writes USER SAFE to safety.txt.

    Must be issued before starting an imaging run; ``doit_cmd`` checks this
    at multiple safety gates throughout the night.
    Command: ``safe!``
    """
    social_server.post_social_message("User has said imaging is safe")
    utils.set_install_dir()
    with open("safety.txt", "w") as file:
        file.write("USER SAFE")

class ImagingState(Enum):
    NONE         = "NONE"
    ACTIVE       = "ACTIVE"
    IN_PRELUDE   = "IN_PRELUDE"
    DONE_PRELUDE = "DONE_PRELUDE"
    IN_MAIN      = "IN_MAIN"
    DONE_MAIN    = "DONE_MAIN"
    IN_FLATS     = "IN_FLATS"
    DONE_FLATS   = "DONE_FLATS"


def set_imaging_state(state: ImagingState) -> None:
    """Persist the current imaging phase to imaging.txt and announce it on the web chat.

    External processes (NINA bat scripts) also call this via set_imaging_state.bat
    to signal phase transitions like DONE_PRELUDE or DONE_FLATS.
    """
    print (f"IMAGING_STATE {state.value}")
    utils.set_install_dir()
    with open("imaging.txt", "w") as file:
        file.write(f"IMAGING_STATE {state.value}")
    social_server.post_social_message(f"Imaging state: {state.value}")


def set_mode(mode: str) -> None:
    """Write the scheduler mode (auto or manual) to mode.txt and announce on the web chat."""
    utils.set_install_dir()
    with open("mode.txt", "w") as file:
        file.write(f"MODE {mode.upper()}")
    social_server.post_social_message(f"Mode: {mode}")


def mode_cmd(words: list[str], account: str) -> None:
    """Set the scheduler to auto or manual mode. Command: ``mode auto|manual``"""
    if len(words) < 3 or words[2] not in ("auto", "manual"):
        social_server.post_social_message("Usage: mode auto|manual")
        return
    set_mode(words[2])


def get_mode() -> str:
    """Read the current scheduler mode from mode.txt. Returns 'auto' or 'manual'."""
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
    """Open the roof only if the telescope mount power is off.

    Safety measure: refuses to toggle the roof if the mount is powered on,
    since a powered mount may be tracking and the scope could be in the
    path of the roof.
    """
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


def _wait_for_nina_exit(poll_interval: int = 30, timeout: int = 600) -> None:
    """Block until no NINA.exe process is running.

    Polls every *poll_interval* seconds. Gives up after *timeout* seconds
    and logs a warning, but does not raise — the caller can proceed anyway.
    """
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq NINA.exe"],
            capture_output=True, text=True, shell=True,
        )
        if "NINA.exe" not in result.stdout:
            _logger.info("NINA process has exited")
            return
        _logger.info("NINA still running, waiting %ds", poll_interval)
        time.sleep(poll_interval)
    _logger.warning("Timed out waiting for NINA to exit after %ds", timeout)


def on_nina(words: Optional[list[str]], account: Optional[str]) -> None:
    """Launch the NINA prelude script (blocking).

    Runs on_nina.bat which connects the mount, performs a meridian flip if
    needed, runs autofocus, and slews to the target. Blocks until the bat
    file exits.
    """
    print("Starting Nina")
    subprocess.run([os.path.join(_SCRIPTS_DIR, "on_nina.bat")], shell=True)
    print("Done with Nina")


def image_nina1(words: Optional[list[str]], account: Optional[str]) -> None:
    """Launch the primary NINA imaging sequence (non-blocking Popen)."""
    print("Starting Nina")
    subprocess.Popen([os.path.join(_SCRIPTS_DIR, "image_nina1.bat")], shell=True)
    print("Done with Nina")

def image_nina2(words: Optional[list[str]], account: Optional[str]) -> None:
    """Launch the secondary NINA imaging sequence (non-blocking Popen)."""
    print("Starting Nina")
    subprocess.Popen([os.path.join(_SCRIPTS_DIR, "image_nina2.bat")], shell=True)
    print("Done with Nina")


def home_and_park(words: Optional[list[str]], account: Optional[str]) -> None:
    """Slew the scope home and park it via NINA (non-blocking Popen). No imaging."""
    print("Starting Nina home and park")
    subprocess.Popen([os.path.join(_SCRIPTS_DIR, "home_and_park.bat")], shell=True)
    print("Done with Nina home and park")


def shutdown(words: list[str], account: str) -> None:
    """Placeholder for a future shutdown command. Currently a no-op."""
    return


def print_help(account: str) -> None:
    """Post the list of available super-user commands to the web chat. Only responds to super users."""
    if not is_super_user(account):
        return
    reply = "Available SU commands are\n"
    keywords = get_super_user_commands()
    for word in keywords:
        reply += word + "\n"
    social_server.post_social_message(reply)


def dbb_cmd(words: list[str], account: str) -> None:
    """Rehash and fully rebuild the imaging queue from scratch. Command: ``dbb``"""
    instructions.rehash_db()
    instructions.create_instructions_table(True)


def dbr_cmd(words: list[str], account: str) -> None:
    """Rehash the imaging queue and regenerate the instructions table. Command: ``dbr``"""
    instructions.rehash_db()
    instructions.create_instructions_table()


def dbd_cmd(words: list[str], account: str) -> None:
    """Delete an entry from the imaging queue by ID. Command: ``dbd <id>``"""
    instructions.delete_instruction_db(words[2])

    instructions.create_instructions_table()


def dbc_cmd(words: list[str], account: str) -> None:
    """Mark an imaging queue entry as completed by ID. Command: ``dbc <id>``"""
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
    Give a DSO top scheduling priority, or reset all priorities.
    Usage: prioritize <dso>  e.g. prioritize m 31
           prioritize        (no args — resets all waiting objects to equal priority)
    """
    if len(words) < 3:
        count = instructions.reset_all_priorities_db()
        social_server.post_social_message(f"All priorities reset ({count} objects)")
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

    from astropy.time import Time
    above_horizon, _ = astro_dso_visibility.get_above_horizon_time(dso, Time.now())
    above_horizon_seconds = above_horizon.total_seconds() if above_horizon is not None else None

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    template_path = Path(os.path.join(_project_root, cfg["nina"]["sequence_input"]))
    output_path = Path(cfg["nina"]["sequence_output"])

    try:
        filter_plan = nina_sequence_gen.generate_sequence(
            template_path=template_path,
            dso_name=dso_name,
            ra_hours=ra_hours,
            dec_degrees=dec_degrees,
            output_path=output_path,
            above_horizon_seconds=above_horizon_seconds,
        )
        plan_str = "  ".join(f"{f}×{n}" for f, n in filter_plan.items()) if filter_plan else "no filter plan"
        social_server.post_social_message(
            f"Sequence generated for {dso_name} "
            f"(RA {ra_hours:.4f}h  Dec {dec_degrees:+.4f}°) → {output_path.name}\n"
            f"{plan_str}"
        )
        _logger.info("sequence_cmd: generated sequence for %s  plan=%s", dso_name, filter_plan)
    except Exception as e:
        _logger.exception("sequence_cmd: failed for %s", dso_name)
        social_server.post_social_message(f"Failed to generate sequence for {dso_name}: {e}")


def get_super_user_commands() -> dict[str, Callable]:
    """Return the command-name → handler mapping for all super-user commands."""
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
        "doflats": doflats_cmd,
        "stats": image_stats_cmd,
    }


def is_super_user(account: str) -> bool:
    """Return True if *account* is in the configured Super Users list."""
    cfg = config.data()

    super_users = cfg["Super Users"]
    if account in super_users:
        return True
    else:
        return False


def do_super_user_command(words: list[str], account: str) -> bool:
    """Dispatch a super-user command. Returns True if a handler ran, False otherwise."""
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
    """Return True if the observatory has been marked safe via ``safe!`` command."""
    utils.set_install_dir()
    try:
        with open("safety.txt", "r") as file:
            first_line = file.readline()
    except FileNotFoundError:
        return False
    return first_line == "USER SAFE"

def get_scheduler_state() -> dict:
    """Read the scheduler's current state from scheduler_state.json.

    Returns a dict with keys ``state``, ``dso``, and ``will image tonight``.
    Falls back to default unknown values if the file is missing or unreadable.
    """
    utils.set_install_dir()
    try:
        import json as _json
        with open("scheduler_state.json", "r") as f:
            return _json.load(f)
    except Exception:
        return {"state": "unknown", "dso": "unknown", "will image tonight": "unknown"}


def get_imaging_state() -> ImagingState:
    """Read the current imaging state from ``imaging.txt``. Returns NONE if missing."""
    utils.set_install_dir()
    try:
        with open("imaging.txt", "r") as file:
            line = file.readline().strip()
        parts = line.split()
        if len(parts) == 2 and parts[0] == "IMAGING_STATE":
            try:
                return ImagingState(parts[1])
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    return ImagingState.NONE


def is_imaging() -> bool:
    """Return True if any imaging activity is currently in progress."""
    return get_imaging_state() != ImagingState.NONE

def image_cmd(words: list[str], account: str) -> None:
    """Start a full imaging run in a background thread (non-blocking)."""
    if is_imaging():
        pushover.push_message("Already imaging, cannot restart")
    else:
        def _run():
            try:
                doit_cmd(words, account)
            finally:
                set_imaging_state(ImagingState.NONE)
        threading.Thread(target=_run, daemon=True).start()



def doit_cmd(words: list[str], account: str) -> None:
    """
    Full observatory imaging run. Called by image_cmd (manual trigger) or the
    scheduler (auto mode). Manages the entire night: safety checks → roof open
    → NINA prelude → NINA main imaging.

    operand meanings:
        1 = full run (prelude + image_nina1)
        2 = full run (prelude + image_nina2)
        3 = full run (prelude + home_and_park, no imaging)

    State machine transitions written to imaging.txt during this function:
        NONE → ACTIVE   (immediately, guards against concurrent calls)
        ACTIVE → IN_PRELUDE   (just before on_nina.bat launches)
        IN_PRELUDE → DONE_PRELUDE   (written externally by NINA/bat when prelude ends)
        DONE_PRELUDE → IN_MAIN   (just before image_nina bat launches)
        IN_MAIN → NONE   (written by image_cmd when doit_cmd returns, or by NINA bat)

    Safety checks ("safe!" / "unsafe!") are read from safety.txt and must be
    explicitly set by a super-user before and during the run. Any failed check
    aborts immediately without closing the roof (that is handled by end.py).
    """

    # ------------------------------------------------------------------ #
    # Guard: refuse to run if another imaging session is already active.  #
    # This can happen if the scheduler fires while a manual run is live,  #
    # or if a previous crash left the state file in a non-NONE state.     #
    # ------------------------------------------------------------------ #
    current = get_imaging_state()
    if current != ImagingState.NONE:
        msg = f"Imaging already in progress (state: {current.value}), aborting"
        _logger.warning(msg)
        social_server.post_social_message(msg)
        return

    # Claim the run immediately so no concurrent caller can slip through.
    set_imaging_state(ImagingState.ACTIVE)
    _logger.info("doit_cmd")

    # Kill any stale NINA or PWI4 processes before starting a fresh run.
    subprocess.run(
        [r"C:\Users\iriso\Documents\development\srt\scripts\kill_nina_pwi4.bat"],
        shell=True
    )

    # Persist imaging start time so end.py can compute the post-imaging summary.
    try:
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(_root, "imaging_start.txt"), "w") as _f:
            _f.write(datetime.now().isoformat())
    except Exception:
        pass
    cfg = config.data()

    # Path used for camera snapshots shown in Pushover notifications.
    inside_view = cfg["camera safety"]["scope_view"]

    # operand selects which NINA script to run for the main imaging session.
    operand = 1
    if len(words) > 2:
        operand = int(words[2])

    pushover.push_message(f"imaging! in mode {operand}")

    # Wait time reused in several places: 1 minute between state transitions.
    wait_time = 1 * 60
    utils.set_install_dir()

    # ------------------------------------------------------------------ #
    # Pre-flight: confirm the roof is physically closed before proceeding. #
    # Opening a roof that is already open would break the motor sequence.  #
    # ------------------------------------------------------------------ #
    parked, closed, open, mod_date = get_status_with_lights()
    if not closed:
        pushover.push_message("roof is not closed, stopping", inside_view)
        return

    # ------------------------------------------------------------------ #
    # Safety gate 1: check before the initial wait.                        #
    # The user must have previously issued "safe!" via the web chat.       #
    # ------------------------------------------------------------------ #
    pushover.push_message("Roof is closed, starting run in 1 min", inside_view)
    if not is_safe():
        pushover.push_message("not safe 1, stopping")
        return

    # One-minute pause gives the operator a last chance to abort via "unsafe!".
    time.sleep(wait_time)

    # ------------------------------------------------------------------ #
    # Safety gate 2: re-check after the wait in case conditions changed.   #
    # ------------------------------------------------------------------ #
    if not is_safe():
        pushover.push_message("not safe 2, stopping")
        return

    # Announce via Sonos + blinking lights so anyone in the observatory   #
    # knows the roof is about to move, then physically open it.           #
    announce_roof_movement("The roof will be opening in one minute")
    ok = open_roof_with_option(True)
    print("ok=", str(ok))
    if not ok:
        # Vision safety confirmed the roof did not open successfully.
        pushover.push_message("problem opening roof, stopping", inside_view)
        return

    # ------------------------------------------------------------------ #
    # Safety gate 3: check after roof is open, before starting NINA.      #
    # ------------------------------------------------------------------ #
    pushover.push_message("roof is open, starting imaging in 1 min", inside_view)
    time.sleep(wait_time)

    if not is_safe():
        pushover.push_message("not safe 3, stopping", inside_view)
        return

    if operand in (1, 2, 3):
        print("starting Nina")

        # ------------------------------------------------------------------ #
        # Prelude phase: on_nina.bat connects the mount, runs a meridian       #
        # flip if needed, performs an autofocus run, and slews to the target.  #
        # State is set to IN_PRELUDE so external observers / the bat file      #
        # know what phase we are in. The bat file signals completion by        #
        # calling: set_imaging_state.bat DONE_PRELUDE                          #
        # ------------------------------------------------------------------ #
        set_imaging_state(ImagingState.IN_PRELUDE)
        on_nina(None, None)

        # Poll until NINA signals that the prelude is done (via bat file).
        # on_nina uses subprocess.run so it blocks until the bat exits, but
        # the bat may exit before NINA finishes its internal sequence. Polling
        # here decouples us from that timing with a generous 10 minute timeout.
        _logger.info("Waiting for prelude to complete (state = DONE_PRELUDE)")
        prelude_timeout = 10 * 60  # 10 minutes max
        prelude_start = time.time()
        while get_imaging_state() != ImagingState.DONE_PRELUDE:
            if time.time() - prelude_start > prelude_timeout:
                pushover.push_message("Prelude timed out, stopping", inside_view)
                set_imaging_state(ImagingState.NONE)
                return
            time.sleep(30)

        pushover.push_message("prelude has finished", inside_view)

        # ------------------------------------------------------------------ #
        # Safety gate 4: check after prelude — conditions may have changed     #
        # during the (potentially long) prelude sequence.                      #
        # ------------------------------------------------------------------ #
        if not is_safe():
            pushover.push_message("not safe 4, stopping")
            return

        # Vision safety confirms scope is in the correct physical state before
        # we commit to launching the main imaging session.
        parked, closed, open, mod_date = get_status_with_lights()
        if not parked:
            # Mount did not park during prelude — something is wrong.
            pushover.push_message("scope is not parked, stopping", inside_view)
            return
        if closed:
            # Roof closed unexpectedly during prelude (wind? end sequence ran?).
            pushover.push_message("roof is closed, stopping", inside_view)
            return
        if not open:
            # Roof is neither fully closed nor fully open — ambiguous state.
            pushover.push_message("roof is not open, stopping", inside_view)
            return

        # ------------------------------------------------------------------ #
        # Main imaging phase: launch the NINA main sequence (non-blocking      #
        # Popen). image_cmd will reset state to NONE when doit_cmd returns.    #
        # For finer tracking, image_nina*.bat can call                         #
        # set_imaging_state.bat DONE_MAIN when the NINA sequence finishes.     #
        # ------------------------------------------------------------------ #
        set_imaging_state(ImagingState.IN_MAIN)
        if operand == 1:
            image_nina1(None, None)
        elif operand == 2:
            image_nina2(None, None)
        elif operand == 3:
            # No imaging — just home the scope and park via NINA.
            home_and_park(None, None)

        _logger.info("Waiting for imaging state to return to NONE")
        while get_imaging_state() != ImagingState.NONE:
            time.sleep(60)
        _logger.info("Imaging state is NONE — main phase complete")

        # Wait for NINA to fully exit before starting flats.
        _logger.info("Waiting for NINA process to exit")
        social_server.post_social_message("Waiting for NINA to exit before flats")
        _wait_for_nina_exit()
        do_flats()



def do_flats() -> None:
    """Run a flats sequence via NINA.

    Sequence:
        1. Power on the telescope mount.
        2. Set imaging state to IN_FLATS.
        3. Launch nina_flats.bat (non-blocking Popen).
        4. Poll imaging state every 30 seconds until DONE_FLATS.
        5. Power off the mount and return.
    """
    _logger.info("Begin flats sequence")
    social_server.post_social_message("Starting flats sequence")

    dev_map = asyncio.run(ku.make_discovery_map())
    asyncio.run(ku.kasa_do(dev_map, {"Telescope mount": 'on'}))
    _logger.info("Mount powered on")

    set_imaging_state(ImagingState.IN_FLATS)

    bat_path = os.path.join(_SCRIPTS_DIR, "nina_flats.bat")
    subprocess.Popen([bat_path], shell=True)
    _logger.info("nina_flats.bat launched")

    _logger.info("Waiting for flats to complete (state = DONE_FLATS)")
    while get_imaging_state() != ImagingState.DONE_FLATS:
        time.sleep(30)

    _logger.info("Flats complete")
    asyncio.run(ku.kasa_do(dev_map, {"Telescope mount": 'off'}))
    _logger.info("Mount powered off")
    social_server.post_social_message("Flats sequence complete")


def doflats_cmd(words: list[str], account: str) -> None:
    """Start a flat-frame capture sequence in a background thread (non-blocking)."""
    if is_imaging():
        pushover.push_message("Already imaging, cannot start flats")
        return
    def _run():
        set_imaging_state(ImagingState.ACTIVE)
        try:
            do_flats()
        finally:
            set_imaging_state(ImagingState.NONE)
    threading.Thread(target=_run, daemon=True).start()


def image_stats_cmd(words: list[str], account: str) -> None:
    """Post per-frame FWHM/eccentricity graph.

    Usage:
        stats        — frames since yesterday's sunset
        stats full   — all LIGHT frames for the same DSO (same root dir)
    """
    from astral import LocationInfo
    from astral.sun import sun
    from fits_processing import fitsfwhm

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]

    mode = words[2] if len(words) > 2 else "recent"

    if mode == "full":
        # Find the most recently modified FITS, then collect all LIGHT frames
        # under the same DSO root directory.
        all_fits = sorted(image_dir.rglob("*.fits"), key=lambda f: f.stat().st_mtime)
        if not all_fits:
            social_server.post_social_message("No FITS files found")
            return
        latest = all_fits[-1]
        # Walk up to find the DSO-level directory (parent of scope/date/LIGHT).
        # Structure: Targets/<DSO>/<scope>/<date>/LIGHT/<file>.fits
        dso_dir = latest.parent
        while dso_dir.parent != image_dir and dso_dir.parent != dso_dir:
            dso_dir = dso_dir.parent
        fits_files = sorted(dso_dir.rglob("*.fits"), key=lambda f: f.stat().st_mtime)
        social_server.post_social_message(
            f"Stats (full) for {dso_dir.name}: {len(fits_files)} frames"
        )
    else:
        loc = cfg["location"]
        city = LocationInfo(loc["city"], "USA", loc["timezone"], loc["latitude"], loc["longitude"])
        yesterday = datetime.now() - timedelta(days=1)
        sunset = sun(city.observer, date=yesterday)["sunset"]
        start_ts = sunset.replace(tzinfo=None).timestamp()
        social_server.post_social_message(
            f"Stats since {sunset.strftime('%Y-%m-%d %H:%M')}"
        )
        fits_files = sorted(
            (f for f in image_dir.rglob("*.fits") if f.stat().st_mtime >= start_ts),
            key=lambda f: f.stat().st_mtime,
        )

    if not fits_files:
        social_server.post_social_message("No FITS files found for the requested period")
        return

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])
    output_path = Path(os.path.join(scratch_dir, "stats_plot.jpg"))

    plot_path, frames_with_stars = fitsfwhm.save_stats_plot(
        fits_files, output_path, arcsec_per_pixel=arcsec_per_pixel
    )

    if frames_with_stars == 0:
        social_server.post_social_message(
            f"No stars detected in any of the {len(fits_files)} frames"
        )
        return

    social_server.post_social_message(
        f"FWHM & eccentricity — {frames_with_stars}/{len(fits_files)} frames",
        str(plot_path)
    )


if __name__ == "__main__":
    announce_roof_movement("The roof will be opening in 5 Minutes")