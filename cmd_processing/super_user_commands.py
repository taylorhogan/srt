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
    """Announce upcoming roof movement via Sonos."""
    try:
        sonos_utils.sonos_say(text, speaker_name, volume)
    except Exception as e:
        _logger.error("Sonos announcement failed: %s", e)


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
                MAX_ROOF_CHECKS = 5
                for attempt in range(MAX_ROOF_CHECKS):
                    parked, closed, is_open, mod_date = get_status_with_lights()
                    if is_open and parked:
                        return True
                    if attempt < MAX_ROOF_CHECKS - 1:
                        msg = f"Roof open not confirmed (attempt {attempt + 1}/{MAX_ROOF_CHECKS}), waiting 5 min"
                        social_server.post_social_message(msg)
                        _logger.warning(msg)
                        time.sleep(5 * 60)
                social_server.post_social_message(f"Roof could not be confirmed open after {MAX_ROOF_CHECKS} attempts, stopping")
                _logger.warning("Roof open check failed after %d attempts", MAX_ROOF_CHECKS)
                return False

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


def _kill_nina() -> None:
    """Forcefully terminate any running NINA.exe process.

    Uses taskkill /F so the kill is unconditional — no prompt, no grace period.
    Safe to call when NINA is not running (taskkill exits silently).
    """
    result = subprocess.run(
        ["taskkill", "/F", "/IM", "NINA.exe"],
        capture_output=True, text=True, shell=True,
    )
    if "SUCCESS" in result.stdout or "success" in result.stdout.lower():
        _logger.info("NINA.exe forcefully terminated")
        social_server.post_social_message("NINA process terminated")
    else:
        _logger.info("NINA.exe was not running (nothing to kill)")


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
    removed = instructions.normalize_and_deduplicate_db()
    if removed:
        social_server.post_social_message(f"Normalized DSO names, removed {removed} duplicate(s)")
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


_TODO_FILE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "todo.txt")


def todo_cmd(words: list[str], account: str) -> None:
    """Add an idea to the todo list, or display all current items.

    Usage:
        todo                    — show all items
        todo <text...>          — append a new item
    """
    if len(words) > 2:
        item = " ".join(words[2:])
        with open(_TODO_FILE, "a") as f:
            f.write(f"- {item}\n")
        social_server.post_social_message(f"Added: {item}")
    else:
        try:
            with open(_TODO_FILE, "r") as f:
                content = f.read().strip()
            if content:
                social_server.post_social_message(f"Todo list:\n{content}")
            else:
                social_server.post_social_message("Todo list is empty")
        except FileNotFoundError:
            social_server.post_social_message("Todo list is empty")


def log_cmd(words: list[str], account: str) -> None:
    """Show last N lines of iris.log. example: log 50"""
    n = 20
    if len(words) > 2:
        try:
            n = int(words[2])
        except ValueError:
            pass
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_path = os.path.join(_project_root, "iris.log")
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        tail = lines[-n:] if len(lines) >= n else lines
        social_server.post_social_message("".join(tail))
    except FileNotFoundError:
        social_server.post_social_message("iris.log not found")


def update_cmd(words: list[str], account: str) -> None:
    """Pull latest code from git and restart the server. example: update"""
    imaging_state = get_imaging_state()
    if imaging_state != ImagingState.NONE:
        social_server.post_social_message(
            f"Cannot update while imaging is active (state: {imaging_state.value}). "
            f"Issue stop! first, then retry."
        )
        return

    social_server.post_social_message("Update requested — pulling latest code…")

    def _do_update() -> None:
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        result = subprocess.run(
            ["git", "-C", _project_root, "pull"],
            capture_output=True, text=True,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > 500:
            output = "…" + output[-500:]
        if result.returncode != 0:
            social_server.post_social_message(f"git pull failed — aborting restart:\n{output}")
            return
        social_server.post_social_message(f"git pull succeeded:\n{output}")
        social_server.post_social_message("Restarting in 2 seconds…")
        time.sleep(2)
        os._exit(social_server.RESTART_EXIT_CODE)

    threading.Thread(target=_do_update, daemon=True).start()


def optics_cmd(words: list[str], account: str) -> None:
    """Post optical quality diagnostic plots for a FITS frame.

    Usage:
        optics              — latest frame of the last-imaged DSO
        optics <dso>        — latest frame of the named DSO
        optics * <n>        — frame n of the last-imaged DSO
        optics <dso> <n>    — frame n of the named DSO
    """
    threading.Thread(target=_optics_run, args=(words,), daemon=True).start()


def _optics_run(words: list[str]) -> None:
    from fits_processing import fitsfwhm, sky_brightness as sb

    cfg = config.data()
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]
    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])

    # Parse optional args: [dso_name] [frame_number]
    args = words[2:]
    frame_num: Optional[int] = None
    dso_arg: Optional[str] = None
    if args:
        if args[-1].isdigit():
            frame_num = int(args[-1])
            dso_arg = " ".join(args[:-1]).strip() or None
        else:
            dso_arg = " ".join(args).strip()
    if dso_arg == "*":
        dso_arg = None

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(fits_path: Path) -> Optional[Path]:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir: Optional[Path] = None

    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_fits = sorted(
            (f for f in image_dir.rglob("*.fits") if _is_light(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_fits:
            social_server.post_social_message("No FITS frames found")
            return
        dso_dir = _find_dso_dir(all_fits[-1])

    fits_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if _is_light(f)),
        key=lambda f: f.stat().st_mtime,
    ) if dso_dir else []

    if not fits_files:
        social_server.post_social_message("No LIGHT frames found")
        return

    total_frames = len(fits_files)
    if frame_num is None:
        fits_path = fits_files[-1]
        frame_idx = total_frames
    else:
        if frame_num < 1 or frame_num > total_frames:
            social_server.post_social_message(
                f"Frame {frame_num} out of range — {dso_dir.name} has {total_frames} frames"
            )
            return
        fits_path = fits_files[frame_num - 1]
        frame_idx = frame_num

    social_server.post_social_message(
        f"Optics for {dso_dir.name} frame {frame_idx}/{total_frames}"
    )

    metrics = fitsfwhm.compute_optical_metrics(fits_path, arcsec_per_pixel=arcsec_per_pixel)
    if metrics:
        metrics_table_path = Path(os.path.join(scratch_dir, "optical_metrics_table.jpg"))
        fitsfwhm.save_optical_metrics_table(metrics, metrics_table_path)
        social_server.post_social_message("Optical quality metrics", str(metrics_table_path))

    fwhm_heatmap_path = Path(os.path.join(scratch_dir, "fwhm_heatmap.jpg"))
    ecc_heatmap_path  = Path(os.path.join(scratch_dir, "ecc_heatmap.jpg"))
    fwhm_out, ecc_out = fitsfwhm.save_fwhm_heatmaps(
        fits_path, fwhm_heatmap_path, ecc_heatmap_path,
        arcsec_per_pixel=arcsec_per_pixel,
    )
    social_server.post_social_message("FWHM heatmap", str(fwhm_out))
    social_server.post_social_message("Eccentricity heatmap", str(ecc_out))

    dist_plot_path = Path(os.path.join(scratch_dir, "fwhm_vs_distance.jpg"))
    dist_out = fitsfwhm.save_fwhm_vs_distance(
        fits_path, dist_plot_path, arcsec_per_pixel=arcsec_per_pixel,
    )
    social_server.post_social_message("FWHM vs distance from centre", str(dist_out))

    angle_map_path = Path(os.path.join(scratch_dir, "ecc_angle_map.jpg"))
    angle_out = fitsfwhm.save_eccentricity_angle_map(
        fits_path, angle_map_path, arcsec_per_pixel=arcsec_per_pixel,
    )
    social_server.post_social_message("Elongation angle map", str(angle_out))

    sky_heatmap_path = Path(os.path.join(scratch_dir, "sky_heatmap.jpg"))
    sky_heatmap_out, sky_data = sb.save_sky_heatmap(
        fits_path, sky_heatmap_path, arcsec_per_pixel=arcsec_per_pixel,
    )
    if sky_data:
        social_server.post_social_message(sb.sky_summary_text(sky_data), str(sky_heatmap_out))


def active_cmd(words: list[str], account: str) -> None:
    """List all DSOs that have LIGHT frames, with sub counts per filter."""
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])

    if not image_dir.exists():
        social_server.post_social_message("Image directory not found")
        return

    try:
        dso_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir())
    except Exception as exc:
        social_server.post_social_message(f"active: scan failed — {exc}")
        return

    results: dict[str, dict[str, int]] = {}
    for dso_dir in dso_dirs:
        by_filter: dict[str, int] = {}
        for f in dso_dir.rglob("*.fits"):
            if f.parent.name.upper() != "LIGHT":
                continue
            filter_name = f.parent.parent.name
            by_filter[filter_name] = by_filter.get(filter_name, 0) + 1
        if by_filter:
            results[dso_dir.name] = by_filter

    if not results:
        social_server.post_social_message("No LIGHT frames found in image directory")
        return

    lines = []
    for dso, filters in sorted(results.items()):
        total = sum(filters.values())
        filter_str = "  ".join(f"{k}: {v}" for k, v in sorted(filters.items()))
        lines.append(f"{dso} ({total})  {filter_str}")

    social_server.post_social_message("\n".join(lines))


def drift_cmd(words: list[str], account: str) -> None:
    """Post ZScale-stretched difference images: first-k-frames stack vs golden (L filter only).

    Usage:
        drift           — L frames of the last-imaged DSO
        drift <dso>     — L frames of the named DSO
        drift *         — same as bare drift (last DSO)
    """
    threading.Thread(target=_drift_run, args=(words,), daemon=True).start()


def _drift_run(words: list[str]) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval
    from stacking import stacker

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scratch_dir = Path(os.path.join(_project_root, cfg["scratch"]["directory"]))

    dso_arg = " ".join(words[2:]).strip() if len(words) > 2 else None
    if dso_arg == "*":
        dso_arg = None

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(fits_path: Path) -> Optional[Path]:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir: Optional[Path] = None

    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_fits = sorted(
            (f for f in image_dir.rglob("*.fits") if _is_light(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_fits:
            social_server.post_social_message("No LIGHT frames found")
            return
        dso_dir = _find_dso_dir(all_fits[-1])

    fits_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if _is_light(f)),
        key=lambda f: f.stat().st_mtime,
    ) if dso_dir else []

    if not fits_files:
        social_server.post_social_message("No LIGHT frames found")
        return

    by_filter = stacker.group_by_filter(fits_files)
    l_files = next(
        (v for k, v in by_filter.items() if k.upper() in ("L", "LUMINANCE", "LUMA")),
        [],
    )
    if not l_files:
        social_server.post_social_message(f"{dso_dir.name}: no L filter frames found")
        return

    social_server.post_social_message(
        f"Drift for {dso_dir.name}: {len(l_files)} L frames — preparing…"
    )

    def _progress(msg: str) -> None:
        social_server.post_social_message(f"Drift [L]: {msg}")

    frames, accepted, fwhm_values = stacker._prepare_for_convergence(
        l_files, progress_cb=_progress,
    )
    n = len(frames)
    arr = np.stack(frames, axis=0)
    weights = stacker._fwhm_weights(fwhm_values, accepted)

    # Re-sort into acquisition (mtime) order so arr[:k] = first k frames acquired.
    sort_order = sorted(range(n), key=lambda i: accepted[i].stat().st_mtime)
    arr = arr[sort_order]
    weights = weights[sort_order]

    golden = stacker._combine_tile(arr, stacker.StackMethod.SIGMA_CLIP_FWHM, weights, sigma=3.0)
    nan_px = np.isnan(golden)
    if nan_px.any():
        golden[nan_px] = float(np.nanmedian(golden))

    from concurrent.futures import ThreadPoolExecutor

    counts = stacker._fib_counts(n)

    def _compute_diff_k(k: int) -> tuple[int, np.ndarray]:
        if k == n:
            return k, np.zeros_like(golden)
        sub_w = weights[:k].copy()
        sub_w /= sub_w.sum()
        nan_mask = np.isnan(arr[:k])
        safe = np.where(nan_mask, 0.0, arr[:k])
        contrib = (~nan_mask).astype(np.float32) * sub_w[:, None, None]
        numer = np.sum(safe * sub_w[:, None, None], axis=0)
        denom = contrib.sum(axis=0)
        subset_mean = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), golden)
        return k, np.abs(subset_mean - golden)

    with ThreadPoolExecutor() as pool:
        diffs = dict(pool.map(_compute_diff_k, counts))

    first_diff = diffs[counts[0]]
    try:
        _, vmax = ZScaleInterval().get_limits(first_diff)
    except Exception:
        vmax = float(np.nanpercentile(first_diff, 99))
    vmin = 0.0

    def _render_k(k: int) -> tuple[int, Path]:
        diff = diffs[k]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(diff, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{dso_dir.name}  L  {k}/{n} frames vs golden", fontsize=10)
        fig.tight_layout(pad=0.5)
        out_path = scratch_dir / f"drift_L_{k:04d}of{n:04d}.jpg"
        fig.savefig(out_path, format="jpeg", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return k, out_path

    with ThreadPoolExecutor() as pool:
        results = dict(pool.map(_render_k, counts))

    for k in counts:
        social_server.post_social_message(f"L  {k}/{n} frames vs golden", str(results[k]))


def stack_cmd(words: list[str], account: str) -> None:
    """Stack all LIGHT frames of a DSO (per filter) and post each as a JPEG.

    Usage:
        stack                 — all filters of the last-imaged DSO
        stack <dso>           — all filters of the named DSO
        stack <dso> <filter>  — only the named filter (e.g. stack m31 ha)
    """
    threading.Thread(target=_stack_run, args=(words,), daemon=True).start()


def _stack_run(words: list[str]) -> None:
    from stacking import stacker
    from fits_processing import fitsfwhm
    from astropy.io import fits as _fits

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scratch_dir = Path(os.path.join(_project_root, cfg["scratch"]["directory"]))

    cal = cfg.get("calibration", {})
    _bias_paths = stacker._collect_fits(Path(cal["bias_dir"])) if cal.get("bias_dir") else []
    _dark_paths = stacker._collect_fits(Path(cal["dark_dir"])) if cal.get("dark_dir") else []
    _flat_dir = Path(cal["flat_dir"]) if cal.get("flat_dir") else None
    _flat_dirs = {k: Path(v) for k, v in cal.get("flat_dirs", {}).items()} or None

    # words[0] is the bot mention, words[1] is "stack"; remainder is dso (+ optional filter)
    extra = words[2:] if len(words) > 2 else []

    # Filter aliases: last token is treated as a filter if it matches a known short name.
    _FILTER_ALIASES = {
        "L": {"L", "LUMINANCE", "LUMA"},
        "R": {"R", "RED"},
        "G": {"G", "GREEN"},
        "B": {"B", "BLUE"},
        "HA": {"HA", "H-ALPHA", "HALPHA"},
        "OIII": {"OIII", "O3"},
        "SII": {"SII", "S2"},
    }

    def _canonical_filter(token: str) -> Optional[str]:
        t = token.upper().replace("-", "").replace("_", "")
        for canon, aliases in _FILTER_ALIASES.items():
            if t in {a.replace("-", "").replace("_", "") for a in aliases}:
                return canon
        return None

    filter_arg: Optional[str] = None
    if extra and _canonical_filter(extra[-1]) is not None:
        filter_arg = _canonical_filter(extra[-1])
        dso_arg = " ".join(extra[:-1]).strip() or None
    else:
        dso_arg = " ".join(extra).strip() or None

    if dso_arg == "*":
        dso_arg = None

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(fits_path: Path) -> Optional[Path]:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir: Optional[Path] = None
    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_fits = sorted(
            (f for f in image_dir.rglob("*.fits") if _is_light(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_fits:
            social_server.post_social_message("No LIGHT frames found")
            return
        dso_dir = _find_dso_dir(all_fits[-1])

    fits_files = sorted(f for f in dso_dir.rglob("*.fits") if _is_light(f)) if dso_dir else []
    if not fits_files:
        social_server.post_social_message("No LIGHT frames found")
        return

    groups = stacker.group_by_filter(fits_files)

    if filter_arg is not None:
        matching = {
            name: paths for name, paths in groups.items()
            if _canonical_filter(name) == filter_arg
        }
        if not matching:
            social_server.post_social_message(
                f"{dso_dir.name}: no {filter_arg} frames found"
            )
            return
        groups = matching

    social_server.post_social_message(
        f"Stacking {dso_dir.name}: "
        + ", ".join(f"{name}={len(paths)}" for name, paths in sorted(groups.items()))
    )

    for filter_name, paths in sorted(groups.items()):
        def _progress(msg: str, _fn: str = filter_name, _dso: str = dso_dir.name) -> None:
            social_server.post_social_message(f"{_dso} {_fn}: {msg}")

        try:
            flat_paths = stacker._resolve_flat_paths(filter_name, _flat_dir, _flat_dirs)
            result, info = stacker.stack(
                paths,
                method=stacker.StackMethod.SIGMA_CLIP_FWHM,
                bias_paths=_bias_paths,
                dark_paths=_dark_paths,
                flat_paths=flat_paths,
                max_fwhm_multiplier=1.5,
                progress_cb=_progress,
            )
        except Exception as exc:
            social_server.post_social_message(
                f"{dso_dir.name} {filter_name}: stack failed — {exc}"
            )
            continue

        safe_filter = filter_name.replace(" ", "_")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        fits_path = scratch_dir / f"stack_{dso_dir.name}_{safe_filter}.fits"
        _fits.PrimaryHDU(result.astype("float32")).writeto(fits_path, overwrite=True)

        try:
            _, fwhm_arcsec, star_count, ecc = fitsfwhm.calculate_fwhm(
                fits_path, arcsec_per_pixel=arcsec_per_pixel,
            )
        except Exception as exc:
            _logger.warning("stack: FWHM measurement failed for %s: %s", fits_path.name, exc)
            fwhm_arcsec, star_count, ecc = 0.0, 0, 0.0

        if star_count > 0:
            metrics = f"FWHM {fwhm_arcsec:.2f}″   ecc {ecc:.2f}   stars {star_count}"
        else:
            metrics = "no stars detected"

        out_path = scratch_dir / f"stack_{dso_dir.name}_{safe_filter}.jpg"
        jpg = stacker._save_jpg(
            result, out_path,
            title=(
                f"{dso_dir.name}  {filter_name}  {info['n_frames']} frames "
                f"({info['method']})\n{metrics}"
            ),
        )
        social_server.post_social_message(
            f"{dso_dir.name} {filter_name}: {info['n_frames']} frames stacked — {metrics}",
            str(jpg),
        )


_BAD_FWHM_ARCSEC = 3.5
_BAD_ECC = 0.45


def bad_cmd(words: list[str], account: str) -> None:
    """Rename LIGHT frames that fail quality thresholds to .bad.

    Failing == 0 stars detected OR FWHM > 3.5″ OR eccentricity > 0.45.
    Renames `<frame>.fits` to `<frame>.fits.bad` so they're skipped by
    the stacker (which only globs *.fits).

    Reuses the frame_stats.json cache populated by the `stats` command;
    any uncached frames are measured on the spot and written back to the
    cache.

    Usage:
        bad                 — dry-run on last-imaged DSO
        bad <dso>           — dry-run on named DSO
        bad <dso> go        — actually rename (default is dry-run)
        bad go              — dry-run on last DSO (use `bad * go` to rename)
    """
    threading.Thread(target=_bad_run, args=(words,), daemon=True).start()


def _bad_run(words: list[str]) -> None:
    import json as _json
    import warnings as _warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from fits_processing import fitsfwhm

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]

    # Parse args: optional <dso> and optional trailing "go"
    extra = list(words[2:]) if len(words) > 2 else []
    do_rename = False
    if extra and extra[-1].lower() == "go":
        do_rename = True
        extra = extra[:-1]
    dso_arg = " ".join(extra).strip() or None
    if dso_arg == "*":
        dso_arg = None

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(fits_path: Path) -> Optional[Path]:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_fits = sorted(
            (f for f in image_dir.rglob("*.fits") if _is_light(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_fits:
            social_server.post_social_message("No LIGHT frames found")
            return
        dso_dir = _find_dso_dir(all_fits[-1])

    fits_files = sorted(f for f in dso_dir.rglob("*.fits") if _is_light(f)) if dso_dir else []
    if not fits_files:
        social_server.post_social_message(f"{dso_dir.name}: no LIGHT frames found")
        return

    # Load existing stats cache (path → entry)
    cache_path = dso_dir / "frame_stats.json"
    cached_by_path: dict[str, dict] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                existing = _json.load(f)
            if isinstance(existing, list):
                for entry in existing:
                    if "path" in entry:
                        cached_by_path[str(Path(entry["path"]))] = entry
        except Exception:
            pass

    def _analyse(fits_path: Path) -> dict:
        from astropy.io import fits as _fits
        filter_name = "Unknown"
        try:
            with _fits.open(fits_path) as hdul:
                filter_name = str(hdul[0].header.get("FILTER", "Unknown")).strip()
        except Exception:
            pass
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _, fwhm_arcsec, star_count, ecc = fitsfwhm.calculate_fwhm(
                    fits_path, arcsec_per_pixel=arcsec_per_pixel
                )
            return {
                "path": str(fits_path),
                "filter": filter_name,
                "fwhm_arcsec": round(float(fwhm_arcsec), 3) if star_count else None,
                "eccentricity": round(float(ecc), 3) if star_count else None,
                "star_count": int(star_count),
            }
        except Exception as exc:
            _logger.warning("bad: could not analyse %s: %s", fits_path.name, exc)
            return {"path": str(fits_path), "filter": filter_name,
                    "fwhm_arcsec": None, "eccentricity": None, "star_count": 0}

    need_analysis = [f for f in fits_files if str(f) not in cached_by_path]
    cached_count = len(fits_files) - len(need_analysis)
    social_server.post_social_message(
        f"bad {dso_dir.name}: {len(fits_files)} frames "
        f"({cached_count} cached, {len(need_analysis)} to analyse)"
    )

    if need_analysis:
        new_entries: list[dict] = [None] * len(need_analysis)
        max_workers = min(8, len(need_analysis))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_analyse, f): i for i, f in enumerate(need_analysis)}
            for fut in as_completed(future_map):
                new_entries[future_map[fut]] = fut.result()
        for entry in new_entries:
            cached_by_path[str(Path(entry["path"]))] = entry
        tmp = cache_path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                _json.dump(list(cached_by_path.values()), f, default=str, indent=2)
            tmp.replace(cache_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # Decide which frames are bad. Legacy `stats` cache entries don't carry
    # `star_count`; in those rows, `eccentricity is None` is the project's
    # marker for "no stars detected" (see _image_stats_run).
    bad: list[tuple[Path, str]] = []  # (path, reason)
    for f in fits_files:
        entry = cached_by_path.get(str(f), {})
        fwhm = entry.get("fwhm_arcsec")
        ecc = entry.get("eccentricity")
        if "star_count" in entry:
            star_count = entry["star_count"]
        else:
            star_count = 0 if ecc is None else None

        if star_count == 0:
            bad.append((f, "0 stars"))
        elif fwhm is not None and fwhm > _BAD_FWHM_ARCSEC:
            bad.append((f, f"FWHM {fwhm:.2f}\" > {_BAD_FWHM_ARCSEC}\""))
        elif ecc is not None and ecc > _BAD_ECC:
            bad.append((f, f"ecc {ecc:.2f} > {_BAD_ECC}"))

    from stacking import stacker as _stacker

    def _filter_for(f: Path) -> str:
        entry = cached_by_path.get(str(f), {})
        name = entry.get("filter")
        if name:
            return str(name).strip()
        return _stacker.read_filter(f)

    # Per-filter totals before/after the would-be rename
    total_by_filter: dict[str, int] = {}
    bad_by_filter: dict[str, int] = {}
    bad_set = {p for p, _ in bad}
    for f in fits_files:
        fn = _filter_for(f) or "Unknown"
        total_by_filter[fn] = total_by_filter.get(fn, 0) + 1
        if f in bad_set:
            bad_by_filter[fn] = bad_by_filter.get(fn, 0) + 1

    def _remaining_summary() -> str:
        parts = []
        for fn in sorted(total_by_filter):
            remaining = total_by_filter[fn] - bad_by_filter.get(fn, 0)
            parts.append(f"{fn}: {remaining}/{total_by_filter[fn]}")
        return "Remaining frames per filter — " + "  ".join(parts)

    if not bad:
        social_server.post_social_message(
            f"{dso_dir.name}: all {len(fits_files)} frames pass thresholds "
            f"(FWHM≤{_BAD_FWHM_ARCSEC}\", ecc≤{_BAD_ECC}, stars>0)\n"
            f"{_remaining_summary()}"
        )
        return

    verb = "Renamed" if do_rename else "Would rename"
    lines = [f"{verb} {len(bad)}/{len(fits_files)} frame(s) in {dso_dir.name}:"]
    for f, reason in bad[:40]:
        lines.append(f"  {f.name}  — {reason}")
    if len(bad) > 40:
        lines.append(f"  …and {len(bad) - 40} more")
    lines.append(_remaining_summary())
    if not do_rename:
        lines.append(f"Re-run as `bad {dso_arg or '*'} go` to actually rename.")
    social_server.post_social_message("\n".join(lines))

    if not do_rename:
        return

    failed: list[tuple[Path, str]] = []
    renamed_paths: list[str] = []
    for f, _reason in bad:
        target = f.with_suffix(f.suffix + ".bad")
        try:
            f.rename(target)
            renamed_paths.append(str(f))
        except Exception as exc:
            failed.append((f, str(exc)))

    # Drop the renamed entries from the cache so stats / bad / anything else
    # that reads frame_stats.json doesn't carry orphan rows pointing at
    # paths that no longer exist.
    if renamed_paths:
        for p in renamed_paths:
            cached_by_path.pop(p, None)
        tmp = cache_path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                _json.dump(list(cached_by_path.values()), f, default=str, indent=2)
            tmp.replace(cache_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    if failed:
        social_server.post_social_message(
            f"Rename failed for {len(failed)} frame(s); first error: "
            f"{failed[0][0].name} — {failed[0][1]}"
        )


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
        "todo": todo_cmd,
        "active": active_cmd,
        "stats": image_stats_cmd,
        "snr": snr_cmd,
        "transit": transit_cmd,
        "log": log_cmd,
        "update": update_cmd,
        "optics": optics_cmd,
        "drift": drift_cmd,
        "stack": stack_cmd,
        "bad": bad_cmd,
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
        from fits_processing import frame_watcher
        cfg = config.data()
        frame_watcher.start(
            Path(cfg["nina"]["image_dir"]),
            cfg["nina"]["arc_sec_per_pixel"],
        )
        def _run():
            try:
                doit_cmd(words, account)
            finally:
                frame_watcher.stop()
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
        IN_MAIN → NONE   (written by the NINA main sequence via an External Script step calling set_imaging_state.bat NONE)

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

        # Kill NINA before starting flats so there is no leftover process
        # from the main sequence holding a lock or confusing the new instance.
        _logger.info("Terminating NINA before starting flats")
        social_server.post_social_message("Terminating NINA before flats")
        _kill_nina()
        do_flats()

        instr = instructions.get_dso_object_tonight()
        eon_dso = instr.get("dso") if instr else None
        eon_words = ["snr", eon_dso] if eon_dso else ["snr"]
        social_server.post_social_message(
            f"End of night: SNR analysis for {eon_dso or 'most recent DSO'}…"
        )
        try:
            _snr_run(eon_words)
        except Exception:
            _logger.exception("End-of-night SNR failed")


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

    # Visually verify mount is parked and roof is closed before flats.
    # Only checks — does not move the scope or roof.
    _logger.info("Visual safety check before flats")
    parked, closed, is_open, mod_date = get_status_with_lights()

    if parked:
        _logger.info("Visual check: mount is parked")
    else:
        social_server.post_social_message("WARNING: Mount does not appear parked before flats")
        _logger.warning("Visual check: mount does not appear parked before flats")
    if closed:
        _logger.info("Visual check: roof is closed")
    else:
        social_server.post_social_message("WARNING: Roof does not appear closed before flats")
        _logger.warning("Visual check: roof does not appear closed before flats")

    set_imaging_state(ImagingState.IN_FLATS)

    bat_path = os.path.join(_SCRIPTS_DIR, "nina_flats.bat")
    subprocess.Popen([bat_path], shell=True)
    _logger.info("nina_flats.bat launched")

    _logger.info("Waiting for flats to complete (state = DONE_FLATS, timeout 30 min)")
    deadline = time.time() + 1800  # 30 minutes
    while get_imaging_state() != ImagingState.DONE_FLATS:
        if time.time() > deadline:
            _logger.warning("Flats timed out after 30 minutes — terminating NINA")
            social_server.post_social_message("WARNING: Flats timed out after 30 min, terminating NINA")
            _kill_nina()
            break
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


def snr_cmd(words: list[str], account: str) -> None:
    """Post stack-convergence curve in a background thread (non-blocking)."""
    threading.Thread(target=_snr_run, args=(words,), daemon=True).start()


def _snr_run(words: list[str]) -> None:
    """Worker for snr_cmd.

    Usage:
        snr           — convergence curve for the DSO currently being / last imaged
        snr <dso>     — convergence curve for the named DSO

    Loads all LIGHT frames, groups by filter, and posts one convergence plot per
    filter showing normalised RMSE vs Fibonacci-spaced frame counts.
    """
    from stacking import stacker

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])

    dso_arg = " ".join(words[2:]).strip() if len(words) > 2 else None

    social_server.post_social_message("Convergence: scanning for FITS files…")

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(fits_path: Path) -> Optional[Path]:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir: Optional[Path] = None

    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_fits = sorted(
            (f for f in image_dir.rglob("*.fits") if _is_light(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_fits:
            social_server.post_social_message("No LIGHT frames found")
            return
        dso_dir = _find_dso_dir(all_fits[-1])

    fits_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if _is_light(f)),
        key=lambda f: f.stat().st_mtime,
    ) if dso_dir else []

    if not fits_files:
        social_server.post_social_message("No LIGHT frames found for the requested target")
        return

    social_server.post_social_message(
        f"Convergence for {dso_dir.name}: {len(fits_files)} light frames — computing…"
    )

    by_filter = stacker.group_by_filter(fits_files)

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])

    from fits_processing import convergence as _conv

    from datetime import date as _date

    saved: dict[str, dict] = {}
    for fn, paths in by_filter.items():
        out = Path(scratch_dir) / f"convergence_{fn}.jpg"
        gold = Path(scratch_dir) / f"golden_{fn}.jpg"
        def _progress(msg: str, _fn: str = fn) -> None:
            social_server.post_social_message(f"Convergence [{_fn}]: {msg}")
        try:
            _, _, slope_pct, final_rmse_pct = stacker.convergence_curve(
                paths,
                filter_name=fn,
                output_path=out,
                golden_output_path=gold,
                progress_cb=_progress,
            )
        except Exception as exc:
            social_server.post_social_message(f"Convergence [{fn}]: failed — {exc}")
            continue
        saved[fn] = {
            "tail_slope_pct": round(slope_pct, 6),
            "final_rmse_pct": round(final_rmse_pct, 4),
            "frame_count": len(paths),
            "updated": _date.today().isoformat(),
        }
        social_server.post_social_message(
            f"Stack convergence vs golden — {fn}  ({len(paths)} frames)  slope {slope_pct:+.4f}%/frame  RMSE {final_rmse_pct:.2f}%",
            str(out),
        )
        social_server.post_social_message(
            f"Golden stack — {fn}  ({len(paths)} frames)",
            str(gold),
        )

    if saved and dso_dir is not None:
        try:
            _conv.save_convergence(dso_dir.name, saved)
        except Exception:
            _logger.exception("_snr_run: failed to save convergence for %s", dso_dir.name)

        slope_threshold = float(cfg.get("convergence", {}).get("tail_slope_threshold", 0.40))
        rmse_threshold  = float(cfg.get("convergence", {}).get("rmse_done_threshold",   5.0))
        min_frames      = int(cfg.get("convergence", {}).get("min_frames_per_filter",    30))

        lines: list[str] = []
        for fn, info in sorted(saved.items()):
            n          = info["frame_count"]
            slope_abs  = abs(info["tail_slope_pct"])
            rmse       = info.get("final_rmse_pct")

            needed_min = max(0, min_frames - n)
            needed_slope = (
                max(0, int(n * (slope_abs / slope_threshold - 1)))
                if slope_threshold > 0 and n > 0 else 0
            )
            rmse_over = rmse is not None and rmse > rmse_threshold

            if needed_min > 0:
                lines.append(
                    f"  {fn}: {n} frames — need {needed_min} more to reach minimum ({min_frames} required)"
                )
            elif needed_slope > 0 and rmse_over:
                lines.append(
                    f"  {fn}: {n} frames — need ~{needed_slope} more"
                    f" (slope {slope_abs:.2f}% > {slope_threshold:.2f}% target,"
                    f" RMSE {rmse:.1f}% > {rmse_threshold:.1f}% limit)"
                )
            elif needed_slope > 0:
                lines.append(
                    f"  {fn}: {n} frames — need ~{needed_slope} more"
                    f" (slope {slope_abs:.2f}% > {slope_threshold:.2f}% target)"
                )
            elif rmse_over:
                lines.append(
                    f"  {fn}: {n} frames — slope converged but RMSE {rmse:.1f}% still above {rmse_threshold:.1f}% limit"
                )
            else:
                lines.append(f"  {fn}: done ✓  ({n} frames, slope {slope_abs:.2f}%, RMSE {rmse:.1f}%)")

        social_server.post_social_message(
            f"{dso_dir.name} — exposures needed:\n" + "\n".join(lines)
        )


def transit_cmd(words: list[str], account: str) -> None:
    """Search saved subs for transit-like dips. Background thread (non-blocking).

    Usage: transit <dso> <filter>
    """
    threading.Thread(target=_transit_run, args=(words,), daemon=True).start()


def _transit_run(words: list[str]) -> None:
    """Worker for transit_cmd."""
    if len(words) < 4:
        social_server.post_social_message("Usage: transit <dso> <filter>")
        return

    filter_name = words[-1].strip()
    dso_arg = " ".join(words[2:-1]).strip()
    if not dso_arg or not filter_name:
        social_server.post_social_message("Usage: transit <dso> <filter>")
        return

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = Path(os.path.join(_project_root, cfg["scratch"]["directory"]))
    out_path = scratch_dir / f"transit_{dso_arg.replace(' ', '_')}_{filter_name}.jpg"

    def _progress(msg: str) -> None:
        social_server.post_social_message(f"Transit [{dso_arg}/{filter_name}]: {msg}")

    social_server.post_social_message(
        f"Transit search for {dso_arg} [{filter_name}]: starting…"
    )

    try:
        from transit_search import transit as _ts
        entry = _ts.run_transit_search(
            dso_name=dso_arg,
            filter_name=filter_name,
            image_dir=image_dir,
            output_plot_path=out_path,
            progress_cb=_progress,
        )
    except Exception as exc:
        _logger.exception("_transit_run: failed")
        social_server.post_social_message(f"Transit [{dso_arg}/{filter_name}]: failed — {exc}")
        return

    cands = entry.get("candidates", [])
    if cands:
        top = cands[0]
        period = top.get("bls_period_d")
        period_str = f"P={period:.3f}d depth={top.get('bls_depth', 0):.3f}" if period else "no BLS"
        summary = (
            f"Transit [{dso_arg}/{filter_name}]: {entry['n_stars']} stars, "
            f"{entry['frame_count']} frames over {entry['baseline_days']:.1f}d. "
            f"Top: ({top['x']:.0f},{top['y']:.0f}) dip σ={top['max_dip_sigma']:.1f}  {period_str}. "
            f"Manual verification required."
        )
    else:
        summary = (
            f"Transit [{dso_arg}/{filter_name}]: complete, no candidates above threshold."
        )

    if out_path.exists():
        social_server.post_social_message(summary, str(out_path))
    else:
        social_server.post_social_message(summary)


def image_stats_cmd(words: list[str], account: str) -> None:
    """Post per-frame FWHM/eccentricity graph in a background thread (non-blocking)."""
    threading.Thread(target=_image_stats_run, args=(words, account), daemon=True).start()


def _image_stats_run(words: list[str], account: str) -> None:
    """Worker for image_stats_cmd.

    Usage:
        stats           — all LIGHT frames for the DSO currently being imaged
        stats <dso>     — all LIGHT frames for the named DSO

    For each FITS file in scope the path is looked up in <dso_dir>/frame_stats.json.
    Cached entries are used as-is; only files missing from the cache are opened and
    analysed.  Newly analysed frames are written back to the cache so subsequent
    runs skip them too.
    """
    import json as _json
    import warnings as _warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from fits_processing import fitsfwhm
    from fits_processing import sky_brightness as sb

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]

    dso_arg = " ".join(words[2:]).strip() if len(words) > 2 else None

    social_server.post_social_message("Stats: scanning for FITS files…")

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(fits_path: Path) -> Optional[Path]:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [
            d for d in image_dir.iterdir()
            if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Multiple matches — pick the one with the most recent LIGHT frame
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits") if _is_light(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir: Optional[Path] = None

    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_fits = sorted(
            (f for f in image_dir.rglob("*.fits") if _is_light(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_fits:
            social_server.post_social_message("No LIGHT frames found")
            return
        dso_dir = _find_dso_dir(all_fits[-1])

    fits_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if _is_light(f)),
        key=lambda f: f.stat().st_mtime,
    ) if dso_dir else []
    social_server.post_social_message(
        f"Stats for {dso_dir.name if dso_dir else '?'}: "
        f"{len(fits_files)} light frames found…"
    )

    if not fits_files:
        social_server.post_social_message("No LIGHT frames found for the requested period")
        return

    # ── Load existing cache (keyed by normalised path string) ──────────────
    cache_path: Optional[Path] = (dso_dir / "frame_stats.json") if dso_dir else None
    cached_by_path: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            with open(cache_path) as f:
                existing = _json.load(f)
            if isinstance(existing, list):
                for entry in existing:
                    if "path" in entry:
                        cached_by_path[str(Path(entry["path"]))] = entry
        except Exception:
            pass

    # ── Analyse only files not already in the cache ─────────────────────────
    def _analyse_fits(fits_path: Path) -> dict:
        from astropy.io import fits as _fits
        try:
            with _fits.open(fits_path) as hdul:
                hdr = hdul[0].header
            date_obs = hdr.get("DATE-OBS")
            try:
                obs_dt = datetime.fromisoformat(date_obs.rstrip("Z")) if date_obs else None
            except (ValueError, AttributeError):
                obs_dt = None
            if obs_dt is None:
                obs_dt = datetime.fromtimestamp(fits_path.stat().st_mtime)
            filter_name = str(hdr.get("FILTER", "Unknown")).strip()
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _, fwhm_arcsec, star_count, ecc = fitsfwhm.calculate_fwhm(
                    fits_path, arcsec_per_pixel=arcsec_per_pixel
                )
            if star_count == 0:
                hfr = hdr.get("HFR")
                fwhm_arcsec = float(hfr) * 2.0 * arcsec_per_pixel if hfr else None
                ecc = None
            else:
                fwhm_arcsec = round(float(fwhm_arcsec), 3)
                ecc = round(float(ecc), 3)
            sky = sb.measure_sky(fits_path, arcsec_per_pixel=arcsec_per_pixel)
            return {
                "path":            str(fits_path),
                "time":            obs_dt.isoformat(),
                "filter":          filter_name,
                "fwhm_arcsec":     fwhm_arcsec,
                "eccentricity":    ecc,
                "star_count":      int(star_count),
                "sky_adu_per_s":   round(sky["sky_adu_per_s"], 2)       if sky else None,
                "sky_mag_arcsec2": round(sky["sky_mag_arcsec2"], 2)
                                   if sky and sky.get("sky_mag_arcsec2") is not None else None,
            }
        except Exception as exc:
            _logger.warning("stats: could not analyse %s: %s", fits_path.name, exc)
            return {"path": str(fits_path), "time": "", "filter": "Unknown",
                    "fwhm_arcsec": None, "eccentricity": None, "star_count": 0,
                    "sky_adu_per_s": None, "sky_mag_arcsec2": None}

    need_analysis = [f for f in fits_files if str(f) not in cached_by_path]
    cached_count  = len(fits_files) - len(need_analysis)

    if need_analysis:
        social_server.post_social_message(
            f"{cached_count} cached, {len(need_analysis)} new — analysing…"
        )
        new_entries: list[dict] = [None] * len(need_analysis)
        max_workers = min(8, len(need_analysis))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_analyse_fits, f): i for i, f in enumerate(need_analysis)}
            for fut in as_completed(future_map):
                new_entries[future_map[fut]] = fut.result()

        # Merge new entries into the cache and save
        for entry in new_entries:
            cached_by_path[str(Path(entry["path"]))] = entry
        if cache_path:
            tmp = cache_path.with_suffix(".tmp")
            try:
                with open(tmp, "w") as f:
                    _json.dump(list(cached_by_path.values()), f, default=str, indent=2)
                tmp.replace(cache_path)
            except Exception:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        social_server.post_social_message(f"All {cached_count} frames served from cache")

    # Build the ordered frame list matching the original fits_files order
    frames = [cached_by_path[str(f)] for f in fits_files if str(f) in cached_by_path]

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])
    output_path = Path(os.path.join(scratch_dir, "stats_plot.jpg"))

    plot_path, frames_with_stars = fitsfwhm.save_stats_plot_from_cache(frames, output_path)

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