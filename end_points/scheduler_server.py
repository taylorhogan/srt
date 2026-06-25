"""
scheduler_server.py — Nightly imaging state machine for the Iris observatory.

Overview
--------
This module is one of the two long-running processes that make up SRT
(started together by ``end_points/start_srt.py``).  Its sole job is to
drive the observatory through a repeating daily cycle:

    WAITING_FOR_NOON
        → NOON_CHECK
        → WAITING_FOR_PRE_SUNSET
        → PRE_SUNSET_CHECK
        → IMAGING
        → WAITING_FOR_NOON  (next day)

At each gate the scheduler decides whether conditions justify imaging
tonight, posts a status update to the web chat, and advances (or retreats)
to the next state.

State descriptions
------------------
WAITING_FOR_NOON
    Sleeps in 30-second increments until the next calendar noon.

NOON_CHECK
    Recalculates DSO visibility for all queued targets, picks the best
    one for tonight, posts the imaging grid and plan to the web chat, and
    decides whether tonight is worth imaging (≥ _MIN_GOOD_HOURS).

WAITING_FOR_PRE_SUNSET
    Sleeps until one hour before today's sunset.

PRE_SUNSET_CHECK
    Re-evaluates conditions closer to showtime.  If good, generates the
    N.I.N.A sequence and moves to IMAGING; otherwise waits for noon.

IMAGING
    Fires ``super_user_commands.image_cmd`` (which is non-blocking —
    it calls ``subprocess.Popen``).  Then polls ``get_imaging_state()``
    every 60 seconds until ``end.py`` resets the state to
    ``ImagingState.NONE``, signalling that the hardware has shut down.

WAITING_FOR_BOOT
    A one-minute recovery pause entered after any unhandled exception
    before falling back to WAITING_FOR_NOON.

Inter-process communication
---------------------------
- **MQTT** (``paho-mqtt``): the social server can query the current
  observatory state by publishing to ``utils.topic_to_sched``; this
  module replies with a JSON payload on ``iris/from_sched``.
- **imaging.txt** (file on disk, written by ``super_user_commands``):
  shared state between this process, ``doit_cmd`` / ``end.py``, and the
  N.I.N.A Windows batch scripts.  The scheduler polls this file while
  waiting for an imaging run to complete.
- **Web chat**: all human-readable status updates are posted via
  ``social_server.post_social_message`` (routed to the web chat, with
  optional Mastodon mirroring).
- **Pushover**: not used directly here; called downstream by
  ``super_user_commands.doit_cmd`` during the actual imaging run.

Dependencies (non-stdlib)
-------------------------
- ``iris_astronomy`` — DSO visibility, calendar, weather / sunset times.
- ``control.instructions`` — JSON-backed DSO request queue.
- ``cmd_processing.social_server`` — web chat server and message posting.
- ``cmd_processing.super_user_commands`` — imaging state enum + image_cmd.
- ``nina_gen.nina_sequence_gen`` — N.I.N.A sequence generator.
- ``utils`` — MQTT helpers, logging.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path

import numpy as np
from prefect import flow, task

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from iris_astronomy import astro_dso_visibility, obs_calendar, weather
from iris_astronomy.astro_dso_visibility import best_object_tonight
from iris_astronomy.weather import get_sunrise_sunset
from configs import config
from control import instructions
from cmd_processing import social_server
from utils import utils, pushover
from cmd_processing import super_user_commands
from nina_gen import nina_sequence_gen

CFG = config.data()
LOGGER = utils.set_logger()
CFG["logger"]["logging"] = LOGGER

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Minimum number of "good" imaging hours required for the scheduler to
# commit to imaging tonight.  A night with fewer than this many hours
# above the horizon (at acceptable air mass) is skipped.
_MIN_GOOD_HOURS = 3

# MQTT client reference; assigned in main() after connection.
client = None

# Mutable dict broadcast over MQTT when the social server polls us.
# Keys are kept stable so callers can parse the JSON reliably.
observatory_state = {
    "state": "Unknown",
    "dso": "Unknown",
    "will image tonight": "Unknown"
}


class State(Enum):
    """States of the nightly scheduling state machine.

    Transitions::

        WAITING_FOR_BOOT  ──► WAITING_FOR_NOON
        WAITING_FOR_NOON  ──► NOON_CHECK
        NOON_CHECK        ──► WAITING_FOR_PRE_SUNSET  (good night)
        NOON_CHECK        ──► WAITING_FOR_NOON        (bad night)
        WAITING_FOR_PRE_SUNSET ──► PRE_SUNSET_CHECK
        PRE_SUNSET_CHECK  ──► IMAGING                 (still good)
        PRE_SUNSET_CHECK  ──► WAITING_FOR_NOON        (conditions degraded)
        IMAGING           ──► WAITING_FOR_NOON        (always, after run ends)
        * any state       ──► WAITING_FOR_BOOT        (on unhandled exception)
    """
    WAITING_FOR_NOON = auto()
    NOON_CHECK = auto()
    WAITING_FOR_PRE_SUNSET = auto()
    PRE_SUNSET_CHECK = auto()
    IMAGING = auto()
    WAITING_FOR_BOOT = auto()


# ---------------------------------------------------------------------------
# MQTT message handler
# ---------------------------------------------------------------------------

def message_handling(client, userdata, msg):
    """Respond to MQTT status queries from the social server.

    The social server publishes a message to ``utils.topic_to_sched``
    whenever it needs the current observatory state (e.g. to reply to a
    ``status`` command).  This handler serialises
    ``observatory_state`` to JSON and publishes it back on
    ``iris/from_sched``.

    Args:
        client: The paho-mqtt client instance.
        userdata: Unused; required by the paho callback signature.
        msg: The incoming paho ``MQTTMessage``.
    """
    if msg.topic == utils.topic_to_sched:
        print("incoming message", msg)
        json_payload = json.dumps(observatory_state)
        topic = "iris/from_sched"
        LOGGER.info(f"Send `{json_payload}` to topic `{topic}`")
        result = client.publish(topic, json_payload)
        if result[0] == 0:
            print(f"Send `{json_payload}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def set_state(state: State, dso=None, will_image_tonight=None):
    """Update the in-memory ``observatory_state`` dict and log the transition.

    The dict is serialised to JSON and broadcast over MQTT on the next
    query from the social server (see ``message_handling``).

    Args:
        state: The new ``State`` enum value.
        dso: Optional DSO name to record (e.g. ``"NGC 891"``).
            If ``None``, the existing value is preserved.
        will_image_tonight: Optional bool indicating whether Iris will
            image tonight.  If ``None``, the existing value is preserved.
    """
    global observatory_state
    observatory_state["state"] = state.name
    if dso is not None:
        observatory_state["dso"] = dso
    if will_image_tonight is not None:
        observatory_state["will image tonight"] = will_image_tonight
    LOGGER.info("State: %s", state.name)
    print(f"State: {state.name}")
    # Persist so social_server can read it without MQTT.
    try:
        utils.set_install_dir()
        with open("scheduler_state.json", "w") as f:
            json.dump(observatory_state, f)
    except Exception:
        LOGGER.warning("Could not write scheduler_state.json")


def _wait_until(target: datetime):
    """Block the calling thread until *target* is reached.

    Sleeps in 30-second increments to remain responsive to OS signals
    and to avoid busy-waiting.  Handles both tz-aware and tz-naive
    datetimes: if *target* carries timezone info, ``now`` is computed in
    the same timezone so the comparison is always apples-to-apples.

    Args:
        target: The datetime at which to stop waiting.
    """
    if target.tzinfo is not None:
        now = lambda: datetime.now(target.tzinfo)
    else:
        now = datetime.now
    while now() < target:
        time.sleep(30)


def _next_noon() -> datetime:
    """Return a tz-naive datetime for the next upcoming noon.

    If the current time is already past noon today, returns noon
    tomorrow; otherwise returns noon today.

    Returns:
        A tz-naive ``datetime`` set to 12:00:00 local time.
    """
    now = datetime.now()
    noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now >= noon:
        noon += timedelta(days=1)
    return noon


def _pre_sunset_time() -> datetime:
    """Return the datetime that is 10 minutes before today's sunset."""
    _, sunset = weather.get_sunrise_sunset()
    return sunset - timedelta(minutes=10)


def _get_best_object() -> tuple[str, int, datetime]:
    """Query the instruction queue for the best DSO to image tonight.

    Reads ``my_instructions.json`` (path from config) and delegates to
    ``best_object_tonight``, which scores targets by visibility window,
    air mass, and priority.

    Returns:
        A ``(dso_name, good_hours, best_start)`` tuple where *good_hours*
        is the number of hours the target is above the horizon at
        acceptable air mass tonight, and *best_start* is the datetime
        when the target first clears the horizon with good conditions
        (may be ``None`` if no good hours exist).
    """
    instructions_path = os.path.join(_PROJECT_ROOT, CFG["location"]["instructions"])
    best_name, best_start, best_good_hours, grid_html = best_object_tonight(instructions_path)
    return best_name, best_good_hours, best_start, grid_html


def _imaging_plan_message(dso_name: str, best_good_hours: float, best_start: datetime) -> str:
    """Build a human-readable summary of tonight's imaging plan in observatory local time.

    Fetches today's sunset time and converts it (and *best_start*) to the
    observatory's configured timezone, then formats a three-line summary
    suitable for posting to the web chat.

    Args:
        dso_name: The name of the target DSO (e.g. ``"M 31"``).
        best_good_hours: Estimated good imaging hours for this target.
        best_start: Datetime when the DSO first clears the horizon with
            good conditions tonight (from ``best_object_tonight``).
            May be ``None``; falls back to ``"unknown"`` in that case.

    Returns:
        A multi-line string with target name, imaging duration,
        sunset time, and DSO rise time.  Falls back to ``"unknown"``
        for times if the weather API call or best_start is unavailable.
    """
    try:
        from zoneinfo import ZoneInfo  # stdlib, Python 3.9+
        tz = ZoneInfo(CFG["location"]["timezone"])
        _, sunset = get_sunrise_sunset()
        sunset_local = sunset.astimezone(tz)
        sunset_str = sunset_local.strftime("%H:%M %Z")
        if best_start is not None:
            if best_start.tzinfo is not None:
                start_local = best_start.astimezone(tz)
            else:
                start_local = best_start.replace(tzinfo=tz)
            start_str = start_local.strftime("%H:%M %Z")
        else:
            start_str = "unknown"
    except Exception:
        sunset_str = "unknown"
        start_str = "unknown"
    return (
        f"Target: {dso_name}\n"
        f"Imaging time: {best_good_hours:.1f}h\n"
        f"Sunset: {sunset_str}  |  DSO rises: ~{start_str}"
    )


def _generate_nina_sequence(dso_name: str):
    """Resolve DSO coordinates and write a N.I.N.A sequence file to disk.

    Resolves *dso_name* via ``instructions.resolve_target_by_name`` — stored
    RA/Dec from the queued record wins, falling back to a Simbad name lookup —
    extracts RA/Dec, then calls
    ``nina_sequence_gen.generate_sequence`` to patch the JSON template
    (``cfg["nina"]["sequence_input"]``) and write the output sequence
    (``cfg["nina"]["sequence_output"]``).

    The output path is a Windows path consumed by N.I.N.A running on the
    imaging PC; it must be accessible from the scheduler host (e.g.
    via a network share or the observatory's cross-OS file system mount).

    Args:
        dso_name: The target name, e.g. ``"NGC 891"`` or ``"M 42"``.

    Side effects:
        Writes a JSON file to ``cfg["nina"]["sequence_output"]``.
        Logs an error if *dso_name* cannot be resolved; in that case
        no file is written and N.I.N.A will use whatever sequence was
        there previously.
    """
    # Prefer the queued record's stored RA/Dec (positional targets have no
    # resolvable name); fall back to a Simbad name lookup for named DSOs.
    dso = instructions.resolve_target_by_name(dso_name)
    if dso is None:
        LOGGER.error("Could not resolve coordinates for %s", dso_name)
        return
    ra_hours = dso.coord.ra.hour
    dec_degrees = dso.coord.dec.deg
    template_path = Path(os.path.join(_PROJECT_ROOT, CFG["nina"]["sequence_input"]))
    output_path = Path(CFG["nina"]["sequence_output"])
    nina_sequence_gen.generate_sequence(
        template_path=template_path,
        dso_name=dso_name,
        ra_hours=ra_hours,
        dec_degrees=dec_degrees,
        output_path=output_path,
    )
    LOGGER.info("Generated Nina sequence for %s", dso_name)



# ---------------------------------------------------------------------------
# Start-up state selection
# ---------------------------------------------------------------------------

def _initial_state() -> State:
    """Choose the correct starting state based on the current time of day.

    Prevents the scheduler from always starting at WAITING_FOR_NOON when
    it is restarted mid-afternoon or at other points in the day.

    Logic:
        - Before noon → ``WAITING_FOR_NOON`` (wait for noon today).
        - Between noon and one hour before sunset → ``NOON_CHECK``
          (noon has already passed; run the noon logic immediately).
        - After one hour before sunset → ``WAITING_FOR_NOON``
          (too late to start tonight; wait for noon tomorrow).

    Returns:
        The appropriate ``State`` to start the machine in.
    """
    now = datetime.now()
    noon = now.replace(hour=12, minute=0, second=0, microsecond=0)

    if now < noon:
        LOGGER.info("Starting before noon — entering WAITING_FOR_NOON")
        return State.WAITING_FOR_NOON

    try:
        pre_sunset = _pre_sunset_time()
        # pre_sunset may be tz-aware; normalise for comparison
        if pre_sunset.tzinfo is not None:
            now_cmp = datetime.now(pre_sunset.tzinfo)
        else:
            now_cmp = now

        if now_cmp < pre_sunset:
            LOGGER.info("Starting after noon but before 10 min before sunset — entering NOON_CHECK")
            return State.NOON_CHECK
    except Exception:
        LOGGER.warning("Could not determine sunset time at startup, defaulting to WAITING_FOR_NOON")

    LOGGER.info("Starting after 1h before sunset — waiting for next day")
    return State.WAITING_FOR_NOON


# ---------------------------------------------------------------------------
# Prefect tasks — one per stage of the nightly cycle
# ---------------------------------------------------------------------------

@task(name="noon-check")
def noon_check_task() -> tuple[str, int, datetime, str]:
    """Refresh visibility data, pick tonight's best target, and decide whether to image."""
    set_state(State.NOON_CHECK)
    instructions.calc_and_store_hours_above_horizon()
    best_name, best_good_hours, best_start, grid_html = _get_best_object()
    LOGGER.info("Noon check: best=%s good_hours=%d", best_name, best_good_hours)

    mode = super_user_commands.get_mode()
    social_server.post_social_message(f"Imaging mode: {mode}")

    if best_good_hours >= _MIN_GOOD_HOURS:
        social_server.tonight_cmd(["me", "tonight", best_name], 2, "", "")
        social_server.post_social_message(
            f"Planning to image tonight\n{_imaging_plan_message(best_name, best_good_hours, best_start)}"
        )
        obs_calendar.set_today_stat('image', best_name)
        set_state(State.NOON_CHECK, best_name, True)
    else:
        if grid_html:
            social_server.post_html_message(grid_html)
        social_server.post_social_message(
            f"Not enough good imaging hours tonight ({best_good_hours}h), skipping"
        )
        obs_calendar.set_today_stat('weather', best_name)
        set_state(State.NOON_CHECK, best_name, False)

    return best_name, best_good_hours, best_start, grid_html


@task(name="wait-for-pre-sunset")
def wait_for_pre_sunset_task():
    """Sleep until 10 minutes before today's sunset."""
    set_state(State.WAITING_FOR_PRE_SUNSET)
    target = _pre_sunset_time()
    LOGGER.info("Waiting until 10 min before sunset at %s", target)
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(CFG["location"]["timezone"])
        target_local = target.astimezone(tz) if target.tzinfo else target.replace(tzinfo=tz)
        social_server.post_social_message(f"Imaging check at {target_local.strftime('%H:%M %Z')} (10 min before sunset)")
    except Exception:
        social_server.post_social_message("Waiting until 10 min before sunset for imaging check")
    _wait_until(target)


@task(name="pre-sunset-check")
def pre_sunset_check_task() -> tuple[str, int]:
    """Re-evaluate conditions close to sunset and post the updated imaging grid."""
    set_state(State.PRE_SUNSET_CHECK)
    best_name, best_good_hours, best_start, grid_html = _get_best_object()
    if grid_html:
        social_server.post_html_message(grid_html)
    LOGGER.info("Pre-sunset check: best=%s good_hours=%d", best_name, best_good_hours)

    mode = super_user_commands.get_mode()
    social_server.post_social_message(f"Imaging mode: {mode}")

    if best_good_hours >= _MIN_GOOD_HOURS:
        social_server.post_social_message(
            f"Confirmed — generating sequence\n{_imaging_plan_message(best_name, best_good_hours, best_start)}"
        )
        social_server.post_dso_preview(best_name)
        set_state(State.PRE_SUNSET_CHECK, best_name, True)
    else:
        social_server.post_social_message(
            f"Conditions not good enough at sunset ({best_good_hours}h), skipping tonight"
        )
        set_state(State.PRE_SUNSET_CHECK, best_name, False)

    return best_name, best_good_hours


@task(name="generate-nina-sequence")
def generate_sequence_task(dso_name: str):
    """Resolve DSO coordinates and write the N.I.N.A sequence file to disk."""
    _generate_nina_sequence(dso_name)


@task(name="run-imaging", timeout_seconds=9 * 60 * 60)
def imaging_task():
    """Launch the full observatory imaging run and wait for it to complete."""
    set_state(State.IMAGING)
    LOGGER.info("Starting imaging run")
    if super_user_commands.get_mode() == "auto":
        super_user_commands.image_cmd(["", "image!!", "1"], "iris")
        LOGGER.info("Waiting for imaging state to return to NONE")
        while super_user_commands.get_imaging_state() != super_user_commands.ImagingState.NONE:
            time.sleep(60)
        LOGGER.info("Imaging state is NONE — run complete")
    else:
        social_server.post_social_message("Mode is manual — skipping auto imaging")


# ---------------------------------------------------------------------------
# Prefect flow — one complete nightly cycle
# ---------------------------------------------------------------------------

@flow(name="nightly-imaging-cycle")
def nightly_cycle():
    """Drive the observatory through one complete nightly cycle.

    Noon check → (if good) wait for pre-sunset → pre-sunset re-check →
    (if still good) generate NINA sequence → imaging run.
    Returns early without imaging if conditions don't meet the threshold
    at either check point.
    """
    best_name, good_hours, best_start, grid_html = noon_check_task()

    if good_hours < _MIN_GOOD_HOURS:
        LOGGER.info("Skipping tonight — not enough good hours (%d)", good_hours)
        return

    wait_for_pre_sunset_task()

    best_name, good_hours = pre_sunset_check_task()

    if good_hours < _MIN_GOOD_HOURS:
        LOGGER.info("Conditions degraded at sunset — skipping tonight")
        return

    generate_sequence_task(best_name)
    imaging_task()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Initialise subsystems and run the nightly scheduling loop.

    Startup sequence:
        1. Write ``safety.txt`` as ``USER SAFE``.
        2. Reset imaging state to ``NONE`` (clears any stale ``imaging.txt``).
        3. Force mode to ``manual`` — operator must switch to ``auto`` via
           the web chat before unattended imaging is triggered.
        4. Connect to the MQTT broker (continues without it if unavailable).
        5. Enter the ``while True`` loop: wait for noon, run ``nightly_cycle()``.
           Each nightly run is recorded by Prefect as a named flow run with
           per-task states, timing, and logs.
    """
    print("Starting Scheduler Server")
    super_user_commands.safe_cmd(None, None)
    # Don't reset imaging state if NINA is still capturing — a supervisor
    # relaunch (e.g. after a social-server crash) restarts us mid-run, and
    # clearing imaging.txt would let a manual command start a second NINA run.
    # But a completion marker (DONE_FLATS) or idle NONE is safe to clear even
    # when NINA is merely open: those are not an in-progress capture, and a
    # stale DONE_FLATS surviving a restart strands the next nightly cycle.
    imaging_state = super_user_commands.get_imaging_state()
    _imaging_terminal = {super_user_commands.ImagingState.NONE,
                         super_user_commands.ImagingState.DONE_FLATS}
    if super_user_commands.is_nina_running() and imaging_state not in _imaging_terminal:
        LOGGER.warning("NINA running mid-run (%s) at scheduler startup — leaving imaging state intact",
                       imaging_state.value)
    else:
        super_user_commands.set_imaging_state(super_user_commands.ImagingState.NONE)
    super_user_commands.set_mode("manual")
    LOGGER.info('Start Scheduler')
    try:
        client = utils.connect_mqtt()
        client.subscribe(utils.topic_to_sched)
        client.on_message = message_handling
        client.loop_start()
    except ConnectionRefusedError:
        LOGGER.warning("MQTT broker not available — continuing without it")

    # If restarted after noon but before pre-sunset, skip straight to the
    # noon-check flow instead of waiting until tomorrow's noon.
    skip_first_wait = (_initial_state() == State.NOON_CHECK)

    while True:
        try:
            if not skip_first_wait:
                set_state(State.WAITING_FOR_NOON)
                target = _next_noon()
                LOGGER.info("Waiting for noon at %s", target)
                _wait_until(target)
            skip_first_wait = False
            nightly_cycle()
        except Exception:
            LOGGER.exception("Unhandled exception in scheduler loop")
            try:
                social_server.post_social_message("Oops — scheduler had a problem, recovering")
            except Exception:
                LOGGER.exception("Also failed to post exception notice")
            set_state(State.WAITING_FOR_BOOT)
            time.sleep(60)


if __name__ == '__main__':
    main()
