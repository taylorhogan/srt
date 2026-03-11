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
tonight, posts a status update to Mastodon, and advances (or retreats)
to the next state.

State descriptions
------------------
WAITING_FOR_NOON
    Sleeps in 30-second increments until the next calendar noon.

NOON_CHECK
    Recalculates DSO visibility for all queued targets, picks the best
    one for tonight, posts the imaging grid and plan to Mastodon, and
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
- **Mastodon**: all human-readable status updates are posted via
  ``social_server.post_social_message``.
- **Pushover**: not used directly here; called downstream by
  ``super_user_commands.doit_cmd`` during the actual imaging run.

Dependencies (non-stdlib)
-------------------------
- ``iris_astronomy`` — DSO visibility, calendar, weather / sunset times.
- ``control.instructions`` — JSON-backed DSO request queue.
- ``cmd_processing.social_server`` — Mastodon posting.
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

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from iris_astronomy import astro_dso_visibility, obs_calendar, weather
from iris_astronomy.astro_dso_visibility import best_object_tonight, is_a_dso_object
from iris_astronomy.weather import get_sunrise_sunset
from configs import config
from control import instructions
from cmd_processing import social_server
from utils import utils, pushover
from cmd_processing import super_user_commands
from nina_gen import nina_sequence_gen
from fits_processing import fitsfwhm

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
    Mastodon ``status`` command).  This handler serialises
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


def _one_hour_before_sunset() -> datetime:
    """Return the datetime that is one hour before today's sunset.

    Delegates to ``weather.get_sunrise_sunset()`` which calls the
    Open-Meteo API.  The returned datetime may be tz-aware (UTC or
    local, depending on the weather module implementation).

    Returns:
        A ``datetime`` one hour before today's sunset.
    """
    _, sunset = weather.get_sunrise_sunset()
    return sunset - timedelta(hours=1)


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
    best_name, best_start, best_good_hours = best_object_tonight(instructions_path)
    return best_name, best_good_hours, best_start


def _send_grid_to_mastodon():
    """Post the nightly imaging grid image to Mastodon.

    Reads the pre-generated grid PNG from the path in config
    (``cfg["location"]["image_grid"]``) and posts it with a caption.
    Called at both NOON_CHECK and PRE_SUNSET_CHECK.
    """
    image_grid_path = os.path.join(_PROJECT_ROOT, CFG["location"]["image_grid"])
    social_server.post_social_message("Tonight's imaging grid", image=image_grid_path)


def _imaging_plan_message(dso_name: str, best_good_hours: float, best_start: datetime) -> str:
    """Build a human-readable summary of tonight's imaging plan in observatory local time.

    Fetches today's sunset time and converts it (and *best_start*) to the
    observatory's configured timezone, then formats a three-line summary
    suitable for posting to Mastodon.

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

    Looks up *dso_name* in the astropy/Simbad catalogue via
    ``is_a_dso_object``, extracts RA/Dec, then calls
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
    dso = is_a_dso_object(dso_name)
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
# Post-imaging quality summary
# ---------------------------------------------------------------------------

def _post_imaging_summary(imaging_start: datetime) -> None:
    """Find all FITS files captured since *imaging_start*, compute median
    quality metrics across them, and post a summary to Mastodon.

    Walks ``cfg["nina"]["image_dir"]`` recursively for ``*.fits`` files
    whose modification time is >= *imaging_start*.  For each file,
    ``fitsfwhm.calculate_fwhm`` is called to get per-frame star count,
    FWHM (pixels and arcsec), and eccentricity.  The median of each
    metric across all frames is then posted.

    Args:
        imaging_start: The datetime just before imaging was launched,
            used as the cutoff for which FITS files belong to this run.
    """
    image_dir = Path(CFG["nina"]["image_dir"])
    arcsec_per_pixel = CFG["nina"]["arc_sec_per_pixel"]
    start_ts = imaging_start.timestamp()

    try:
        fits_files = sorted(
            (f for f in image_dir.rglob("*.fits") if f.stat().st_mtime >= start_ts),
            key=lambda f: f.stat().st_mtime,
        )
    except Exception:
        LOGGER.exception("Failed to scan image directory %s", image_dir)
        social_server.post_social_message("Imaging complete — could not scan image directory")
        return

    if not fits_files:
        social_server.post_social_message("Imaging complete — no new FITS files found")
        return

    LOGGER.info("Found %d FITS files since imaging start", len(fits_files))

    fwhm_px_list: list[float] = []
    fwhm_arcsec_list: list[float] = []
    star_count_list: list[int] = []
    ecc_list: list[float] = []

    for fits_file in fits_files:
        try:
            mean_px, mean_arcsec, star_count, mean_ecc = fitsfwhm.calculate_fwhm(
                fits_file, arcsec_per_pixel=arcsec_per_pixel
            )
            if star_count > 0:
                fwhm_px_list.append(mean_px)
                fwhm_arcsec_list.append(mean_arcsec)
                star_count_list.append(star_count)
                ecc_list.append(mean_ecc)
        except Exception:
            LOGGER.exception("FWHM analysis failed for %s", fits_file)

    if not fwhm_px_list:
        social_server.post_social_message(
            f"Imaging complete — {len(fits_files)} frames, no stars detected in any frame"
        )
        return

    median_fwhm_px = float(np.median(fwhm_px_list))
    median_fwhm_arcsec = float(np.median(fwhm_arcsec_list))
    median_stars = float(np.median(star_count_list))
    median_ecc = float(np.median(ecc_list))

    social_server.post_social_message(
        f"Imaging complete — {len(fits_files)} frames ({len(fwhm_px_list)} with stars)\n"
        f"Median FWHM: {median_fwhm_px:.2f} px ({median_fwhm_arcsec:.2f}\")\n"
        f"Median stars/frame: {median_stars:.0f}\n"
        f"Median eccentricity: {median_ecc:.3f}"
    )
    LOGGER.info(
        "Imaging summary: %d frames, median FWHM=%.2f px (%.2f\"), stars=%.0f, ecc=%.3f",
        len(fits_files), median_fwhm_px, median_fwhm_arcsec, median_stars, median_ecc,
    )


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
        pre_sunset = _one_hour_before_sunset()
        # pre_sunset may be tz-aware; normalise for comparison
        if pre_sunset.tzinfo is not None:
            now_cmp = datetime.now(pre_sunset.tzinfo)
        else:
            now_cmp = now

        if now_cmp < pre_sunset:
            LOGGER.info("Starting after noon but before 1h before sunset — entering NOON_CHECK")
            return State.NOON_CHECK
    except Exception:
        LOGGER.warning("Could not determine sunset time at startup, defaulting to WAITING_FOR_NOON")

    LOGGER.info("Starting after 1h before sunset — waiting for next day")
    return State.WAITING_FOR_NOON


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def _run_state_machine():
    """Drive the scheduler through its nightly cycle indefinitely.

    This function never returns under normal operation.  Each iteration
    of the ``while True`` loop handles exactly one state transition.
    Unhandled exceptions are caught, reported to Mastodon, and the
    machine falls back to ``WAITING_FOR_BOOT`` (a one-minute recovery
    pause) before resuming.

    State details
    ~~~~~~~~~~~~~

    **WAITING_FOR_BOOT**
        One-minute sleep used as a recovery buffer after an exception.
        Gives external services (MQTT broker, Mastodon) time to
        stabilise before the machine tries again.

    **WAITING_FOR_NOON**
        Calls ``_next_noon()`` and sleeps via ``_wait_until``.  After
        waking, advances to ``NOON_CHECK``.

    **NOON_CHECK**
        - Recalculates hours-above-horizon for every queued target.
        - Picks the best DSO via ``_get_best_object()``.
        - Posts the imaging grid to Mastodon.
        - Posts the current mode (``auto`` / ``manual``) to Mastodon.
        - If ``best_good_hours >= _MIN_GOOD_HOURS``: posts a plan
          message, updates the calendar, and advances to
          ``WAITING_FOR_PRE_SUNSET``.
        - Otherwise: posts a skip notice, records a weather-block in the
          calendar, and loops back to ``WAITING_FOR_NOON``.

    **WAITING_FOR_PRE_SUNSET**
        Sleeps until one hour before today's sunset, then advances to
        ``PRE_SUNSET_CHECK``.

    **PRE_SUNSET_CHECK**
        Re-evaluates best object and good hours.  If still adequate,
        generates the N.I.N.A sequence file and advances to ``IMAGING``.
        Otherwise skips tonight and loops back to ``WAITING_FOR_NOON``.

    **IMAGING**
        - In ``auto`` mode: calls ``image_cmd(["", "image!!", "1"],
          "iris")`` which launches the full observatory run
          (safety checks → roof open → NINA prelude → NINA main).
          ``image_cmd`` uses ``subprocess.Popen`` internally so this
          call returns quickly; the scheduler then **polls
          ``get_imaging_state()`` every 60 seconds** until ``end.py``
          resets the state to ``ImagingState.NONE``, confirming the run
          has truly finished before cycling back to ``WAITING_FOR_NOON``.
        - In ``manual`` mode: posts a skip notice and immediately
          advances to ``WAITING_FOR_NOON``.
    """
    state = _initial_state()

    while True:
        try:
            if state == State.WAITING_FOR_BOOT:
                set_state(state)
                time.sleep(60)
                state = State.WAITING_FOR_NOON

            elif state == State.WAITING_FOR_NOON:
                set_state(state)
                target = _next_noon()
                LOGGER.info("Waiting for noon at %s", target)
                _wait_until(target)
                state = State.NOON_CHECK

            elif state == State.NOON_CHECK:
                set_state(state)
                # Refresh visibility metrics for all queued targets.
                instructions.calc_and_store_hours_above_horizon()
                best_name, best_good_hours, best_start = _get_best_object()
                _send_grid_to_mastodon()
                LOGGER.info("Noon check: best=%s good_hours=%d", best_name, best_good_hours)

                mode = super_user_commands.get_mode()
                social_server.post_social_message(f"Imaging mode: {mode}")
                if best_good_hours >= _MIN_GOOD_HOURS:
                    # Good night — announce the plan and proceed.
                    social_server.tonight_cmd(["me", "tonight", best_name], 2, "", "")
                    social_server.post_social_message(
                        f"Planning to image tonight\n{_imaging_plan_message(best_name, best_good_hours, best_start)}"
                    )
                    obs_calendar.set_today_stat('image', best_name)
                    set_state(state, best_name, True)
                    state = State.WAITING_FOR_PRE_SUNSET
                else:
                    # Not enough imaging time — skip tonight.
                    social_server.post_social_message(
                        f"Not enough good imaging hours tonight ({best_good_hours}h), skipping"
                    )
                    obs_calendar.set_today_stat('weather', best_name)
                    set_state(state, best_name, False)
                    state = State.WAITING_FOR_NOON

            elif state == State.WAITING_FOR_PRE_SUNSET:
                set_state(state)
                target = _one_hour_before_sunset()
                LOGGER.info("Waiting for 1h before sunset at %s", target)
                _wait_until(target)
                state = State.PRE_SUNSET_CHECK

            elif state == State.PRE_SUNSET_CHECK:
                set_state(state)
                # Re-check conditions now that we are closer to imaging time.
                best_name, best_good_hours, best_start = _get_best_object()
                _send_grid_to_mastodon()
                LOGGER.info("Pre-sunset check: best=%s good_hours=%d", best_name, best_good_hours)

                mode = super_user_commands.get_mode()
                social_server.post_social_message(f"Imaging mode: {mode}")
                if best_good_hours >= _MIN_GOOD_HOURS:
                    social_server.post_social_message(
                        f"Confirmed — generating sequence\n{_imaging_plan_message(best_name, best_good_hours, best_start)}"
                    )
                    set_state(state, best_name, True)
                    # Write the N.I.N.A sequence to disk before NINA starts.
                    _generate_nina_sequence(best_name)
                    state = State.IMAGING
                else:
                    social_server.post_social_message(
                        f"Conditions not good enough at sunset ({best_good_hours}h), skipping tonight"
                    )
                    set_state(state, best_name, False)
                    state = State.WAITING_FOR_NOON

            elif state == State.IMAGING:
                set_state(state)
                LOGGER.info("Starting imaging run")
                if super_user_commands.get_mode() == "auto":
                    imaging_start = datetime.now()
                    super_user_commands.image_cmd(["", "image!!", "1"], "iris")
                    # image_cmd launches NINA via Popen (non-blocking); wait here
                    # until end.py or the NINA bat resets the state to NONE.
                    LOGGER.info("Waiting for imaging state to return to NONE")
                    while super_user_commands.get_imaging_state() != super_user_commands.ImagingState.NONE:
                        time.sleep(60)
                    LOGGER.info("Imaging state is NONE — run complete")
                    _post_imaging_summary(imaging_start)
                else:
                    social_server.post_social_message("Mode is manual — skipping auto imaging")
                state = State.WAITING_FOR_NOON

        except Exception:
            LOGGER.exception("Exception in state %s", state)
            try:
                social_server.get_mastodon_instance().status_post(
                    f"Oops I had a problem in state {state.name}"
                )
            except Exception:
                LOGGER.exception("Also failed to post exception notice to Mastodon")
            state = State.WAITING_FOR_BOOT


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Initialise subsystems and start the scheduling loop.

    Startup sequence:
        1. Write ``safety.txt`` as ``USER SAFE`` so the run can proceed
           if mode is ``auto`` (``safe_cmd`` does this).
        2. Reset imaging state to ``NONE`` (clears any stale ``imaging.txt``
           left by a previous crash).
        3. Force mode to ``manual`` — the operator must explicitly switch
           to ``auto`` mode via Mastodon before any unattended imaging
           will be triggered.
        4. Connect to the MQTT broker and subscribe to the inbound topic.
           If the broker is unavailable (e.g. running in development),
           the scheduler continues without MQTT — it simply won't respond
           to status queries.
        5. Enter ``_run_state_machine()``, which never returns.
    """
    print("Starting Scheduler Server")
    super_user_commands.safe_cmd(None, None)
    super_user_commands.ImagingState(super_user_commands.ImagingState.NONE)
    super_user_commands.set_mode("manual")
    LOGGER.info('Start Scheduler')
    try:
        client = utils.connect_mqtt()
        client.subscribe(utils.topic_to_sched)
        client.on_message = message_handling
        client.loop_start()
    except ConnectionRefusedError:
        LOGGER.warning("MQTT broker not available — continuing without it")
    _run_state_machine()


if __name__ == '__main__':
    main()
