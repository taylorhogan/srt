import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path

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

CFG = config.data()
LOGGER = utils.set_logger()
CFG["logger"]["logging"] = LOGGER

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MIN_GOOD_HOURS = 3

client = None
observatory_state = {
    "state": "Unknown",
    "dso": "Unknown",
    "will image tonight": "Unknown"
}


class State(Enum):
    WAITING_FOR_NOON = auto()
    NOON_CHECK = auto()
    WAITING_FOR_PRE_SUNSET = auto()
    PRE_SUNSET_CHECK = auto()
    IMAGING = auto()
    WAITING_FOR_BOOT = auto()


def message_handling(client, userdata, msg):
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


def set_state(state: State, dso=None, will_image_tonight=None):
    global observatory_state
    observatory_state["state"] = state.name
    if dso is not None:
        observatory_state["dso"] = dso
    if will_image_tonight is not None:
        observatory_state["will image tonight"] = will_image_tonight
    LOGGER.info("State: %s", state.name)
    print(f"State: {state.name}")


def _wait_until(target: datetime):
    """Sleep in 30-second increments until target time (handles tz-aware or naive)."""
    if target.tzinfo is not None:
        now = lambda: datetime.now(target.tzinfo)
    else:
        now = datetime.now
    while now() < target:
        time.sleep(30)


def _next_noon() -> datetime:
    now = datetime.now()
    noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now >= noon:
        noon += timedelta(days=1)
    return noon


def _one_hour_before_sunset() -> datetime:
    _, sunset = weather.get_sunrise_sunset()
    return sunset - timedelta(hours=1)


def _get_best_object() -> tuple[str, int]:
    instructions_path = os.path.join(_PROJECT_ROOT, CFG["location"]["instructions"])
    best_name, best_start, best_good_hours = best_object_tonight(instructions_path)
    return best_name, best_good_hours


def _send_grid_to_mastodon():
    image_grid_path = os.path.join(_PROJECT_ROOT, CFG["location"]["image_grid"])
    social_server.post_social_message("Tonight's imaging grid", image=image_grid_path)


def _imaging_plan_message(dso_name: str, best_good_hours: float) -> str:
    """Build a human-readable summary of tonight's imaging plan."""
    try:
        _, sunset = get_sunrise_sunset()
        imaging_start = sunset - timedelta(hours=1)
        sunset_str = sunset.strftime("%H:%M %Z")
        start_str = imaging_start.strftime("%H:%M %Z")
    except Exception:
        sunset_str = "unknown"
        start_str = "unknown"
    return (
        f"Target: {dso_name}\n"
        f"Imaging time: {best_good_hours:.1f}h\n"
        f"Sunset: {sunset_str}  |  Imaging starts: ~{start_str}"
    )


def _generate_nina_sequence(dso_name: str):
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


def _initial_state() -> State:
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


def _run_state_machine():
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
                instructions.calc_and_store_hours_above_horizon()
                best_name, best_good_hours = _get_best_object()
                _send_grid_to_mastodon()
                LOGGER.info("Noon check: best=%s good_hours=%d", best_name, best_good_hours)

                mode = super_user_commands.get_mode()
                social_server.post_social_message(f"Imaging mode: {mode}")
                if best_good_hours >= _MIN_GOOD_HOURS:
                    social_server.tonight_cmd(["me", "tonight", best_name], 2, "", "")
                    social_server.post_social_message(
                        f"Planning to image tonight\n{_imaging_plan_message(best_name, best_good_hours)}"
                    )
                    obs_calendar.set_today_stat('image', best_name)
                    set_state(state, best_name, True)
                    state = State.WAITING_FOR_PRE_SUNSET
                else:
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
                best_name, best_good_hours = _get_best_object()
                _send_grid_to_mastodon()
                LOGGER.info("Pre-sunset check: best=%s good_hours=%d", best_name, best_good_hours)

                mode = super_user_commands.get_mode()
                social_server.post_social_message(f"Imaging mode: {mode}")
                if best_good_hours >= _MIN_GOOD_HOURS:
                    social_server.post_social_message(
                        f"Confirmed — generating sequence\n{_imaging_plan_message(best_name, best_good_hours)}"
                    )
                    set_state(state, best_name, True)
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
                    super_user_commands.image_cmd(["", "image!!", "1"], "iris")
                else:
                    social_server.post_social_message("Mode is manual — skipping auto imaging")
                LOGGER.info("Imaging run complete")
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


def main():
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
