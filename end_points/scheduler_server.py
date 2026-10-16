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
import threading
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
from prefect import flow, task

# Prefect starts a temporary API server for each flow run on a RANDOM port in
# 8000-9000, and tests the port by binding 127.0.0.1 only. The web chat listens
# on 0.0.0.0:8095, which Windows does not count as a conflict, so on 2026-09-07
# the noon flow drew 8095: its server sat on 127.0.0.1:8095 shadowing the web
# chat for every localhost client (the preview worker, the recorder's notice,
# in-process posts all got Prefect's 404), the flow itself failed on its own
# health check, and the stray server outlived the flow. The conductor's 8096
# is in the same range. Keep Prefect well clear of anything of ours.
try:
    from prefect.server.api.server import SubprocessASGIServer as _PrefectASGI
    _PrefectASGI._port_range = range(8700, 8999)
except Exception:  # noqa: BLE001 -- a Prefect layout change must not stop the scheduler
    pass

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from iris_astronomy import astro_dso_visibility, obs_calendar, weather
from iris_astronomy.astro_dso_visibility import best_object_tonight
from iris_astronomy.weather import get_sunrise_sunset
from configs import config
from control import instructions
from control import slot_plan
from cmd_processing import social_server
from utils import utils, pushover
from cmd_processing import super_user_commands
from nina_gen import nina_sequence_gen, sequence_lint

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

# The DSO's rise time from the noon check, kept so the notification written at
# sequence-generation time can quote it. Prefect tasks do not share locals.
_LAST_BEST_START: dict = {}
# Signature (control.slot_plan.signature) of the plan the sequence on disk
# was generated from. The pre-sunset check compares against it and
# regenerates on ANY difference -- a second slot appearing or vanishing, a
# window moving -- not only when the best target's name changes.
_PLAN_ON_DISK: dict = {}

def _slots_record(slots) -> list:
    """The slot plan as plain JSON for scheduler_state.json."""
    out = []
    for s in slots or []:
        out.append({"dso": s.name,
                    "start": s.start.isoformat(timespec="minutes"),
                    "end": s.end.isoformat(timespec="minutes"),
                    "hours": int(s.good_hours)})
    return out


def set_state(state: State, dso=None, will_image_tonight=None, slots=None):
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
    if slots is not None:
        # The night's slot plan (control/slot_plan). The shadow conductor reads
        # its length as the number of slots; the chat reads it for the plan.
        observatory_state["slots"] = _slots_record(slots)
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


def _timeline_rows(dso_name: str, best_start, output_path, slots, plans) -> tuple:
    """Tonight as (rows, title) for control/night_timeline.

    Filters, counts and exposure lengths are READ BACK from the written
    sequence file (nina_sequence_gen.read_back), never taken from *plans*:
    on 2026-10-14 the report showed the plan's LRGB while the file said Ha.
    *plans* is kept only for the single-target fallback's shape.
    """
    from zoneinfo import ZoneInfo
    from control import night_timeline
    tz = ZoneInfo(CFG["location"]["timezone"])
    sunset = weather.get_sunrise_sunset()[1]
    back = nina_sequence_gen.read_back(output_path)        # [(name, counts, exposures)]
    if slots and len(back) == len(slots):
        rows = [night_timeline.SlotPlan(s.name, s.start, s.end, counts, exposures)
                for s, (_name, counts, exposures) in zip(slots, back)]
    else:
        start = best_start if best_start is None or getattr(best_start, "tzinfo", None) \
            else best_start.replace(tzinfo=tz)
        end = slots[0].end if slots and slots[0].name == dso_name else None
        counts, exposures = (back[0][1], back[0][2]) if back else ({}, {})
        rows = [night_timeline.SlotPlan(dso_name, start, end, counts, exposures)]
    return (night_timeline.table(sunset, rows, tz),
            "Tonight: " + " + ".join(r.name for r in rows))


# Which check is running, for the plan published to the site (2026-10-15).
_CHECK = {"name": "noon"}
_PLAN_STATUS_PATH = os.path.join(_PROJECT_ROOT, "local", "night_plan_status.json")


def _plan_picture_path() -> str:
    scratch = os.path.join(_PROJECT_ROOT, CFG["scratch"]["directory"])
    os.makedirs(scratch, exist_ok=True)
    return os.path.join(scratch, "night_timeline.jpg")


def _publish_plan_to_site(verdict: str, title: str = "", summary: str = "",
                          image=None, reason: str = "", restamp: bool = False) -> None:
    """Write the plan sidecar the live panel reads (control/night_plan_status)
    and trigger a render, so the site shows the same picture Pushover got --
    or the "no imaging tonight" card. Never raises: the site is downstream
    of the night, not a condition of it."""
    from control import night_plan_status
    try:
        if restamp:
            night_plan_status.restamp(_PLAN_STATUS_PATH, _CHECK["name"])
        else:
            night_plan_status.write(_PLAN_STATUS_PATH, verdict, _CHECK["name"], image,
                                    title=title, summary=summary, reason=reason)
    except Exception:
        LOGGER.exception("could not write the plan status for the site")
        return
    try:
        from scripts import live_trigger
        live_trigger.trigger("plan-%s-%s" % (verdict, _CHECK["name"]))
    except Exception:
        LOGGER.exception("could not trigger the live render for the plan")


def _publish_skip_to_site(reason: str) -> None:
    """Draw the "no imaging tonight" card and publish it (noon, sunset, or a
    refusal at start)."""
    from control import night_timeline
    image = None
    try:
        rows = night_timeline.skip_rows(datetime.now().astimezone(), _CHECK["name"], reason)
        image = night_timeline.picture(rows, "Tonight: no imaging", _plan_picture_path())
    except Exception:
        LOGGER.exception("could not draw the no-imaging card")
    _publish_plan_to_site("skip", title="Tonight: no imaging", image=image, reason=reason)


def _push_imaging_plan(dso_name: str, good_hours: float, best_start, output_path,
                       slots=None, plans=None):
    """Push what will be imaged tonight, when, and in which filters.

    Since 2026-10-08 a timeline (time | event: roof open/close, each target's
    start and filters, flats and analysis done); since 2026-10-14 drawn as a
    picture for Pushover (a three-line summary above it) and an HTML table
    for the chat, because Pushover's proportional font never lined the text
    columns up. The text timeline is the fallback for either, and the older
    paragraph below the fallback if no timeline can be built at all.
    """
    from control import night_timeline
    try:
        rows, title = _timeline_rows(dso_name, best_start, output_path, slots, plans)
        msg = night_timeline.render([(r.time, r.approx, r.line) for r in rows], title=title)
    except Exception:
        LOGGER.exception("Could not build the timeline; falling back to the summary")
        msg = None
    if msg:
        image = None
        try:
            image = night_timeline.picture(rows, title, _plan_picture_path(),
                                           source=Path(output_path).name)
        except Exception:
            LOGGER.exception("Could not draw the timeline picture; pushing the text")
        # The same picture to the site (2026-10-15).
        _publish_plan_to_site("imaging", title=title, summary=night_timeline.summary(rows, title),
                              image=image)
        try:
            if image:
                pushover.push_message(night_timeline.summary(rows, title), image=image)
            else:
                pushover.push_message(msg)
        except Exception:
            LOGGER.exception("Could not push the imaging plan")
        try:
            social_server.post_html_message(night_timeline.html(rows, title,
                                                                source=Path(output_path).name))
        except Exception:
            LOGGER.exception("Could not post the timeline table; posting the text")
            social_server.post_social_message(msg)
        return

    try:
        rows = nina_sequence_gen.describe_sequence(output_path)
    except Exception:
        LOGGER.exception("Could not read back the generated sequence")
        rows = []

    lines = [f"Tonight: {dso_name}"]
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(CFG["location"]["timezone"])
        sunset = weather.get_sunrise_sunset()[1]
        lines.append("Sunset %s" % sunset.astimezone(tz).strftime("%H:%M"))
        if best_start is not None:
            start = best_start if getattr(best_start, "tzinfo", None) else best_start.replace(tzinfo=tz)
            lines.append("Starts ~%s, %.1fh usable" % (start.astimezone(tz).strftime("%H:%M"), good_hours))
        else:
            lines.append("%.1fh usable" % good_hours)
    except Exception:
        lines.append("%.1fh usable" % good_hours)

    if rows:
        total = sum(n * e for _f, n, e in rows)
        for filt, n, exp in rows:
            lines.append("  %-6s %3d x %.0fs = %.1fh" % (filt, n, exp, n * exp / 3600.0))
        lines.append("Total %.1fh over %d frames" % (total / 3600.0, sum(n for _f, n, _e in rows)))
    else:
        lines.append("  (no filter plan — template defaults)")

    try:
        pushover.push_message("\n".join(lines))
    except Exception:
        LOGGER.exception("Could not push the imaging plan")
    social_server.post_social_message("\n".join(lines))


def _filter_plans_for(slots) -> dict:
    """{slot name: its stored filter plan} for the plan signature. Never raises."""
    out = {}
    for s in slots or []:
        try:
            out[s.name] = instructions.get_filter_plan(s.name) or {}
        except Exception:
            out[s.name] = {}
    return out


def _second_slot_message(slots) -> str:
    """'\nThen <dso> HH:MM-HH:MM (Nh)' per later slot when the plan has more than one."""
    if len(slots or []) < 2:
        return ""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(CFG["location"]["timezone"])
        return "".join("\nThen %s %s-%s (%dh) — slot %d" % (
            s.name, s.start.astimezone(tz).strftime("%H:%M"),
            s.end.astimezone(tz).strftime("%H:%M"), s.good_hours, k + 2)
            for k, s in enumerate(slots[1:]))
    except Exception:
        return "".join("\nThen %s" % s.name for s in slots[1:])


def _generate_nina_slots_sequence(slots):
    """Write a two-target sequence for *slots* (control.slot_plan.Slot list).

    Same resolution as the one-target path per slot; the template's target
    container is cloned per slot (nina_sequence_gen.generate_slots_sequence).
    Falls back to the one-target path if a slot cannot be resolved, so a bad
    second pick costs the second slot and never the night.
    """
    specs = []
    for s in slots:
        dso = instructions.resolve_target_by_name(s.name)
        if dso is None:
            LOGGER.error("Could not resolve coordinates for slot %s; single-slot night", s.name)
            first = slots[0]
            return _generate_nina_sequence(first.name, first.good_hours, end=first.end)
        specs.append({"name": s.name, "ra_hours": dso.coord.ra.hour,
                      "dec_degrees": dso.coord.dec.deg, "seconds": s.seconds,
                      "start": s.start, "end": s.end})
    template_path = Path(os.path.join(_PROJECT_ROOT, CFG["nina"]["sequence_input"]))
    output_path = Path(CFG["nina"]["sequence_output"])
    state_script = os.path.join(_PROJECT_ROOT, "scripts", "set_imaging_state.bat")
    plans = nina_sequence_gen.generate_slots_sequence(template_path, specs, output_path,
                                                      state_script=state_script)
    LOGGER.info("Generated %d-slot Nina sequence: %s", len(slots),
                "; ".join("%s %s" % (s.name, p) for s, p in zip(slots, plans)))
    return plans, output_path


def _generate_nina_sequence(dso_name: str, good_hours: float = 0.0, end=None):
    """Resolve DSO coordinates and write a N.I.N.A sequence file to disk.

    *end* is the planner's window end for the target (control/slot_plan.Slot
    .end): the hour after its last hour that is above the treeline, inside
    the dark hours AND forecast clear -- i.e. the earliest of horizon, dawn
    and weather. It becomes the target container's hard TimeCondition, so
    the sequence stops when the plan says the useful hours are over instead
    of looping the container until dawn. None keeps the template's dawn.

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
    # The hours the noon check judged usable are the budget the filter split
    # divides. Without this the scheduler generated sequences with no filter
    # plan at all and N.I.N.A ran whatever iteration counts the template
    # happened to carry -- only the webchat `sequence` command ever passed it.
    plan = nina_sequence_gen.generate_sequence(
        template_path=template_path,
        dso_name=dso_name,
        ra_hours=ra_hours,
        dec_degrees=dec_degrees,
        output_path=output_path,
        above_horizon_seconds=(good_hours or 0) * 3600.0,
        end=end,
    )
    LOGGER.info("Generated Nina sequence for %s: %s (ends %s)", dso_name, plan,
                end.strftime("%H:%M") if end is not None else "at dawn")
    return plan, output_path



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
    _CHECK["name"] = "noon"
    instructions.calc_and_store_hours_above_horizon()
    best_name, best_good_hours, best_start, grid_html = _get_best_object()
    LOGGER.info("Noon check: best=%s good_hours=%d", best_name, best_good_hours)

    mode = super_user_commands.get_mode()
    social_server.post_social_message(f"Imaging mode: {mode}")

    if best_good_hours >= _MIN_GOOD_HOURS:
        slots = list(astro_dso_visibility.last_slots)
        social_server.tonight_cmd(["me", "tonight", best_name], 2, "", "")
        # The times, targets and filters follow as one timeline once the
        # sequence is written (generate_sequence_task -> _push_imaging_plan).
        social_server.post_social_message("Planning to image tonight — timeline follows")
        obs_calendar.set_today_stat('image', best_name)
        set_state(State.NOON_CHECK, best_name, True, slots=slots)
    else:
        if grid_html:
            social_server.post_html_message(grid_html)
        social_server.post_social_message(
            f"Not enough good imaging hours tonight ({best_good_hours}h), skipping"
        )
        _publish_skip_to_site("not enough clear hours tonight: %dh forecast, %d needed"
                              % (best_good_hours, _MIN_GOOD_HOURS))
        # A skipped DSO night is not a skipped station night: a pass needs two
        # clear minutes through a gap, not hours of good sky. tonight_cmd (the
        # other branch) arms via the same call, so passes arm EVERY noon.
        social_server._post_station_passes()
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
    _CHECK["name"] = "sunset"
    best_name, best_good_hours, best_start, grid_html = _get_best_object()
    if grid_html:
        social_server.post_html_message(grid_html)
    LOGGER.info("Pre-sunset check: best=%s good_hours=%d", best_name, best_good_hours)

    mode = super_user_commands.get_mode()
    social_server.post_social_message(f"Imaging mode: {mode}")

    if best_good_hours >= _MIN_GOOD_HOURS:
        slots = list(astro_dso_visibility.last_slots)
        social_server.post_social_message(
            f"Confirmed — generating sequence\n{_imaging_plan_message(best_name, best_good_hours, best_start)}"
            + _second_slot_message(slots)
        )
        social_server.post_dso_preview(best_name)
        set_state(State.PRE_SUNSET_CHECK, best_name, True, slots=slots)
    else:
        social_server.post_social_message(
            f"Conditions not good enough at sunset ({best_good_hours}h), skipping tonight"
        )
        _publish_skip_to_site("conditions degraded by sunset: %dh forecast clear, %d needed"
                              % (best_good_hours, _MIN_GOOD_HOURS))
        set_state(State.PRE_SUNSET_CHECK, best_name, False)

    return best_name, best_good_hours


@task(name="generate-nina-sequence")
def generate_sequence_task(dso_name: str, good_hours: float = 0.0, notify: bool = True,
                           slots=None):
    """Write the N.I.N.A sequence, and say what tonight will actually be.

    Two slots in the plan (astro_dso_visibility.last_slots) write a
    two-target sequence; otherwise the single-target path, with the one
    slot's window end as the hard stop. Either way the LAST target ends
    when the planner's good hours end (horizon, dawn or weather, whichever
    comes first), not at dawn regardless.

    *slots* None = the in-process ranking's (astro_dso_visibility.last_slots);
    the executor of the conductor's plan (ADR 0015) passes the plan's own.
    """
    slots = list(astro_dso_visibility.last_slots) if slots is None else list(slots)
    try:
        if len(slots) >= 2:
            got = _generate_nina_slots_sequence(slots)
            dso_name = " + ".join(s.name for s in slots)
            good_hours = float(sum(s.good_hours for s in slots))
        else:
            end = slots[0].end if slots and slots[0].name == dso_name else None
            got = _generate_nina_sequence(dso_name, good_hours, end=end)
    except sequence_lint.SequenceLintError as exc:
        # The generator refused to write a sequence N.I.N.A would skip parts
        # of (2026-09-14: every exposure block). Nothing is on disk but last
        # night's file, so say so loudly now, at noon, and let the flow fail
        # rather than image on a stale plan.
        msg = ("Sequence generation REFUSED for %s: %d lint problem(s); the file on "
               "disk is last night's. %s" % (dso_name, len(exc.problems), exc.problems[0]))
        LOGGER.error(msg)
        social_server.post_social_message("⛔ " + msg)
        pushover.push_message(msg)
        raise
    _PLAN_ON_DISK["sig"] = slot_plan.signature(slots, _filter_plans_for(slots))
    if got and notify:
        _plan, output_path = got
        _push_imaging_plan(dso_name, good_hours, _LAST_BEST_START.get("t"), output_path,
                           slots=slots, plans=_plan)


@task(name="run-imaging", timeout_seconds=9 * 60 * 60)
def imaging_task():
    """Launch the full observatory imaging run and wait for it to complete."""
    set_state(State.IMAGING)
    LOGGER.info("Starting imaging run")
    if super_user_commands.get_mode() == "auto":
        started = super_user_commands.image_cmd(["", "image!!", "1"], "iris")
        if not started:
            _CHECK["name"] = "start"
            _publish_skip_to_site("the imaging run was refused as it was about to start "
                                  "(see the chat for why)")
            # image_cmd has already said why (log, feed, push). Do not poll:
            # the state was never claimed, so it reads NONE at once and used
            # to be reported as a completed run (2026-09-13).
            LOGGER.warning("Auto imaging did not start -- see the refusal above")
            return
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

    # Generated at noon, not at sunset, so the night's plan is known and
    # notified hours in advance -- long enough to change a priority or a filter
    # plan and regenerate before anything moves.
    _LAST_BEST_START["t"] = best_start
    generate_sequence_task(best_name, good_hours)
    noon_name = best_name

    wait_for_pre_sunset_task()

    best_name, good_hours = pre_sunset_check_task()

    if good_hours < _MIN_GOOD_HOURS:
        LOGGER.info("Conditions degraded at sunset — skipping tonight")
        return

    # The pre-sunset check re-runs the ranking and can land on a different
    # PLAN than the sequence on disk was built from: a different best target,
    # a second slot gained or lost, a window moved by the forecast. Any of
    # those makes the file stale, so regenerate on any difference. Until
    # 2026-09-13 only the best target's NAME was compared, and a one-slot
    # file written by an afternoon restart ran a night whose sunset plan had
    # two slots. Silent on the happy path: one notification a night.
    # The filter plans are part of it (2026-10-09): `filters` changed after
    # noon must reach the sequence even when the slots are unchanged.
    sunset_slots = list(astro_dso_visibility.last_slots)
    sunset_sig = slot_plan.signature(sunset_slots, _filter_plans_for(sunset_slots))
    if best_name != noon_name or sunset_sig != _PLAN_ON_DISK.get("sig"):
        LOGGER.info("Plan changed since the sequence was written (%s -> %s; slots %s -> %s) — regenerating",
                    noon_name, best_name, _PLAN_ON_DISK.get("sig"), sunset_sig)
        social_server.post_social_message(
            "Plan changed since the sequence was written — regenerating:\n"
            + slot_plan.describe(list(astro_dso_visibility.last_slots)))
        generate_sequence_task(best_name, good_hours)
    else:
        # Same plan as noon, confirmed: the site's caption says so.
        _publish_plan_to_site("imaging", restamp=True)

    imaging_task()


# ---------------------------------------------------------------------------
# ADR 0015: carrying out the conductor's plan (conductor.drives_checks)
# ---------------------------------------------------------------------------
#
# With the flag on, the conductor's clock makes the noon and sunset decisions
# (iris/conductor/clock.py) and journals each as a NIGHT_PLAN note. This
# process stops ranking: it fetches the decision from GET /v1/plan and does
# everything that follows it exactly as before -- the posts, the calendar,
# the sequence from the plan's own slots, the plan picture, image!! at
# sunset. It never decides whether. An unreachable conductor is no plan, and
# no plan is no night (owner, 2026-10-16): skipping a night is the safe way
# to be wrong.

_PLAN_POLL_S = 30
_UNREACHABLE_PUSH_S = 15 * 60            # push once when the conductor is gone this long
_RESUMED_AGE_S = 5 * 60                  # a record this old at start = a restart, not news
_SUNSET_GRACE = timedelta(hours=2)       # == iris/conductor/clock.SUNSET_GRACE


def _run(task, *args, **kwargs):
    """A Prefect task's plain function. Outside a flow, calling the task would
    start Prefect's temporary API server -- the one that took the web chat's
    port on 2026-09-07 -- for nothing; the executor has no flow."""
    return getattr(task, "fn", task)(*args, **kwargs)


def _local_now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(CFG["location"]["timezone"]))


def _record_age_s(rec: dict) -> float:
    try:
        return (_local_now() - datetime.fromisoformat(rec["ts"])).total_seconds()
    except Exception:
        return 0.0


class _PlanPoller:
    """GET /v1/plan with the one-push-per-outage rule."""

    def __init__(self, day: str):
        self.day = day
        self._gone_since = None
        self._pushed = False

    def poll(self):
        from iris import client as conductor_client
        reply = conductor_client.get_plan(self.day)
        if reply is not None:
            self._gone_since, self._pushed = None, False
            return reply
        now = _local_now()                  # the cycle's clock (fakeable), not time.time()
        self._gone_since = self._gone_since or now
        if not self._pushed and (now - self._gone_since).total_seconds() > _UNREACHABLE_PUSH_S:
            msg = ("Iris: the conductor has not answered for %d min, so there is no "
                   "plan for tonight and nothing will image. Restart start_srt, then "
                   "scripts/service_check.py." % (_UNREACHABLE_PUSH_S // 60))
            LOGGER.error(msg)
            pushover.push_message(msg, priority=1)
            self._pushed = True
        return None


def _await_record(poller: _PlanPoller, check: str, until: datetime, after_seq: int = 0):
    """Poll until the *check* record (newer than *after_seq*) exists or *until*.

    -> (record or None, found_on_first_poll)."""
    first = True
    while True:
        reply = poller.poll()
        rec = (reply or {}).get(check)
        if rec and int(rec.get("seq") or 0) > after_seq:
            return rec, first
        first = False
        if _local_now() >= until:
            return None, False
        time.sleep(_PLAN_POLL_S)


def _plan_slots(plan: dict) -> list:
    from control import night_plan
    return night_plan.slots_from_record((plan or {}).get("slots") or [])


def _plan_best_start(plan: dict):
    try:
        return datetime.fromisoformat(plan["best_start"]) if plan.get("best_start") else None
    except Exception:
        return None


def _cross_check(check: str, conductor_plan: Optional[dict], post_grid: bool) -> None:
    """Rollout step 2 of ADR 0015: rank in-process too, act on nothing, journal
    whether the answer agrees. A daemon thread: if this ranking hangs (the
    2026-10-15 noon), only this thread hangs. Off: conductor.executor_cross_check."""
    if not bool((CFG.get("conductor") or {}).get("executor_cross_check", True)):
        return

    def run():
        from control import night_plan
        from iris import client as conductor_client
        try:
            best, hours, _start, grid_html = _get_best_object()
            ours = {"dso": best, "will image tonight": int(hours) >= _MIN_GOOD_HOURS,
                    "slots": _slots_record(list(astro_dso_visibility.last_slots))}
            diffs = night_plan.compare(conductor_plan or {"will_image": False}, ours)
            conductor_client.post_event(
                "EXECUTOR_PLAN_DIFF" if diffs else "EXECUTOR_PLAN_MATCH", "scheduler",
                {"check": check, "diffs": diffs, "scheduler": ours}, kind="note")
            if post_grid and grid_html:
                social_server.post_html_message(grid_html)
        except Exception:
            LOGGER.exception("executor cross-check failed (%s)", check)

    threading.Thread(target=run, daemon=True, name="executor-cross-check").start()


def _skip_noon(rec: Optional[dict], why_none: str = "") -> None:
    outcome = (rec or {}).get("outcome", "none")
    plan = (rec or {}).get("plan") or {}
    best, hours = plan.get("best") or "", int(plan.get("good_hours") or 0)
    set_state(State.NOON_CHECK, best or None, False)
    if outcome == "bad":
        social_server.post_social_message(
            f"Not enough good imaging hours tonight ({hours}h), skipping")
        _publish_skip_to_site("not enough clear hours tonight: %dh forecast, %d needed"
                              % (hours, _MIN_GOOD_HOURS))
        obs_calendar.set_today_stat('weather', best)
    else:
        why = (rec or {}).get("why") or why_none or "no plan"
        social_server.post_social_message(
            f"No imaging tonight: the conductor's noon plan is {outcome} ({why})")
        _publish_skip_to_site("no plan for tonight: %s" % why)
    social_server._post_station_passes()
    LOGGER.info("Skipping tonight — conductor's noon plan: %s", outcome)


def _carry_out_noon(rec: dict, resumed: bool):
    """The noon check's actions for the conductor's good plan. -> slots."""
    plan = rec["plan"]
    best, hours = plan["best"], int(plan["good_hours"])
    slots = _plan_slots(plan)
    _LAST_BEST_START["t"] = _plan_best_start(plan)
    _CHECK["name"] = "noon"
    if resumed:
        # A restart after noon: the night was already announced. Rebuild the
        # sequence from the SAME plan, quietly; never re-plan (2026-09-13).
        social_server.post_social_message(
            "Scheduler restarted: carrying on with the conductor's noon plan\n"
            + slot_plan.describe(slots))
        set_state(State.NOON_CHECK, best, True, slots=slots)
        _run(generate_sequence_task, best, hours, notify=False, slots=slots)
        return slots
    social_server.post_social_message(f"Imaging mode: {super_user_commands.get_mode()}")
    social_server.tonight_cmd(["me", "tonight", best], 2, "", "")
    social_server.post_social_message("Planning to image tonight — timeline follows")
    obs_calendar.set_today_stat('image', best)
    set_state(State.NOON_CHECK, best, True, slots=slots)
    _run(generate_sequence_task, best, hours, slots=slots)
    return slots


def _carry_out_replan(rec: dict):
    """A `replan` landed between noon and sunset. -> slots, or None if the night is off."""
    if rec.get("outcome") != "good":
        _skip_noon(rec)
        return None
    plan = rec["plan"]
    slots = _plan_slots(plan)
    _LAST_BEST_START["t"] = _plan_best_start(plan)
    social_server.post_social_message("Replanned by request — new plan:\n" + slot_plan.describe(slots))
    set_state(State.WAITING_FOR_PRE_SUNSET, plan["best"], True, slots=slots)
    _run(generate_sequence_task, plan["best"], int(plan["good_hours"]), slots=slots)
    return slots


def _carry_out_sunset(rec: dict, noon_name: str) -> None:
    plan = rec["plan"]
    best, hours = plan["best"], int(plan["good_hours"])
    slots = _plan_slots(plan)
    _CHECK["name"] = "sunset"
    social_server.post_social_message(f"Imaging mode: {super_user_commands.get_mode()}")
    social_server.post_social_message(
        f"Confirmed — generating sequence\n{_imaging_plan_message(best, hours, _plan_best_start(plan))}"
        + _second_slot_message(slots))
    social_server.post_dso_preview(best)
    set_state(State.PRE_SUNSET_CHECK, best, True, slots=slots)
    sig = slot_plan.signature(slots, _filter_plans_for(slots))
    if best != noon_name or sig != _PLAN_ON_DISK.get("sig"):
        LOGGER.info("Plan changed since the sequence was written (%s -> %s) — regenerating",
                    noon_name, best)
        social_server.post_social_message(
            "Plan changed since the sequence was written — regenerating:\n"
            + slot_plan.describe(slots))
        _run(generate_sequence_task, best, hours, slots=slots)
    else:
        _publish_plan_to_site("imaging", restamp=True)
    _run(imaging_task)


def driven_cycle():
    """One day carried out from the conductor's plan (ADR 0015)."""
    now = _local_now()
    day = now.date().isoformat()
    _, sunset = weather.get_sunrise_sunset()
    poller = _PlanPoller(day)
    set_state(State.NOON_CHECK)

    # --- noon
    rec, first = _await_record(poller, "noon", until=sunset)
    resumed = bool(rec) and first and _record_age_s(rec) > _RESUMED_AGE_S
    if rec is None or rec.get("outcome") != "good":
        if resumed:
            # Restarted on a day already called off: it was said at the time.
            LOGGER.info("Skipping tonight — restarted after a %s noon plan", rec.get("outcome"))
            return
        _skip_noon(rec, "the conductor gave no noon plan before the sunset check")
        return
    reply = poller.poll() or {}
    if resumed and reply.get("sunset"):
        # Restarted after tonight's sunset decision: the run was launched (or
        # refused) by the scheduler that was running then. Never launch twice.
        LOGGER.info("Restarted after the sunset check (%s): not launching",
                    reply["sunset"].get("outcome"))
        return
    slots = _carry_out_noon(rec, resumed)
    _cross_check("noon", rec.get("plan"), post_grid=False)
    noon_name, noon_seq = rec["plan"]["best"], int(rec["seq"])

    # --- the afternoon: wait for the sunset decision, carrying out any replan
    set_state(State.WAITING_FOR_PRE_SUNSET)
    social_server.post_social_message(
        "Imaging check at %s (10 min before sunset)"
        % (sunset - timedelta(minutes=10)).strftime("%H:%M %Z"))
    deadline = sunset + _SUNSET_GRACE + timedelta(minutes=15)
    srec = None
    while _local_now() < deadline:
        reply = poller.poll() or {}
        newer = reply.get("noon")
        if newer and int(newer.get("seq") or 0) > noon_seq:
            noon_seq = int(newer["seq"])
            slots = _carry_out_replan(newer)
            if slots is None:
                return
            noon_name = newer["plan"]["best"]
        if reply.get("sunset"):
            srec = reply["sunset"]
            break
        time.sleep(_PLAN_POLL_S)
    if srec is None:
        msg = "Iris: no sunset decision came from the conductor; not imaging tonight."
        LOGGER.error(msg)
        social_server.post_social_message(msg)
        pushover.push_message(msg, priority=1)
        return

    # --- sunset
    set_state(State.PRE_SUNSET_CHECK)
    _cross_check("sunset", srec.get("plan"), post_grid=True)
    if srec.get("outcome") != "good":
        plan = srec.get("plan") or {}
        why = srec.get("why") or ("%dh forecast clear, %d needed"
                                  % (int(plan.get("good_hours") or 0), _MIN_GOOD_HOURS))
        social_server.post_social_message(
            f"Conditions not good enough at sunset ({why}), skipping tonight")
        _publish_skip_to_site("conditions at sunset: %s" % why)
        set_state(State.PRE_SUNSET_CHECK, plan.get("best") or None, False)
        LOGGER.info("Conditions degraded at sunset — skipping tonight (%s)", srec.get("outcome"))
        return
    _carry_out_sunset(srec, noon_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Initialise subsystems and run the nightly scheduling loop.

    Startup sequence:
        1. PRESERVE ``safety.txt`` and ``mode.txt`` exactly as found.
        2. Reset imaging state to ``NONE`` (clears any stale ``imaging.txt``),
           unless NINA is mid-capture.
        3. Connect to the MQTT broker (continues without it if unavailable).
        4. Enter the ``while True`` loop: wait for noon, run ``nightly_cycle()``.

    Startup deliberately writes NEITHER safety.txt NOR mode.txt. It used to
    write both -- ``USER SAFE`` and ``manual`` -- which meant every restart
    (including the supervisor relaunching after a social-server crash, and the
    daily boot) silently RE-ARMED a safety flag the operator may have cleared
    on purpose, and silently dropped an operator-chosen ``auto`` back to
    manual so unattended imaging just stopped. Both flags are operator
    decisions; only operator commands change them. The readers already fail
    safe when a file is missing (``is_safe()`` -> False, ``get_mode()`` ->
    manual), so a first-ever run needs no initialisation either.
    """
    print("Starting Scheduler Server")
    LOGGER.info("Startup flags preserved: safety=%s mode=%s",
                "SAFE" if super_user_commands.is_safe() else "UNSAFE/unset",
                super_user_commands.get_mode())
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
    LOGGER.info('Start Scheduler')
    try:
        client = utils.connect_mqtt()
        client.subscribe(utils.topic_to_sched)
        client.on_message = message_handling
        client.loop_start()
    except ConnectionRefusedError:
        LOGGER.warning("MQTT broker not available — continuing without it")

    from iris import client as conductor_client
    if conductor_client.drives_checks():
        _driven_main()
        return

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


def _driven_main():
    """main()'s loop when the conductor plans (ADR 0015): wait for noon, carry
    out the day's plan, repeat. A restart after noon goes straight to the
    plan, which the conductor has (or will have) -- nothing is re-decided."""
    LOGGER.info("Scheduler carrying out the conductor's plan (conductor.drives_checks)")
    while True:
        try:
            now = _local_now()
            if now.hour < 12:
                set_state(State.WAITING_FOR_NOON)
                _wait_until(_next_noon())
            driven_cycle()
            set_state(State.WAITING_FOR_NOON)
            _wait_until(_next_noon())
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
