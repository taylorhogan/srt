import asyncio
from datetime import datetime, timedelta
import logging
import os, re, sys
import subprocess
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Optional

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from control import instructions
from hardware_control import kasa_utils as ku
from hardware_control import sonos_utils
from hardware_control import utl_shelly
from hardware_control import pwi4_utils
from hardware_control import pegasus
from cmd_processing import social_server
from cmd_processing import jobs
from utils import utils, pushover
from sentry import vision_safety
from sentry import roof_current_signature as rcs
from sentry import audio_classify as roof_audio
from end_points import end
from iris_astronomy import astro_dso_visibility
from nina_gen import nina_sequence_gen


_logger = utils.set_logger()

# Single-flight guard for SNR/convergence. Each run loads and registers
# full-frame subs across every filter; overlapping runs (a manual `snr` on top
# of the automatic end-of-night run) can exhaust memory and crash the in-process
# web server, so only one is allowed at a time.
_snr_lock = threading.Lock()

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


ROOF_TRAVEL_WAIT_S = 45   # window waited after firing the roof relay
ROOF_STALL_AFTER_S = 30   # motor still drawing power after this ⇒ gear not engaged
ROOF_STALL_MIN_W = 20.0   # act_power above this = motor running (idle ~2.7 W, moving ~300 W)


class RoofStallError(RuntimeError):
    """Roof motor still drawing power ROOF_STALL_AFTER_S after the relay fired."""


def _roof_stall_abort(dev_map: dict, direction: Optional[str],
                      capture, audio_capture, watts: float, elapsed: float) -> None:
    """Emergency path for a roof-motor stall: cut power, alert, bank evidence, abort.

    The drive gear can fail to engage the thread — the motor then spins without
    moving the roof until the end-of-window power-off. Order matters here: the
    Kasa power cut is the safety action and comes before any notification.
    Always raises RoofStallError.
    """
    power_cut = True
    try:
        asyncio.run(ku.kasa_do(dev_map, {"Roof motor": 'off'}))
    except Exception:  # noqa: BLE001
        power_cut = False
        _logger.exception("roof stall: FAILED to cut roof motor power")
    msg = (f"🚨 EMERGENCY: roof motor still drawing {watts:.0f} W {elapsed:.0f}s after the "
           f"{direction or 'move'} trigger — drive gear may not have engaged. "
           + ("Motor power cut." if power_cut
              else "POWER CUT FAILED — motor may still be running!")
           + " Roof state UNKNOWN — inspect before any further roof or scope moves.")
    _logger.error(msg)
    try:
        pushover.push_message(msg, priority=2)
    except Exception:  # noqa: BLE001
        _logger.exception("roof stall: pushover emergency failed")
    try:
        social_server.post_social_message(msg)
    except Exception:  # noqa: BLE001
        _logger.exception("roof stall: social post failed")
    # Bank the evidence: the current trace and audio of the stalled move are
    # exactly what post-mortem needs. Both helpers swallow their own errors.
    if capture is not None:
        rcs.finish_background_capture(capture, status="unlabeled")
    roof_audio.finish_background_capture(audio_capture, status="unlabeled")
    raise RoofStallError(msg)


def _wait_for_roof_travel(dev_map: dict, capture_direction: Optional[str],
                          capture, audio_capture) -> None:
    """Wait out the roof travel window, aborting on a motor stall.

    A normal move runs the motor for only ~11 s (banked signatures: ~300 W
    running vs ~2.7 W idle), so sustained draw past ROOF_STALL_AFTER_S means
    the roof is not travelling. Two consecutive over-threshold readings are
    required so a single bad sample can't cut power spuriously; failed monitor
    reads never trigger — without data this degrades to the plain timed wait.
    """
    if not config.data()["hardware"].get("current_monitor_url"):
        time.sleep(ROOF_TRAVEL_WAIT_S)
        return
    start = time.monotonic()
    time.sleep(ROOF_STALL_AFTER_S)
    consecutive = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= ROOF_TRAVEL_WAIT_S:
            return
        status = utl_shelly.read_current_monitor()
        power = status.get("act_power") if status is not None else None
        if power is None:
            _logger.warning("roof stall watchdog: current read failed at t=%.0fs", elapsed)
            consecutive = 0
        elif power > ROOF_STALL_MIN_W:
            consecutive += 1
            _logger.warning("roof stall watchdog: %.0f W at t=%.0fs (%d/2)",
                            power, elapsed, consecutive)
            if consecutive >= 2:
                _roof_stall_abort(dev_map, capture_direction, capture,
                                  audio_capture, power, elapsed)
        else:
            consecutive = 0
        time.sleep(1)


def toggle_roof(dev_map: dict, capture_direction: Optional[str] = None) -> None:
    """Power the roof motor, trigger the Shelly relay to move the roof, then power off.

    The roof direction (open/close) depends on its current position — the relay
    simply toggles. Waits 45 seconds for the roof to complete its travel, with a
    stall watchdog: if the motor is still drawing power after ROOF_STALL_AFTER_S
    (gear not engaged), power is cut and RoofStallError aborts the caller.

    `capture_direction` ("open"/"close") only labels the banked current
    signature; it does not change which way the roof moves.
    """
    inst = {"Roof motor": 'on'}
    asyncio.run(ku.kasa_do(dev_map, inst))
    time.sleep(10)

    # Best-effort: bank the motor's current signature for anomaly detection.
    # Never let a capture problem disrupt the roof sequence (the helpers swallow
    # their own errors); only start it if a current monitor is configured.
    capture = None
    if config.data()["hardware"].get("current_monitor_url"):
        capture = rcs.start_background_capture(direction=capture_direction, seconds=48)
    # Also bank the roof-move audio (spectrogram + WAV), filed by direction. The
    # mic captures fast mechanical chatter (e.g. a bent wheel) that the 1 Hz
    # current signature cannot resolve. Best-effort: the helper swallows its own
    # errors and the stream is closed before the post-move vision check, so mic
    # capture never overlaps a webcam snapshot.
    audio_capture = roof_audio.start_background_capture(direction=capture_direction)

    if utl_shelly.fire_roof_relay() is None:
        _logger.error("Failed to trigger relay in toggle_roof")
        if capture is not None:
            rcs.finish_background_capture(capture, save=False)
        roof_audio.finish_background_capture(audio_capture, save=False)
        raise RuntimeError("toggle_roof: roof relay trigger failed")
    _wait_for_roof_travel(dev_map, capture_direction, capture, audio_capture)
    inst = {"Roof motor": 'off'}
    asyncio.run(ku.kasa_do(dev_map, inst))

    # Finish + surface the roof-move audio spectrogram, with a verdict from the
    # known-good library. Announce-only for now: the capture stays in unlabeled/
    # regardless of verdict; the user files it in the morning with
    # `audio <open|close> <good|bad>`. Once the library is trusted this can
    # switch to auto-filing. classify() never raises (returns "unknown").
    audio_result = roof_audio.finish_background_capture(audio_capture, status="unlabeled")
    if audio_result and audio_result.get("spectrogram"):
        cls = roof_audio.classify(audio_result["spectrogram"],
                                  audio_result.get("direction"))
        caption = f"Roof {capture_direction or 'move'} audio"
        if cls["verdict"] == "good":
            caption += (f": sounds normal (score {cls['best_score']:.3f} ≥ "
                        f"{cls['threshold']:.3f}, best match {cls['best_match']})")
        elif cls["verdict"] == "bad":
            caption = (f"⚠️ {caption}: does NOT match known-good "
                       f"(score {cls['best_score']:.3f} < {cls['threshold']:.3f})")
            try:
                pushover.push_message(caption)
            except Exception as e:  # noqa: BLE001
                _logger.error("Failed to push roof audio anomaly: %s", e)
        else:
            caption += f" — {cls['note'] or 'not classified'}"
        try:
            # Attach the WAV alongside the spectrogram: the webchat renders an
            # inline player plus download links for both files, so a move's raw
            # audio can be pulled for offline analysis.
            social_server.post_social_message(
                caption,
                image=audio_result["spectrogram"],
                audio=audio_result.get("wav"),
            )
        except Exception as e:  # noqa: BLE001
            _logger.error("Failed to post roof audio spectrogram: %s", e)

    if capture is not None:
        sig = rcs.finish_background_capture(capture, status="unlabeled")
        if sig is not None:
            res = rcs.compare(sig)
            if res.get("is_anomaly"):
                reasons = "; ".join(res["reasons"])
                _logger.warning("Roof current signature anomaly: %s", reasons)
                # Surface to chat + phone so a stress/jam isn't buried in the log
                # during unattended operation. Best-effort: a notification failure
                # must not break the roof sequence.
                alert = f"⚠️ Roof motor current anomaly ({capture_direction or 'unknown'}): {reasons}"
                try:
                    social_server.post_social_message(alert)
                except Exception as e:  # noqa: BLE001
                    _logger.error("Failed to post roof anomaly to chat: %s", e)
                try:
                    pushover.push_message(alert)
                except Exception as e:  # noqa: BLE001
                    _logger.error("Failed to push roof anomaly: %s", e)



def announce_roof_movement(text: str, speaker_name: str = "Observatory", volume: int = 40) -> None:
    """Announce upcoming roof movement via Sonos."""
    try:
        sonos_utils.sonos_say(text, speaker_name, volume)
    except Exception as e:
        _logger.error("Sonos announcement failed: %s", e)


def get_status_with_lights() -> tuple[bool, bool, bool, Any]:
    """Take a camera snapshot and return (parked, closed, open, mod_date) via vision safety."""
    parked, closed, open, mod_date = vision_safety.visual_status()
    _post_vision_decision_image(parked, closed, open)
    return parked, closed, open, mod_date


def _post_vision_decision_image(parked: bool, closed: bool, is_open: bool) -> None:
    """Push the annotated snapshot vision just used to decide scope/roof state.

    Lets a wrong read (e.g. the closed/open templates both matching) be eyeballed
    and the templates/thresholds recalibrated from real images. Best-effort: a
    notification failure must never disrupt a safety check or roof movement.
    """
    try:
        cfg = config.data()
        img = cfg["camera safety"]["scope_view"]
        lm = vision_safety.last_match or {}
        roof = "closed" if closed else ("open" if is_open else "unknown")
        caption = (
            f"Vision: roof={roof} scope={'parked' if parked else 'unparked'} | "
            f"conf c={lm.get('closed', {}).get('conf', 0):.2f} "
            f"o={lm.get('open', {}).get('conf', 0):.2f} "
            f"p={lm.get('parked', {}).get('conf', 0):.2f} "
            f"luma={lm.get('frame_luma', 0):.0f} trusted={lm.get('trusted')}"
        )
        pushover.push_message(caption, img)
    except Exception:
        _logger.exception("failed to push vision decision image")


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
                toggle_roof(dev_map, capture_direction="open")
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
        toggle_roof(dev_map, capture_direction="open")
        return False


def close_roof_with_option(check: bool) -> bool:
    """Close the observatory roof. If *check* is True, verify the scope is parked
    via vision safety before toggling — the absolute rule is that the roof must
    never move unless the scope is confirmed parked (collision risk). Returns
    True only when the roof is confirmed closed afterward.
    """
    dev_map = asyncio.run(ku.make_discovery_map())
    if check:
        parked, closed, is_open, mod_date = get_status_with_lights()
        if not parked:
            social_server.post_social_message("Vision Safety says Scope is NOT parked, therefore will not close")
            return False
        if closed:
            social_server.post_social_message("Vision Safety says roof is already closed")
            return True
        social_server.post_social_message("Vision Safety says scope is parked, closing roof")
        announce_roof_movement("The roof will be closing in one minute")
        toggle_roof(dev_map, capture_direction="close")
        time.sleep(30)
        MAX_ROOF_CHECKS = 5
        for attempt in range(MAX_ROOF_CHECKS):
            parked, closed, is_open, mod_date = get_status_with_lights()
            if closed:
                return True
            if attempt < MAX_ROOF_CHECKS - 1:
                msg = f"Roof close not confirmed (attempt {attempt + 1}/{MAX_ROOF_CHECKS}), waiting 5 min"
                social_server.post_social_message(msg)
                _logger.warning(msg)
                time.sleep(5 * 60)
        social_server.post_social_message(f"Roof could not be confirmed closed after {MAX_ROOF_CHECKS} attempts, stopping")
        _logger.warning("Roof close check failed after %d attempts", MAX_ROOF_CHECKS)
        return False
    else:
        toggle_roof(dev_map, capture_direction="close")
        return False


def _roof_cmd_blocked_reason() -> str | None:
    """Return why a ``roof!!`` command must be ignored, or None if it may run.

    Shared precondition for every ``roof!!`` variant — status included: the roof
    may only be touched when no imaging is in progress AND the telescope mount is
    powered off. A powered mount may be tracking, putting the scope in the roof's
    travel path. The mount-power check is the same Kasa ``isoff`` probe used by
    ``open_if_mount_off_cmd``; if it can't be confirmed we fail safe (refuse).
    """
    if is_imaging() or is_nina_running():
        return "an imaging run is in progress (imaging state is not none)"
    try:
        dev_map = asyncio.run(ku.make_discovery_map())
        mount_off = asyncio.run(ku.kasa_check(dev_map, {"Telescope mount": "isoff"}))
    except Exception as exc:
        _logger.warning("roof!! gate: mount power check failed: %s", exc)
        return f"could not confirm the telescope mount is powered off ({exc})"
    if not mount_off:
        return "the telescope mount is powered on"
    return None


def roof_cmd(words: list[str], account: str) -> None:
    """Move or report the observatory roof. Command: ``roof!! open|close|toggle|status [force]``

    The ``!!`` flags that this command can move hardware (the roof, and indirectly
    the scope's collision envelope), mirroring ``image!!``.

    Subcommands:
        ``roof!! status``  — report scope/roof position via vision safety (no movement).
        ``roof!! open``    — safety-checked open: requires scope parked + roof closed,
                             then opens and confirms via vision.
        ``roof!! close``   — safety-checked close: requires scope parked, then closes
                             and confirms via vision.
        ``roof!! toggle``  — single relay toggle (the hardware just toggles; direction
                             depends on current position). Parked-checked; when the
                             current position is known the direction it will travel is
                             announced.
        append ``force`` to ``open``/``close``/``toggle`` to skip the scope-parked
        vision check (DANGEROUS — collision risk; use only when you can physically
        see the scope is parked).

    SAFETY: every movement path verifies the scope is parked via vision safety
    first and refuses to move unless confirmed, unless ``force`` is given. Movement
    runs on a background thread so the chat stays responsive.
    """
    sub = words[2] if len(words) >= 3 else ""
    force = len(words) >= 4 and words[3] == "force"

    if sub not in ("status", "open", "close", "toggle"):
        social_server.post_social_message("Usage: roof!! open|close|toggle|status [force]")
        return

    # Gate EVERY roof!! variant (status included): only act when imaging state is
    # none AND the telescope mount is powered off. Either condition failing means
    # the roof must not be touched — even a read-only status snapshot is skipped —
    # so report the reason and bail. ``force`` only waives the scope-parked vision
    # check below; it does NOT waive this gate.
    blocked = _roof_cmd_blocked_reason()
    if blocked is not None:
        social_server.post_social_message(f"Ignoring roof!! {sub}: {blocked}")
        return

    if sub == "status":
        # Read-only snapshot. The process-wide camera lock in
        # inside_camera_server serializes it against any in-flight job.
        parked, closed, is_open, mod_date = get_status_with_lights()
        roof_state = "closed" if closed else ("open" if is_open else "ambiguous")
        social_server.post_social_message(
            f"Roof: {roof_state}; scope: {'parked' if parked else 'NOT parked'} "
            f"(vision @ {mod_date})"
        )
        return

    # Never let a roof movement overlap an imaging run or another roof command.
    # The original incident was a `close` issued while an `open` was still in its
    # confirm loop: both jobs raced the single USB camera and the close crashed.
    if is_imaging() or is_nina_running():
        social_server.post_social_message("Cannot move the roof: an imaging run is in progress")
        return
    if not _roof_lock.acquire(blocking=False):
        social_server.post_social_message("Cannot move the roof: another roof command is already running")
        return

    if force:
        social_server.post_social_message(
            f"⚠️ roof!! {sub} FORCE — skipping the scope-parked safety check"
        )

    def _run() -> None:
        try:
            if sub == "open":
                # Announce here (not inside open_roof_with_option) so the imaging
                # run, which announces separately before calling it, doesn't double
                # up — and so a manual `roof!! open` still speaks like close/toggle.
                announce_roof_movement("The roof will be opening in one minute")
                ok = open_roof_with_option(check=not force)
                if force:
                    social_server.post_social_message("Roof open relay fired (forced, unverified)")
                elif ok:
                    social_server.post_social_message("✅ Roof successfully opened")
                else:
                    social_server.post_social_message("❌ Roof failed to open")
            elif sub == "close":
                ok = close_roof_with_option(check=not force)
                if force:
                    social_server.post_social_message("Roof close relay fired (forced, unverified)")
                elif ok:
                    social_server.post_social_message("✅ Roof successfully closed")
                else:
                    social_server.post_social_message("❌ Roof failed to close")
            else:  # toggle
                _toggle_roof_cmd(force)
        finally:
            _roof_lock.release()

    try:
        jobs.spawn(_run)
    except Exception:
        # spawn failed before the worker took ownership of the lock — release it
        # here so a roof command can never be permanently wedged.
        _roof_lock.release()
        raise


def _toggle_roof_cmd(force: bool) -> None:
    """Body of ``roof toggle``: a single parked-checked relay toggle.

    The roof relay only toggles, so the travel direction depends on the current
    position. When that position is known (vision), the direction is inferred,
    announced, and used to label the banked motor current signature.
    """
    dev_map = asyncio.run(ku.make_discovery_map())
    direction = None
    if not force:
        parked, closed, is_open, _ = get_status_with_lights()
        if not parked:
            social_server.post_social_message("Vision Safety says Scope is NOT parked, refusing to toggle roof")
            return
        if closed:
            direction = "open"
        elif is_open:
            direction = "close"
        cur = "closed" if closed else ("open" if is_open else "ambiguous")
        going = f"{direction}ing" if direction else "moving (direction unknown)"
        social_server.post_social_message(f"Scope parked; roof currently {cur} — {going}")
    announce_roof_movement("The roof will be moving in one minute")
    toggle_roof(dev_map, capture_direction=direction)
    parked, closed, is_open, _ = get_status_with_lights()
    new_state = "closed" if closed else ("open" if is_open else "ambiguous")
    # Simple success/failure line: did the roof reach the intended direction?
    # When direction is unknown (forced toggle, or an ambiguous pre-state), we
    # can't claim success against an intent, so just report the resulting state.
    if direction == "open":
        msg = "✅ Roof successfully opened" if is_open else f"❌ Roof failed to open — now {new_state}"
    elif direction == "close":
        msg = "✅ Roof successfully closed" if closed else f"❌ Roof failed to close — now {new_state}"
    else:
        msg = f"Roof toggled — now {new_state}"
    social_server.post_social_message(msg)


def unsafe_cmd(words: list[str], account: str) -> None:
    """Emergency stop: kill NINA, park the scope, close the roof, and shut down.

    Replaces the old soft "mark unsafe" behavior. Brings the observatory to a
    safe state from any point in a run while honoring the absolute hardware
    rules — the scope is parked only with the roof confirmed open, and the roof
    is closed only with the scope confirmed parked. If the scope cannot be
    confirmed parked, the roof is LEFT OPEN and an alert is sent rather than
    risking a collision. Runs on a background thread so the chat stays
    responsive and shows a progress card.
    Command: ``stop!``
    """
    social_server.post_social_message(
        "⛔ EMERGENCY STOP — killing NINA, parking scope, closing roof, shutting down"
    )
    pushover.push_message("EMERGENCY STOP initiated")
    jobs.spawn(_emergency_stop_sequence)


def _power_off_pegasus_train() -> None:
    """Final shutdown step: power off the Pegasus imaging-train ports.

    Called LAST because the vision-safety camera is powered through the Pegasus
    box — cutting these ports blinds it, so it must happen only after all
    roof/mount/vision work is complete. Never raises.
    """
    try:
        if pegasus.power_off_imaging_train():
            _logger.info("emergency: Pegasus imaging-train ports powered off")
        else:
            _logger.warning("emergency: Pegasus power-off not fully acknowledged")
    except Exception:
        _logger.exception("emergency: Pegasus power-off failed")


def _emergency_stop_sequence() -> None:
    """Body of the emergency stop (see :func:`unsafe_cmd`).

    Never closes the roof unless the scope is confirmed parked. The authoritative
    parked check is mount telemetry (``pwi4_utils.get_is_parked``); vision tells
    us the roof position and needs the inside light on to be reliable.
    """
    utils.set_install_dir()
    inside_view = config.data()["camera safety"]["scope_view"]

    # 1. Signal the running imaging worker to unwind (skips the flats sequence),
    #    and set the safety flag so any safety-gated code also bails.
    request_abort()
    try:
        with open("safety.txt", "w") as file:
            file.write("USER UNSAFE")
    except Exception:
        _logger.exception("emergency: failed to write safety.txt")

    # 2. Kill NINA only — PWI4 must stay alive so we can park the mount.
    _kill_nina()
    try:
        from fits_processing import frame_watcher
        frame_watcher.stop()
    except Exception:
        _logger.exception("emergency: frame_watcher.stop failed")

    # 3. Read observatory state. Turn the inside light on first so vision is
    #    reliable (during imaging the room is dark); the parked check uses mount
    #    telemetry and is light-independent.
    dev_map = None
    try:
        dev_map = asyncio.run(ku.make_discovery_map())
        turn_inside_light_on(dev_map)
        time.sleep(30)  # let the camera adjust to the lit room before vision
    except Exception:
        _logger.exception("emergency: could not turn on inside light")

    parked, closed, is_open, _ = get_status_with_lights()
    mount_parked = pwi4_utils.get_is_parked()
    _logger.info(
        "emergency: vision parked=%s closed=%s open=%s; mount_parked=%s",
        parked, closed, is_open, mount_parked,
    )

    # 4. Branch on roof position (safety-critical).
    if is_open:
        # Roof confirmed open → safe to slew/park the scope.
        if not mount_parked:
            social_server.post_social_message("Roof open — parking scope")
            pwi4_utils.park_scope()
            mount_parked = pwi4_utils.get_is_parked()

        if mount_parked:
            # Parked + roof open → end.do_main() closes the roof + full shutdown
            # (it re-confirms parked via vision before toggling the roof).
            social_server.post_social_message("Scope parked — closing roof and shutting down")
            end.do_main()
            # LAST: blinds the vision camera (powered through the Pegasus box).
            _power_off_pegasus_train()
        else:
            # Never close the roof over a scope we can't confirm is parked.
            msg = "EMERGENCY: scope would NOT park — roof LEFT OPEN, manual help needed"
            social_server.post_social_message(msg)
            pushover.push_message(msg, inside_view)
            set_imaging_state(ImagingState.NONE)

    elif closed:
        # Roof already closed.
        if mount_parked:
            # Already a safe geometry → power down only. Do NOT call end.do_main()
            # here: it unconditionally toggles the roof when parked, which would
            # re-open a closed roof.
            social_server.post_social_message("Roof already closed, scope parked — powering down")
            try:
                if dev_map is not None:
                    asyncio.run(ku.kasa_do(dev_map, {
                        "Telescope mount": 'off',
                        "Roof motor": 'off',
                        "Iris inside light": 'off',
                    }))
                utl_shelly.set_dehumidifier(True)
            except Exception:
                _logger.exception("emergency: lightweight shutdown failed")
            # LAST: blinds the vision camera (powered through the Pegasus box).
            _power_off_pegasus_train()
            set_imaging_state(ImagingState.NONE)
        else:
            # Unparked scope under a closed roof — no safe automatic action.
            msg = "EMERGENCY: roof closed but scope NOT parked — manual help needed"
            social_server.post_social_message(msg)
            pushover.push_message(msg, inside_view)
            set_imaging_state(ImagingState.NONE)

    else:
        # Roof position ambiguous (neither confidently open nor closed).
        msg = "EMERGENCY: roof position ambiguous — no safe automatic action, manual help needed"
        social_server.post_social_message(msg)
        pushover.push_message(msg, inside_view)
        set_imaging_state(ImagingState.NONE)


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


# ── Emergency abort signal ─────────────────────────────────────────
# Cooperative stop flag observed by doit_cmd's poll loops so an emergency
# stop (stop!) can unwind a running imaging job cleanly — in particular,
# WITHOUT falling through into the flats sequence, which would power the
# mount back on and relaunch NINA. Cleared by image_cmd/doflats_cmd at the
# start of every fresh run so a stale abort can never kill a new run.
_abort_event = threading.Event()

# Serializes roof movement (open/close/toggle). Acquired non-blocking in the
# dispatch thread so a second roof command is refused rather than queued, and
# released by the worker thread when the movement finishes. A threading.Lock has
# no owner thread, so cross-thread release is legal.
_roof_lock = threading.Lock()


def request_abort() -> None:
    """Signal any in-progress imaging run to abort at its next checkpoint."""
    _abort_event.set()


def clear_abort() -> None:
    """Clear the abort flag. Called when starting a fresh imaging/flats run."""
    _abort_event.clear()


def is_aborting() -> bool:
    """Return True if an emergency abort has been requested."""
    return _abort_event.is_set()


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
        if utl_shelly.fire_roof_relay() is None:
            _logger.error("Failed to trigger relay in open_if_mount_off_cmd")
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


def is_nina_running() -> bool:
    """Return True if a NINA.exe process is currently running.

    Used as a hardware-truth check independent of ``imaging.txt`` (the state
    file can be cleared by a process restart while NINA keeps imaging), so we
    never launch a second imaging run on top of a live one.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq NINA.exe"],
            capture_output=True, text=True, shell=True,
        )
        return "NINA.exe" in (result.stdout or "")
    except Exception:
        _logger.exception("is_nina_running check failed")
        return False


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

    dso = instructions.resolve_target_by_name(dso_name)
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

    jobs.spawn(_do_update)


def optics_cmd(words: list[str], account: str) -> None:
    """Post optical quality diagnostic plots for a FITS frame.

    Usage:
        optics              — latest frame of the last-imaged DSO
        optics <dso>        — latest frame of the named DSO
        optics * <n>        — frame n of the last-imaged DSO
        optics <dso> <n>    — frame n of the named DSO

    Process-isolated (jobs.spawn_process) so the analysis runs on its own core,
    in true parallel with a concurrent command.
    """
    jobs.spawn_process(_optics_run, args=(words,))


def _optics_run(words: list[str]) -> None:
    from fits_processing import fitsfwhm, sky_brightness as sb

    _job_id = jobs.get_current_job()

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

    jobs.raise_if_cancelled(_job_id)
    metrics = fitsfwhm.compute_optical_metrics(fits_path, arcsec_per_pixel=arcsec_per_pixel)
    if metrics:
        metrics_table_path = Path(os.path.join(scratch_dir, "optical_metrics_table.jpg"))
        fitsfwhm.save_optical_metrics_table(metrics, metrics_table_path)
        social_server.post_social_message("Optical quality metrics", str(metrics_table_path))

    jobs.raise_if_cancelled(_job_id)
    fwhm_heatmap_path = Path(os.path.join(scratch_dir, "fwhm_heatmap.jpg"))
    ecc_heatmap_path  = Path(os.path.join(scratch_dir, "ecc_heatmap.jpg"))
    fwhm_out, ecc_out = fitsfwhm.save_fwhm_heatmaps(
        fits_path, fwhm_heatmap_path, ecc_heatmap_path,
        arcsec_per_pixel=arcsec_per_pixel,
    )
    social_server.post_social_message("FWHM heatmap", str(fwhm_out))
    social_server.post_social_message("Eccentricity heatmap", str(ecc_out))

    jobs.raise_if_cancelled(_job_id)
    dist_plot_path = Path(os.path.join(scratch_dir, "fwhm_vs_distance.jpg"))
    dist_out = fitsfwhm.save_fwhm_vs_distance(
        fits_path, dist_plot_path, arcsec_per_pixel=arcsec_per_pixel,
    )
    social_server.post_social_message("FWHM vs distance from centre", str(dist_out))

    jobs.raise_if_cancelled(_job_id)
    angle_map_path = Path(os.path.join(scratch_dir, "ecc_angle_map.jpg"))
    angle_out = fitsfwhm.save_eccentricity_angle_map(
        fits_path, angle_map_path, arcsec_per_pixel=arcsec_per_pixel,
    )
    social_server.post_social_message("Elongation angle map", str(angle_out))

    jobs.raise_if_cancelled(_job_id)
    sky_heatmap_path = Path(os.path.join(scratch_dir, "sky_heatmap.jpg"))
    sky_heatmap_out, sky_data = sb.save_sky_heatmap(
        fits_path, sky_heatmap_path, arcsec_per_pixel=arcsec_per_pixel,
    )
    if sky_data:
        social_server.post_social_message(sb.sky_summary_text(sky_data), str(sky_heatmap_out))


def active_cmd(words: list[str], account: str) -> None:
    """Show, per DSO, a date×filter grid of how many LIGHT subs were taken."""
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

    # results[dso][observing_night][filter] = sub count
    results: dict[str, dict[str, dict[str, int]]] = {}
    for dso_dir in dso_dirs:
        grid: dict[str, dict[str, int]] = {}
        for f in dso_dir.rglob("*.fits"):
            if f.parent.name.upper() != "LIGHT":
                continue
            night, filter_name = _frame_night_filter(f)
            row = grid.setdefault(night, {})
            row[filter_name] = row.get(filter_name, 0) + 1
        if grid:
            results[dso_dir.name] = grid

    if not results:
        social_server.post_social_message("No LIGHT frames found in image directory")
        return

    social_server.post_html_message(_active_tiles_html(results))


# Canonical filter order for the active/live displays; anything else falls in
# afterwards alphabetically. Mirrors FILTER_ORDER in static/chat.html.
_FILTER_ORDER = ["L", "R", "G", "B", "Ha", "OIII", "SII"]

# NINA layout: <DSO>/<scope>/<night>/LIGHT/<date>_<time>_<FILTER>_<...>.fits
# (e.g. 2026-05-17_22-44-45_L_158_300.00s_0073.fits). The night folder already
# follows the noon-rollover convention, so it doubles as the observing night.
_DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FNAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_([^_]+)_")


def _frame_night_filter(f: Path) -> tuple[str, str]:
    """Return (observing_night, filter) for a LIGHT frame.

    Fast path parses both from the NINA path/filename. Frames with a
    non-standard name (e.g. transit captures) fall back to reading the FITS
    header, then to file mtime, so they still land in a sensible date column.
    """
    night_dir = f.parent.parent.name           # folder above LIGHT
    m = _FNAME_RE.match(f.name)
    if m and _DATE_RE.match(night_dir):
        return night_dir, m.group(1)

    filt = "Unknown"
    try:
        from astropy.io import fits
        hdr = fits.getheader(f)
        filt = str(hdr.get("FILTER", "Unknown")).strip() or "Unknown"
        date_obs = hdr.get("DATE-OBS")
        if date_obs:
            dt = datetime.fromisoformat(str(date_obs).rstrip("Z"))
            return (dt - timedelta(hours=12)).date().isoformat(), filt
    except Exception:
        pass

    night = night_dir if _DATE_RE.match(night_dir) else None
    if night is None:
        try:
            night = (datetime.fromtimestamp(f.stat().st_mtime)
                     - timedelta(hours=12)).date().isoformat()
        except OSError:
            night = "unknown"
    return night, filt


def _filter_sort_key(name: str):
    try:
        return (0, _FILTER_ORDER.index(name))
    except ValueError:
        return (1, name.lower())


def _active_tiles_html(results: dict[str, dict[str, dict[str, int]]]) -> str:
    """Render one tile per DSO: a date×filter grid of sub counts.

    Each tile is a self-contained card — columns are filters, rows are
    observing nights (newest first), the cell is the number of subs, with a
    per-night Σ column and a per-filter totals footer. Tiles wrap to fill the
    available width.
    """
    from html import escape

    BORDER, ROW, DIM, TEXT, ACCENT, BRIGHT = (
        "#30363d", "#21262d", "#8b949e", "#c9d1d9", "#3fb950", "#e6edf3")
    TILE_BG, TILE_BORDER, DOT = "#0d1117", "#30363d", "#475061"

    th = (f'padding:3px 7px;border-bottom:1px solid {BORDER};color:{DIM};'
          f'font-size:10.5px;font-weight:600;white-space:nowrap;')
    td = f'padding:3px 7px;border-bottom:1px solid {ROW};color:{TEXT};font-variant-numeric:tabular-nums;'

    out = [
        f'<div style="font-size:13px;font-weight:600;color:{ACCENT};margin-bottom:10px;">'
        f'Active targets — {len(results)} object{"s" if len(results) != 1 else ""}</div>',
        '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;">',
    ]

    for dso, grid in sorted(results.items()):
        filters = sorted({f for row in grid.values() for f in row}, key=_filter_sort_key)
        dso_total = sum(sum(row.values()) for row in grid.values())

        t = [
            f'<div style="background:{TILE_BG};border:1px solid {TILE_BORDER};'
            f'border-radius:8px;padding:10px 12px;min-width:240px;">',
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'gap:14px;margin-bottom:7px;">'
            f'<span style="font-size:12.5px;font-weight:600;color:{BRIGHT};white-space:nowrap;">{escape(dso)}</span>'
            f'<span style="font-size:11px;color:{DIM};">{dso_total} subs</span></div>',
            '<table style="border-collapse:collapse;width:100%;font-size:11px;">',
            f'<thead><tr><th style="{th}text-align:left;">Date</th>',
        ]
        for f in filters:
            t.append(f'<th style="{th}text-align:right;">{escape(f)}</th>')
        t.append(f'<th style="{th}text-align:right;">Σ</th></tr></thead><tbody>')

        for night in sorted(grid.keys(), reverse=True):
            row = grid[night]
            night_total = sum(row.values())
            t.append(f'<tr><td style="{td}white-space:nowrap;color:{DIM};">{escape(night)}</td>')
            for f in filters:
                n = row.get(f)
                cell = str(n) if n else f'<span style="color:{DOT};">·</span>'
                t.append(f'<td style="{td}text-align:right;">{cell}</td>')
            t.append(f'<td style="{td}text-align:right;color:{BRIGHT};font-weight:600;">{night_total}</td></tr>')

        grand = {f: sum(row.get(f, 0) for row in grid.values()) for f in filters}
        foot = f'padding:4px 7px;border-top:2px solid {BORDER};font-weight:600;'
        t.append(f'<tr><td style="{foot}color:{DIM};">All</td>')
        for f in filters:
            t.append(f'<td style="{foot}text-align:right;color:{DIM};">{grand[f]}</td>')
        t.append(f'<td style="{foot}text-align:right;color:{ACCENT};font-weight:700;">{dso_total}</td></tr>')

        t.append('</tbody></table></div>')
        out.append("".join(t))

    out.append('</div>')
    return "".join(out)


def drift_cmd(words: list[str], account: str) -> None:
    """Post ZScale-stretched difference images: first-k-frames stack vs golden (L filter only).

    Usage:
        drift           — L frames of the last-imaged DSO
        drift <dso>     — L frames of the named DSO
        drift *         — same as bare drift (last DSO)
    """
    jobs.spawn(_drift_run, args=(words,))


def _drift_run(words: list[str]) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval
    from stacking import stacker

    _job_id = jobs.get_current_job()
    _cancel = jobs.cancel_cb_for(_job_id)

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
        l_files, progress_cb=_progress, cancel_cb=_cancel,
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

    jobs.raise_if_cancelled(_job_id)
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

    jobs.raise_if_cancelled(_job_id)
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

    Process-isolated (jobs.spawn_process) so the heavy stacking runs on its own
    core, in true parallel with a concurrent command.
    """
    jobs.spawn_process(_stack_run, args=(words,))


def _stack_run(words: list[str]) -> None:
    from stacking import stacker
    from fits_processing import fitsfwhm
    from astropy.io import fits as _fits

    _job_id = jobs.get_current_job()
    _cancel = jobs.cancel_cb_for(_job_id)

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
        jobs.raise_if_cancelled(_job_id)
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
                progress_cb=_progress,
                cancel_cb=_cancel,
            )
        except jobs.Cancelled:
            raise
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


# Frame quality is judged relative to the per-filter median rather than against
# absolute thresholds, so the cuts adapt to the night's seeing and each filter's
# intrinsic star yield. A frame fails if it has no stars, fewer than half the
# median star count, more than 1.5x the median FWHM, or more than 1.5x the median
# eccentricity (all medians taken within the frame's own filter).
_BAD_STAR_FRACTION = 0.5   # reject frames below this fraction of the median star count
_BAD_FWHM_FACTOR = 1.5     # reject frames above this multiple of the median FWHM
_BAD_ECC_FACTOR = 1.5      # reject frames above this multiple of the median eccentricity


def bad_cmd(words: list[str], account: str) -> None:
    """Rename LIGHT frames that fail quality thresholds to .bad.

    Quality is judged per-filter, relative to that filter's median. A frame
    fails if it has no stars, fewer than 50% of the median star count, more
    than 1.5x the median FWHM, or more than 1.5x the median eccentricity.
    Renames `<frame>.fits` to `<frame>.fits.bad` so they're skipped by
    the stacker (which only globs *.fits). Reverse with `dab`.

    Reuses the frame_stats.json cache populated by the `stats` command;
    any uncached frames are measured on the spot and written back to the
    cache.

    Usage:
        bad                 — dry-run on last-imaged DSO
        bad <dso>           — dry-run on named DSO
        bad <dso> go        — actually rename (default is dry-run)
        bad go              — dry-run on last DSO (use `bad * go` to rename)
    """
    jobs.spawn(_bad_run, args=(words,))


def _bad_run(words: list[str]) -> None:
    import json as _json
    import warnings as _warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from fits_processing import fitsfwhm

    _job_id = jobs.get_current_job()

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
            jobs.register_resource(_job_id, pool)
            try:
                future_map = {pool.submit(_analyse, f): i for i, f in enumerate(need_analysis)}
                for fut in as_completed(future_map):
                    jobs.raise_if_cancelled(_job_id)
                    new_entries[future_map[fut]] = fut.result()
            finally:
                jobs.unregister_resource(_job_id, pool)
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

    from stacking import stacker as _stacker

    def _filter_for(f: Path) -> str:
        entry = cached_by_path.get(str(f), {})
        name = entry.get("filter")
        if name:
            return str(name).strip()
        return _stacker.read_filter(f)

    # Pull each frame's cached measurements. Legacy `stats` cache entries don't
    # carry `star_count`; in those rows, `eccentricity is None` is the project's
    # marker for "no stars detected" (see _image_stats_run).
    def _measures(f: Path) -> tuple[Optional[int], Optional[float], Optional[float]]:
        entry = cached_by_path.get(str(f), {})
        fwhm = entry.get("fwhm_arcsec")
        ecc = entry.get("eccentricity")
        if "star_count" in entry:
            star_count = entry["star_count"]
        else:
            star_count = 0 if ecc is None else None
        return star_count, fwhm, ecc

    filt_by_frame: dict[Path, str] = {f: (_filter_for(f) or "Unknown") for f in fits_files}

    # Quality is judged relative to the per-filter median, so blue frames are
    # compared against blue and red against red (star counts and FWHM differ a
    # lot between filters). Medians are taken over frames that actually carry a
    # measurement (star_count > 0; FWHM/ecc not None).
    from statistics import median as _median

    star_samples: dict[str, list[int]] = {}
    fwhm_samples: dict[str, list[float]] = {}
    ecc_samples: dict[str, list[float]] = {}
    for f in fits_files:
        fn = filt_by_frame[f]
        sc, fwhm, ecc = _measures(f)
        if sc:                       # > 0
            star_samples.setdefault(fn, []).append(sc)
        if fwhm is not None:
            fwhm_samples.setdefault(fn, []).append(fwhm)
        if ecc is not None:
            ecc_samples.setdefault(fn, []).append(ecc)

    med_stars = {fn: _median(v) for fn, v in star_samples.items() if v}
    med_fwhm = {fn: _median(v) for fn, v in fwhm_samples.items() if v}
    med_ecc = {fn: _median(v) for fn, v in ecc_samples.items() if v}

    # Decide which frames are bad against their filter's median thresholds.
    bad: list[tuple[Path, str]] = []  # (path, reason)
    for f in fits_files:
        fn = filt_by_frame[f]
        star_count, fwhm, ecc = _measures(f)
        m_stars = med_stars.get(fn)
        m_fwhm = med_fwhm.get(fn)
        m_ecc = med_ecc.get(fn)

        if star_count == 0:
            bad.append((f, "no stars detected"))
        elif (star_count is not None and m_stars is not None
              and star_count < _BAD_STAR_FRACTION * m_stars):
            bad.append((f, f"{star_count} stars < {_BAD_STAR_FRACTION:.0%} of "
                           f"{fn} median {m_stars:.0f}"))
        elif (fwhm is not None and m_fwhm is not None
              and fwhm > _BAD_FWHM_FACTOR * m_fwhm):
            bad.append((f, f"FWHM {fwhm:.2f}\" > {_BAD_FWHM_FACTOR}x "
                           f"{fn} median {m_fwhm:.2f}\""))
        elif (ecc is not None and m_ecc is not None
              and ecc > _BAD_ECC_FACTOR * m_ecc):
            bad.append((f, f"ecc {ecc:.2f} > {_BAD_ECC_FACTOR}x "
                           f"{fn} median {m_ecc:.2f}"))

    # Per-filter totals before/after the would-be rename
    total_by_filter: dict[str, int] = {}
    bad_by_filter: dict[str, int] = {}
    bad_set = {p for p, _ in bad}
    for f in fits_files:
        fn = filt_by_frame[f]
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
            f"{dso_dir.name}: all {len(fits_files)} frames pass per-filter "
            f"thresholds (stars≥{_BAD_STAR_FRACTION:.0%} median, "
            f"FWHM≤{_BAD_FWHM_FACTOR}x median, ecc≤{_BAD_ECC_FACTOR}x median)\n"
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


def dab_cmd(words: list[str], account: str) -> None:
    """Restore frames previously flagged by `bad` back to active (reverse of `bad`).

    Renames `<frame>.fits.bad` back to `<frame>.fits` so the stacker picks them
    up again. `dab` is `bad` backwards, and so is what it does.

    Usage:
        dab                 — dry-run on last-imaged DSO
        dab <dso>           — dry-run on named DSO
        dab <dso> go        — actually restore (default is dry-run)
        dab go              — dry-run on last DSO (use `dab * go` to restore)
    """
    jobs.spawn(_dab_run, args=(words,))


def _dab_run(words: list[str]) -> None:
    _job_id = jobs.get_current_job()

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])

    # Parse args: optional <dso> and optional trailing "go"
    extra = list(words[2:]) if len(words) > 2 else []
    do_rename = False
    if extra and extra[-1].lower() == "go":
        do_rename = True
        extra = extra[:-1]
    dso_arg = " ".join(extra).strip() or None
    if dso_arg == "*":
        dso_arg = None

    def _is_light_bad(f: Path) -> bool:
        # <frame>.fits.bad living directly under a LIGHT directory
        return f.parent.name.upper() == "LIGHT"

    def _find_dso_dir(path: Path) -> Optional[Path]:
        d = path.parent
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
                return max(f.stat().st_mtime for f in d.rglob("*.fits.bad") if _is_light_bad(f))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    if dso_arg:
        dso_dir = _find_dso_dir_by_name(dso_arg)
        if dso_dir is None:
            social_server.post_social_message(f"No image directory found for '{dso_arg}'")
            return
    else:
        all_bad = sorted(
            (f for f in image_dir.rglob("*.fits.bad") if _is_light_bad(f)),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_bad:
            social_server.post_social_message("No .bad frames found")
            return
        dso_dir = _find_dso_dir(all_bad[-1])

    bad_files = sorted(
        f for f in dso_dir.rglob("*.fits.bad") if _is_light_bad(f)
    ) if dso_dir else []
    if not bad_files:
        social_server.post_social_message(f"{dso_dir.name}: no .bad frames to restore")
        return

    verb = "Restored" if do_rename else "Would restore"
    lines = [f"{verb} {len(bad_files)} flagged frame(s) in {dso_dir.name}:"]
    for f in bad_files[:40]:
        lines.append(f"  {f.name} → {f.with_suffix('').name}")
    if len(bad_files) > 40:
        lines.append(f"  …and {len(bad_files) - 40} more")
    if not do_rename:
        lines.append(f"Re-run as `dab {dso_arg or '*'} go` to actually restore.")
    social_server.post_social_message("\n".join(lines))

    if not do_rename:
        return

    failed: list[tuple[Path, str]] = []
    restored = 0
    for f in bad_files:
        jobs.raise_if_cancelled(_job_id)
        target = f.with_suffix("")  # strip the trailing .bad → <frame>.fits
        try:
            if target.exists():
                failed.append((f, "active frame already exists"))
                continue
            f.rename(target)
            restored += 1
        except Exception as exc:
            failed.append((f, str(exc)))

    social_server.post_social_message(
        f"{dso_dir.name}: restored {restored}/{len(bad_files)} frame(s)."
    )
    if failed:
        social_server.post_social_message(
            f"Restore skipped/failed for {len(failed)} frame(s); first: "
            f"{failed[0][0].name} — {failed[0][1]}"
        )


_AUDIO_USAGE = ("Usage: `audio` (list unlabeled + library counts) or "
                "`audio <open|close> <good|bad> [name]` (file the latest — or "
                "named — unlabeled capture of that direction)")


def audio_cmd(words: list[str], account: str) -> None:
    """List or label roof-move audio captures (and the matching current signature).

    Labeling moves the capture's spectrogram + WAV from roof_audio/unlabeled/ to
    the good/bad library that classify() judges future moves against, and files
    the motor-current signature from the same move (same direction, within ±10
    minutes) under the same verdict — one verdict per roof move.

    Command syntax:
        audio                       — list unlabeled captures + library counts
        audio <open|close> <good|bad> [name] — label latest (or named) capture
    """
    args = words[2:]

    if not args:
        entries = roof_audio.list_unlabeled()
        lines = [f"{len(entries)} unlabeled roof audio capture(s):"]
        lines += [f"  {e['base']}" for e in entries[:20]]
        if len(entries) > 20:
            lines.append(f"  …and {len(entries) - 20} more")
        counts = roof_audio.library_counts()
        lines.append("Library:")
        lines += [f"  {status}/{d}: {n}"
                  for status, dirs in counts.items() for d, n in dirs.items()] or ["  (empty)"]
        lines.append("Label with `audio <open|close> <good|bad>`.")
        social_server.post_social_message("\n".join(lines))
        return

    if len(args) < 2 or args[0] not in ("open", "close") or args[1] not in ("good", "bad"):
        social_server.post_social_message(_AUDIO_USAGE)
        return
    direction, verdict = args[0], args[1]
    name = args[2] if len(args) > 2 else None

    res = roof_audio.label(direction, verdict, name=name)
    if res is None:
        social_server.post_social_message(
            f"No unlabeled {direction} audio capture"
            + (f" matching '{name}'" if name else "") + " found.")
        return
    lines = [f"Filed {res['base']} as {verdict}:"]
    lines += [f"  {os.path.basename(p)}" for p in res["moved"]]

    # File the motor-current signature from the same move under the same verdict.
    try:
        sig_path = rcs.label_latest(direction, verdict, near_timestamp=res["base"])
    except Exception as e:  # noqa: BLE001 — audio labeling succeeded; report and move on
        sig_path = None
        _logger.error("Failed to label current signature: %s", e)
    if sig_path:
        lines.append(f"  {os.path.basename(sig_path)} (current signature)")
    else:
        lines.append("  (no matching current signature within ±10 min)")
    social_server.post_social_message("\n".join(lines))


def get_super_user_commands() -> dict[str, Callable]:
    """Return the command-name → handler mapping for all super-user commands."""
    return {
        "dbr": dbr_cmd,
        "dbd": dbd_cmd,
        "dbc": dbc_cmd,
        "dbb": dbb_cmd,
        "image!!": image_cmd,
        "roof!!": roof_cmd,
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
        "transient": transient_cmd,
        "diff": transient_cmd,
        "hr": hr_cmd,
        "log": log_cmd,
        "update": update_cmd,
        "optics": optics_cmd,
        "drift": drift_cmd,
        "stack": stack_cmd,
        "bad": bad_cmd,
        "dab": dab_cmd,
        "audio": audio_cmd,
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
    return first_line.strip() == "USER SAFE"

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
    if is_imaging() or is_nina_running():
        # is_nina_running() catches the case where a process restart cleared
        # imaging.txt but NINA is still capturing — starting again would run two
        # imaging sequences against the same mount/camera.
        pushover.push_message("Already imaging (or NINA still running), cannot restart")
    else:
        from fits_processing import frame_watcher
        cfg = config.data()
        # Clear any stale emergency-abort flag so it can't kill this fresh run.
        clear_abort()
        # Claim the run synchronously, before returning, so callers that poll
        # get_imaging_state() (e.g. the scheduler's imaging_task) never observe
        # NONE and conclude the run finished before the worker thread has even
        # set ACTIVE.
        set_imaging_state(ImagingState.ACTIVE)
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
        jobs.spawn(_run)



def doit_cmd(words: list[str], account: str) -> None:
    """
    Full observatory imaging run. Always launched via image_cmd, which runs it
    in a background thread (manual trigger or the scheduler in auto mode).
    Manages the entire night: safety checks → roof open → NINA prelude →
    NINA main imaging.

    image_cmd claims the run by setting imaging state to ACTIVE synchronously
    before this thread starts (guarding against concurrent callers), so this
    function does not re-check or re-claim the state.

    operand meanings:
        1 = full run (prelude + image_nina1)
        2 = full run (prelude + image_nina2)
        3 = full run (prelude + home_and_park, no imaging)

    State machine transitions written to imaging.txt during this function:
        ACTIVE → IN_PRELUDE   (just before on_nina.bat launches)
        IN_PRELUDE → DONE_PRELUDE   (written externally by NINA/bat when prelude ends)
        DONE_PRELUDE → IN_MAIN   (just before image_nina bat launches)
        IN_MAIN → NONE   (written by the NINA main sequence via an External Script step calling set_imaging_state.bat NONE)

    Safety checks ("safe!" / "unsafe!") are read from safety.txt and must be
    explicitly set by a super-user before and during the run. Any failed check
    aborts immediately without closing the roof (that is handled by end.py).
    """

    # The run has already been claimed (state set to ACTIVE) synchronously by
    # image_cmd before this thread started, which also guarantees no concurrent
    # caller slipped through. Nothing to re-check here.
    _logger.info("doit_cmd")

    # Kill any stale NINA or PWI4 processes before starting a fresh run.
    subprocess.run(
        [os.path.join(_SCRIPTS_DIR, "kill_nina_pwi4.bat")],
        shell=True
    )

    # Persist imaging start time so end.py can compute the post-imaging summary.
    try:
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(_root, "imaging_start.txt"), "w") as _f:
            _f.write(datetime.now().isoformat())
    except Exception:
        pass
    # Persist this run's job id so the end-of-night scripts NINA launches
    # (end.py, smessage.py — separate processes) post onto this job's card.
    jobs.persist_imaging_job()
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
            if is_aborting():
                # Emergency stop fired — the emergency handler owns the safe
                # shutdown from here, so just unwind without touching hardware.
                _logger.info("Prelude wait aborted by emergency stop")
                set_imaging_state(ImagingState.NONE)
                return
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
        while get_imaging_state() != ImagingState.NONE and not is_aborting():
            time.sleep(60)
        if is_aborting():
            # Emergency stop fired — do NOT fall through into flats (which would
            # power the mount back on and relaunch NINA). The emergency handler
            # owns the safe shutdown from here.
            _logger.info("Main imaging wait aborted by emergency stop — skipping flats")
            return
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

    # Flats must be dark — force the inside light off before capturing, regardless
    # of what the End Sequence left it at (a failed shutdown step can leave it on
    # and contaminate the flats).
    try:
        asyncio.run(ku.kasa_do(dev_map, {"Iris inside light": 'off'}))
        _logger.info("Inside light forced off before flats")
    except Exception:
        _logger.exception("Failed to force inside light off before flats")

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
    # Clear any stale emergency-abort flag so it can't kill this fresh run.
    clear_abort()
    def _run():
        set_imaging_state(ImagingState.ACTIVE)
        try:
            do_flats()
        finally:
            set_imaging_state(ImagingState.NONE)
    jobs.spawn(_run)


def snr_cmd(words: list[str], account: str) -> None:
    """Post stack-convergence curve in a separate process (non-blocking).

    Process-isolated (jobs.spawn_process) so a heavy snr stack runs on its own
    core without GIL contention with a concurrent hr/other command.
    """
    jobs.spawn_process(_snr_run, args=(words,))


def _snr_run(words: list[str]) -> None:
    """Run SNR convergence under a single-flight lock.

    Only one convergence run may execute at a time (see :data:`_snr_lock`). If a
    run is already in progress this call is skipped rather than piling a second
    memory-heavy run on top — that overlap can crash the in-process web server.
    """
    if not _snr_lock.acquire(blocking=False):
        social_server.post_social_message("SNR already running — skipping this request")
        _logger.info("SNR skipped: another convergence run is already in progress")
        return
    try:
        _snr_run_locked(words)
    finally:
        _snr_lock.release()


def _snr_run_locked(words: list[str]) -> None:
    """Worker for snr_cmd.

    Usage:
        snr           — convergence curve for the DSO currently being / last imaged
        snr <dso>     — convergence curve for the named DSO

    Loads all LIGHT frames, groups by filter, and posts one convergence plot per
    filter showing normalised RMSE vs Fibonacci-spaced frame counts.
    """
    from stacking import stacker

    _job_id = jobs.get_current_job()
    _cancel = jobs.cancel_cb_for(_job_id)

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
        jobs.raise_if_cancelled(_job_id)
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
                cancel_cb=_cancel,
            )
        except jobs.Cancelled:
            raise
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

        lines = [
            f"  {fn}: slope {info['tail_slope_pct']:+.4f}%/frame"
            for fn, info in sorted(saved.items())
        ]
        social_server.post_social_message(
            f"{dso_dir.name} — convergence slopes:\n" + "\n".join(lines)
        )


def _load_precomputed_fwhm_stars(
    dso_dir: Path, paths: list[Path], arcsec_per_pixel: float
) -> dict[Path, tuple[float, int]]:
    """Build a {path: (fwhm_px, star_count)} map from a DSO's frame_stats.json.

    Lets the stacker reuse the FWHM/star measurements already cached by the
    `stats`/`bad` commands instead of redoing the detection pass. The cache
    stores FWHM in arcseconds; the stacker works in pixels, so convert with
    *arcsec_per_pixel*. Only paths present in *paths* are returned; a missing,
    unreadable, or value-less cache yields an empty map (stacker measures
    normally). Matching is by normalised absolute path so cache keys written on
    a different run still line up.
    """
    import json as _json

    cache_path = dso_dir / "frame_stats.json"
    if not cache_path.exists() or not arcsec_per_pixel:
        return {}
    try:
        with open(cache_path) as fh:
            rows = _json.load(fh)
    except Exception:
        _logger.warning("hr: could not read %s", cache_path, exc_info=True)
        return {}
    by_norm = {
        os.path.normcase(os.path.abspath(r["path"])): r
        for r in rows
        if isinstance(r, dict) and r.get("path")
    }
    out: dict[Path, tuple[float, int]] = {}
    for p in paths:
        r = by_norm.get(os.path.normcase(os.path.abspath(str(p))))
        if not r:
            continue
        fa = r.get("fwhm_arcsec")
        fwhm_px = float(fa) / arcsec_per_pixel if fa else 0.0
        out[p] = (fwhm_px, int(r.get("star_count") or 0))
    return out


def hr_cmd(words: list[str], account: str) -> None:
    """Build a Gaia-calibrated colour–magnitude (H–R) diagram. Background job.

    Usage: hr <dso> [bluefilter redfilter]
        hr m13            — auto-pick the two most-imaged filters
        hr m13 B R        — use B for colour-blue, R for colour-red

    Process-isolated (jobs.spawn_process) so the heavy stack/photometry runs on
    its own core, in true parallel with a concurrent snr/other command.
    """
    jobs.spawn_process(_hr_run, args=(words,))


def _hr_run(words: list[str]) -> None:
    """Worker for hr_cmd: stack two filters, plate-solve, photometer, plot CMD."""
    from stacking import stacker
    from photometry import cmd_diagram

    _job_id = jobs.get_current_job()
    _cancel = jobs.cancel_cb_for(_job_id)

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = Path(os.path.join(_project_root, cfg["scratch"]["directory"]))
    astap_exe = cfg.get("hardware", {}).get("astap_exe", "")
    arcsec = float(cfg["nina"]["arc_sec_per_pixel"])

    args = words[2:]
    # `hr <dso> <blue> <red>` — three-or-more tokens means the last two are
    # filters; otherwise the whole tail is the DSO name and filters auto-pick.
    requested_filters: Optional[tuple[str, str]] = None
    if len(args) >= 3:
        requested_filters = (args[-2], args[-1])
        dso_arg = " ".join(args[:-2]).strip()
    else:
        dso_arg = " ".join(args).strip()
    if not dso_arg:
        social_server.post_social_message(
            "Usage: hr <dso> [bluefilter redfilter]  (e.g. `hr m13` or `hr m13 B R`)")
        return

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

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

    dso_dir = _find_dso_dir_by_name(dso_arg)
    if dso_dir is None:
        social_server.post_social_message(f"hr: no image directory found for '{dso_arg}'")
        return

    fits_files = [f for f in dso_dir.rglob("*.fits") if _is_light(f)]
    if not fits_files:
        social_server.post_social_message(f"hr: no LIGHT frames for {dso_dir.name}")
        return

    by_filter = stacker.group_by_filter(fits_files)

    # Resolve filters: explicit request (case-insensitive) or auto-pick.
    if requested_filters:
        lookup = {k.lower(): k for k in by_filter}
        resolved = [lookup.get(f.lower()) for f in requested_filters]
        if None in resolved:
            social_server.post_social_message(
                f"hr: filter not found. {dso_dir.name} has: "
                f"{', '.join(sorted(by_filter)) or '(none)'}")
            return
        blue, red = resolved
        if blue == red:
            social_server.post_social_message(
                "hr: need two different filters to form a colour")
            return
        # Normalise to wavelength order (blue→red). build_cmd pins the "blue"
        # filter onto Gaia BP and the "red" onto RP, so a reversed pair (e.g.
        # `hr m92 r b`) miscalibrates and flips the colour axis. The auto-pick
        # path already sorts; do the same here so explicit order can't matter.
        if cmd_diagram.filter_rank(blue) > cmd_diagram.filter_rank(red):
            blue, red = red, blue
            social_server.post_social_message(
                f"hr: swapped to wavelength order — blue {blue}, red {red}")
    else:
        pair = cmd_diagram.choose_filters(by_filter, min_frames=2)
        if pair is None:
            social_server.post_social_message(
                f"hr: need two filters for a colour. {dso_dir.name} has: "
                f"{', '.join(f'{k}×{len(v)}' for k, v in sorted(by_filter.items()))}")
            return
        blue, red = pair

    out = scratch_dir / f"cmd_{dso_dir.name}_{blue}_{red}.jpg"

    def _progress(msg: str) -> None:
        social_server.post_social_message(f"HR [{dso_dir.name} {blue}/{red}]: {msg}")

    _progress(f"building CMD from {len(by_filter[blue])} {blue} + "
              f"{len(by_filter[red])} {red} subs…")

    # Reuse FWHM/star measurements cached by `stats`/`bad` so the stacker skips
    # redoing the detection pass on frames it already knows.
    precomputed = _load_precomputed_fwhm_stars(
        dso_dir, by_filter[blue] + by_filter[red], arcsec)
    if precomputed:
        _progress(f"reusing {len(precomputed)} cached FWHM/star measurements")

    try:
        stats = cmd_diagram.build_cmd(
            by_filter[blue], by_filter[red], blue, red, dso_dir.name, out,
            astap_exe, arcsec, progress_cb=_progress, cancel_cb=_cancel,
            precomputed_fwhm_stars=precomputed,
            observatory_name=cfg["location"].get("observatory_name", "this telescope"),
        )
    except jobs.Cancelled:
        raise
    except RuntimeError as exc:
        social_server.post_social_message(f"HR [{dso_dir.name}]: {exc}")
        return
    except Exception as exc:
        _logger.exception("_hr_run failed")
        social_server.post_social_message(f"HR [{dso_dir.name}]: failed — {exc}")
        return

    age_str = (
        f"\nΔ{red}(TO−HB) ≈ {stats['delta_mag']:.2f} mag → age ≈ "
        f"{stats['age_gyr']:.0f} Gyr (rough)"
        if stats.get("age_gyr") else ""
    )
    social_server.post_social_message(
        f"Colour–magnitude diagram — {dso_dir.name}  ({blue}−{red})\n"
        f"{stats['n_stars']} stars · {stats['n_gaia']} Gaia anchors · "
        f"{stats.get('n_members', 0)} cluster members · "
        f"ZP {blue} {stats['zp_blue']:+.2f}, {red} {stats['zp_red']:+.2f}"
        f"{age_str}",
        str(out),
    )


def transit_cmd(words: list[str], account: str) -> None:
    """Search saved subs for transit-like dips. Runs in a separate process.

    Usage: transit <dso> [filter]   (filter optional; omit or use * for all subs)

    Process-isolated (jobs.spawn_process) so the heavy search runs on its own
    core, in true parallel with a concurrent command.
    """
    jobs.spawn_process(_transit_run, args=(words,))


def _transit_run(words: list[str]) -> None:
    """Worker for transit_cmd."""
    if len(words) < 3:
        social_server.post_social_message("Usage: transit <dso> [filter]")
        return

    # Filter is optional. A trailing "*" (or no filter at all) means "all subs".
    args = words[2:]
    if args and args[-1].strip() == "*":
        args = args[:-1]            # drop the explicit "all" marker
        filter_name = "*"
    elif len(args) >= 2:
        filter_name = args[-1].strip()
        args = args[:-1]
    else:
        filter_name = "*"           # no filter given → all subs
    dso_arg = " ".join(args).strip()
    if not dso_arg:
        social_server.post_social_message("Usage: transit <dso> [filter]")
        return

    filter_label = "all" if filter_name == "*" else filter_name

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = Path(os.path.join(_project_root, cfg["scratch"]["directory"]))
    out_path = scratch_dir / f"transit_{dso_arg.replace(' ', '_')}_{filter_label}.jpg"

    def _progress(msg: str) -> None:
        social_server.post_social_message(f"Transit [{dso_arg}/{filter_label}]: {msg}")

    # Capture the job id now (binding is active here) so the cancel check works
    # regardless of which thread run_transit_search ends up polling it from.
    _job_id = jobs.get_current_job()

    def _is_cancelled() -> bool:
        return jobs.is_cancelled(_job_id)

    social_server.post_social_message(
        f"Transit search for {dso_arg} [{filter_label}]: starting…"
    )

    from transit_search import transit as _ts
    try:
        entry = _ts.run_transit_search(
            dso_name=dso_arg,
            filter_name=filter_name,
            image_dir=image_dir,
            output_plot_path=out_path,
            progress_cb=_progress,
            cancel_cb=_is_cancelled,
        )
    except _ts.TransitCancelled:
        social_server.post_social_message(f"Transit [{dso_arg}/{filter_label}]: cancelled.")
        return
    except Exception as exc:
        _logger.exception("_transit_run: failed")
        social_server.post_social_message(f"Transit [{dso_arg}/{filter_label}]: failed — {exc}")
        return

    cands = entry.get("candidates", [])
    if cands:
        top = cands[0]
        loc = (f"RA {top['ra_deg']:.4f} Dec {top['dec_deg']:+.4f}"
               if top.get("ra_deg") is not None else f"({top['x']:.0f},{top['y']:.0f})")
        gmag = f" G={top['gaia_g_mag']:.1f}" if top.get("gaia_g_mag") is not None else ""
        depth_pct = top.get("transit_depth", 0.0) * 100
        dur_h = top.get("transit_duration_d", 0.0) * 24.0
        period = top.get("bls_period_d")
        period_str = f" P={period:.3f}d" if period else ""
        # Confidence from the significance tests (small FAP + large field-z +
        # large dip/bump ⇒ likely a real, one-sided transit, not noise/systematic).
        sig = top.get("significance") or {}
        conf = ""
        if sig:
            # A metric can be None for distinct reasons; say which, rather than a
            # bare "n/a". perm_fap is only None when the permutation test was
            # skipped for too few finite epochs (it needs ≥30); a falsy
            # n_permutations means the whole dict is an uncomputed placeholder.
            nperm = sig.get("n_permutations") or 0
            fap = sig.get("perm_fap")
            if fap is not None:
                fap_str = (f"<{1.0/(nperm+1):.3f}" if nperm and fap <= 1.0 / (nperm + 1)
                           else f"{fap:.3f}")
            elif not nperm:
                fap_str = "n/a (not computed)"
            else:
                fap_str = "n/a (<30 epochs)"
            # field_z None with field_fap present ⇒ MAD==0 (degenerate/flat
            # field); both None ⇒ too few comparison stars (<5).
            fz = sig.get("field_z")
            if fz is not None:
                fz_str = f"{fz:.1f}"
            else:
                fz_str = "n/a (flat field)" if sig.get("field_fap") is not None else "n/a (<5 stars)"
            db = sig.get("dip_bump")
            db_str = f"{db:.1f}×" if db is not None else "n/a"
            conf = f" | confidence: FAP {fap_str}, field-z {fz_str}, dip/bump {db_str}"
        summary = (
            f"Transit [{dso_arg}/{filter_label}]: {entry['n_stars']} stars, "
            f"{entry['frame_count']} frames over {entry['baseline_days']:.1f}d. "
            f"Top: {loc}{gmag} depth={depth_pct:.2f}% dur={dur_h:.2f}h "
            f"score={top.get('score', 0):.1f}{period_str}.{conf}"
        )
    else:
        summary = (
            f"Transit [{dso_arg}/{filter_label}]: complete, no candidates above threshold."
        )

    if out_path.exists():
        social_server.post_social_message(summary, str(out_path))
    else:
        social_server.post_social_message(summary)

    # Annotated reference frame showing where the candidate is.
    field_img = entry.get("field_image")
    if field_img and os.path.exists(field_img):
        social_server.post_social_message("Candidate location (circled):", field_img)


def _describe_transient(dso: str, filt: str, entry: dict, top: dict,
                        n_total: int) -> str:
    """Two-sentence plain-language readout of the single best candidate: where it
    is, how strong it is, and what its shape/brightness metrics imply."""
    loc = (f"RA {top['ra_deg']:.4f} Dec {top['dec_deg']:+.4f}"
           if top.get("ra_deg") is not None else f"pixel ({top['x']:.0f}, {top['y']:.0f})")
    off = top.get("offset_from_nucleus_px")
    off_str = f", {off:.0f}px from the nucleus" if off is not None else ""

    # Catalogue identity: a named SIMBAD object or a nearby Gaia star both usually
    # mean the source is *known* (variable star, galaxy nucleus) — not a new SN.
    if top.get("simbad_name"):
        otype = f", {top['simbad_otype']}" if top.get("simbad_otype") else ""
        ident = f" — SIMBAD match: {top['simbad_name']}{otype}"
    elif top.get("gaia_source_id"):
        g = f", G={top['gaia_g_mag']:.1f}" if top.get("gaia_g_mag") is not None else ""
        sep = top.get("gaia_sep_arcsec")
        sep_str = f" {sep:.1f}\" away" if sep is not None else ""
        ident = f" — Gaia DR3 {top['gaia_source_id']}{g}{sep_str} (likely that star, not new)"
    elif top.get("ra_deg") is not None:
        ident = " — no SIMBAD/Gaia catalogue match"
    else:
        ident = ""

    # Clickable SIMBAD cone-search on the resolved position for a human to inspect.
    link = ""
    if top.get("ra_deg") is not None:
        link = (f" SIMBAD: https://simbad.u-strasbg.fr/simbad/sim-coo?"
                f"Coord={top['ra_deg']:.5f}%20{top['dec_deg']:+.5f}"
                f"&Radius=10&Radius.unit=arcsec")

    a, elong = top.get("a"), top.get("elongation")
    point_like = a is not None and elong is not None and a <= 1.3 and elong <= 1.4
    if elong is not None and elong <= 1.2 and a is not None and a <= 1.1:
        shape = "compact and round"
    elif point_like:
        shape = "compact"
    else:
        shape = "extended / irregular"

    tflux, sflux = top.get("template_flux"), top.get("science_flux")
    if tflux is not None and sflux is not None and tflux > 0 and sflux > 0:
        bright = f", ~{sflux / tflux:.0f}× brighter than the template"
    elif sflux is not None and (tflux is None or tflux <= 0):
        bright = ", newly appeared (no measurable template flux)"
    else:
        bright = ""

    known = bool(top.get("simbad_name")) or bool(top.get("gaia_source_id"))
    if known:
        verdict = ("it coincides with a catalogued object, so it is most likely that "
                   "known source rather than a supernova")
    elif point_like:
        verdict = "its profile is consistent with a genuine new point source, worth a review"
    else:
        verdict = ("its profile is extended, so it is more likely a subtraction residual "
                   "than a real source")

    shape_bits = []
    if a is not None:
        shape_bits.append(f"semi-major {a:.1f}px")
    if elong is not None:
        shape_bits.append(f"elongation {elong:.2f}")
    shape_metrics = f" ({', '.join(shape_bits)})" if shape_bits else ""

    others = f" (best of {n_total} candidates)" if n_total and n_total > 1 else ""
    return (
        f"Transient [{dso}/{filt}]: {entry['n_template_frames']} template + "
        f"{entry['n_science_frames']} science frames over {entry['baseline_days']:.1f}d. "
        f"Best candidate: {loc}, SNR {top['snr']:.1f}{off_str}{others}{ident}. "
        f"It looks {shape}{shape_metrics}{bright} — {verdict}.{link}"
    )


def transient_cmd(words: list[str], account: str) -> None:
    """Difference the newest night against prior nights to find new sources.

    Usage: transient <dso> <filter>   (alias: diff)

    Process-isolated (jobs.spawn_process) so the heavy differencing runs on its
    own core, in true parallel with a concurrent command.
    """
    jobs.spawn_process(_transient_run, args=(words,))


def _transient_run(words: list[str]) -> None:
    """Worker for transient_cmd / diff_cmd."""
    if len(words) < 4:
        social_server.post_social_message("Usage: transient <dso> <filter>")
        return

    # Differencing is per-filter, so the filter is required (unlike transit).
    filter_name = words[-1].strip()
    dso_arg = " ".join(words[2:-1]).strip()
    if not dso_arg or not filter_name:
        social_server.post_social_message("Usage: transient <dso> <filter>")
        return

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = Path(os.path.join(_project_root, cfg["scratch"]["directory"]))
    out_path = scratch_dir / f"transient_{dso_arg.replace(' ', '_')}_{filter_name}.jpg"

    def _progress(msg: str) -> None:
        social_server.post_social_message(f"Transient [{dso_arg}/{filter_name}]: {msg}")

    # Capture the job id now (binding is active here) so the cancel check works
    # regardless of which thread run_transient_search polls it from.
    _job_id = jobs.get_current_job()

    def _is_cancelled() -> bool:
        return jobs.is_cancelled(_job_id)

    social_server.post_social_message(
        f"Transient search for {dso_arg} [{filter_name}]: starting…"
    )

    from transient_search import difference as _tr
    try:
        entry = _tr.run_transient_search(
            dso_name=dso_arg,
            filter_name=filter_name,
            image_dir=image_dir,
            output_plot_path=out_path,
            progress_cb=_progress,
            cancel_cb=_is_cancelled,
            top_n=1,  # present only the single best candidate
        )
    except _tr.TransientCancelled:
        social_server.post_social_message(f"Transient [{dso_arg}/{filter_name}]: cancelled.")
        return
    except Exception as exc:
        _logger.exception("_transient_run: failed")
        social_server.post_social_message(f"Transient [{dso_arg}/{filter_name}]: failed — {exc}")
        return

    cands = entry.get("candidates", [])
    survivors = [c for c in cands if not c.get("gaia_rejected")]
    n_total = entry.get("n_candidates", len(cands))
    if survivors:
        summary = _describe_transient(dso_arg, filter_name, entry, survivors[0], n_total)
    else:
        rej = len(cands) - len(survivors)
        extra = f" ({rej} matched known Gaia stars)" if rej else ""
        summary = (
            f"Transient [{dso_arg}/{filter_name}]: complete, no new sources above "
            f"threshold.{extra}"
        )

    if out_path.exists():
        social_server.post_social_message(summary, str(out_path))
    else:
        social_server.post_social_message(summary)


def image_stats_cmd(words: list[str], account: str) -> None:
    """Post per-frame FWHM/eccentricity graph in a background thread (non-blocking)."""
    jobs.spawn(_image_stats_run, args=(words, account))


def _image_stats_run(words: list[str], account: str) -> None:
    """Worker for image_stats_cmd.

    Usage:
        stats            — latest session for the DSO currently being imaged
        stats <dso>      — latest session for the named DSO
        stats <dso> all  — full multi-night history for the named DSO

    By default only the most recent observing session is plotted (see
    imaging_artifacts.gather_dso_frames) so a single big night can't dominate
    the frame-count x-axis. A trailing "all" opts into the full history.

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

    _job_id = jobs.get_current_job()

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]

    # A trailing "all" token requests the full multi-night history; the default
    # is the most recent observing session only (so one big night can't dominate).
    extra = list(words[2:])
    latest_session_only = True
    if extra and extra[-1].lower() == "all":
        latest_session_only = False
        extra = extra[:-1]
    dso_arg = " ".join(extra).strip() or None

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
            # Need a minimum number of stars before the FWHM/ecc fit is
            # trustworthy; below the floor, fall back to the HFR header value.
            if star_count < 5:
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
            jobs.register_resource(_job_id, pool)
            try:
                future_map = {pool.submit(_analyse_fits, f): i for i, f in enumerate(need_analysis)}
                for fut in as_completed(future_map):
                    jobs.raise_if_cancelled(_job_id)
                    new_entries[future_map[fut]] = fut.result()
            finally:
                jobs.unregister_resource(_job_id, pool)

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

    # Build the frame list via the shared helper (reconciled with disk + sorted,
    # latest session by default) so this tile and the eager imaging-card render
    # never diverge. `stats <dso> all` opts into the full multi-night history.
    from fits_processing import imaging_artifacts
    frames = imaging_artifacts.gather_dso_frames(
        dso_dir, latest_session_only=latest_session_only
    )

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

    # Min / median / max summary, posted after the graph. FWHM/ecc skip frames
    # with no stars (those carry None); star count includes every frame.
    import statistics as _statistics

    def _mmm(values: list) -> Optional[tuple]:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return min(vals), _statistics.median(vals), max(vals)

    # Build rows as (label, [min, median, max] formatted strings), then align
    # each numeric column to a common width so the slashes line up.
    rows: list[tuple[str, list[str]]] = []
    stars = _mmm([fr.get("star_count") for fr in frames])
    if stars:
        rows.append(("Stars", [f"{v:.0f}" for v in stars]))
    fwhm = _mmm([fr.get("fwhm_arcsec") for fr in frames])
    if fwhm:
        rows.append(("FWHM", [f"{v:.2f}\"" for v in fwhm]))
    ecc = _mmm([fr.get("eccentricity") for fr in frames])
    if ecc:
        rows.append(("Ecc", [f"{v:.3f}" for v in ecc]))

    if rows:
        label_w = max(len(label) for label, _ in rows)
        col_w = [max(len(row[i]) for _, row in rows) for i in range(3)]
        summary_lines = ["━━ Frame stats (min / median / max) ━━"]
        for label, values in rows:
            cols = " / ".join(v.rjust(col_w[i]) for i, v in enumerate(values))
            summary_lines.append(f"{label.ljust(label_w)} : {cols}")
        social_server.post_social_message("\n".join(summary_lines))


if __name__ == "__main__":
    announce_roof_movement("The roof will be opening in 5 Minutes")