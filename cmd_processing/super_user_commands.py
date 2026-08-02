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
    # known-good library. A move that classify()'s judges "good" is auto-filed
    # into the good library (the library is now mature enough to self-extend);
    # "bad"/"unknown" stay in unlabeled/ for a human to review with
    # `audio <open|close> <good|bad>`. classify() never raises (returns "unknown").
    audio_result = roof_audio.finish_background_capture(audio_capture, status="unlabeled")
    if audio_result and audio_result.get("spectrogram"):
        cls = roof_audio.classify(audio_result["spectrogram"],
                                  audio_result.get("direction"))
        caption = f"Roof {capture_direction or 'move'} audio"
        if cls["verdict"] == "good":
            caption += (f": sounds normal (score {cls['best_score']:.3f} ≥ "
                        f"{cls['threshold']:.3f}, best match {cls['best_match']})")
            # Auto-file to the good library. This MOVES the PNG + WAV out of
            # unlabeled/, so repoint audio_result at the new paths before the
            # chat post below reads them.
            promo = roof_audio.promote_to_good(audio_result)
            if promo and promo.get("moved"):
                caption += " — filed to known-good library"
                for m in promo["moved"]:
                    if m.lower().endswith(".png"):
                        audio_result["spectrogram"] = m
                    elif m.lower().endswith(".wav"):
                        audio_result["wav"] = m
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
            else:
                # Within the good envelope → auto-file this move's current
                # signature into the good library so it keeps maturing
                # alongside the audio. compare() is O(n) so this library isn't
                # cost-capped. Best-effort; a filing failure must not break the
                # roof sequence.
                try:
                    near = sig.get("timestamp", "").replace(":", "-") or None
                    rcs.label_latest(capture_direction, "good", near_timestamp=near)
                except Exception as e:  # noqa: BLE001
                    _logger.error("Failed to auto-file good current signature: %s", e)



def announce_roof_movement(text: str, speaker_name: str = "Observatory", volume: int = 40) -> None:
    """Announce upcoming roof movement via Sonos."""
    try:
        sonos_utils.sonos_say(text, speaker_name, volume)
    except Exception as e:
        _logger.error("Sonos announcement failed: %s", e)


def get_status_with_lights() -> tuple[bool, bool, bool, Any]:
    """Take a camera snapshot and return (parked, closed, open, mod_date) via vision safety.

    visual_status() retries internally on a garbage (torn/starved/unreadable)
    webcam frame, so a single corrupt snapshot doesn't block a roof move on its
    own — it reads as untrusted ("not parked") and gets a couple more chances to
    resolve before we act on it.
    """
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
        # Name which gate a template failed (position vs confidence) — but only
        # for templates whose failure explains a negative/ambiguous verdict.
        # A closed roof makes the "open" template fail by design (and vice
        # versa); listing that reads like a problem when the read is clean.
        suspects = []
        if not parked:
            suspects.append("parked")
        if not closed and not is_open:
            suspects.extend(["closed", "open"])
        fails = " ".join(
            f"{name}:{lm.get(name, {}).get('verdict')}"
            for name in suspects
            if lm.get(name, {}).get("verdict") not in (None, "ok")
        )
        caption = (
            f"Vision: roof={roof} scope={'parked' if parked else 'unparked'} | "
            f"conf c={lm.get('closed', {}).get('conf', 0):.2f} "
            f"o={lm.get('open', {}).get('conf', 0):.2f} "
            f"p={lm.get('parked', {}).get('conf', 0):.2f} "
            f"luma={lm.get('frame_luma', 0):.0f} trusted={lm.get('trusted')}"
            + (f" | {fails}" if fails else "")
        )
        pushover.push_message(caption, img)
    except Exception:
        _logger.exception("failed to push vision decision image")


def _vision_fail_reason(state: str) -> str:
    """Why the last vision snapshot didn't confirm *state* ('parked'/'closed'/'open'):
    which gate failed — template found in the wrong place ('position') or match
    confidence under the threshold ('confidence') — with the measured values.
    Empty string when the state passed or no verdict is available."""
    info = (vision_safety.last_match or {}).get(state) or {}
    verdict = info.get("verdict")
    if not verdict or verdict == "ok":
        return ""
    return (
        f" — {state} template: {verdict}"
        f" (conf {info.get('conf', 0):.2f}, off by {info.get('error', 0):.0f}px)"
    )


def _mount_power_blocked_reason(dev_map: dict | None = None) -> str | None:
    """Return why the telescope mount's power state forbids touching the roof,
    or None when the mount is confirmed off.

    A powered mount may be tracking, putting the scope in the roof's travel
    path. Probes the Kasa ``{"Telescope mount": "isoff"}`` state, running its
    own discovery when *dev_map* is None; if the state can't be confirmed we
    fail safe (refuse).
    """
    try:
        if dev_map is None:
            dev_map = asyncio.run(ku.make_discovery_map())
        mount_off = asyncio.run(ku.kasa_check(dev_map, {"Telescope mount": "isoff"}))
    except Exception as exc:
        _logger.warning("roof gate: mount power check failed: %s", exc)
        return f"could not confirm the telescope mount is powered off ({exc})"
    if not mount_off:
        return "the telescope mount is powered on"
    return None


def _roof_move_blocked_reason(imaging_run: bool = False, dev_map: dict | None = None) -> str | None:
    """Return why the roof must not MOVE right now, or None if movement may proceed.

    Combined gate for every movement path (``roof!! open/close/toggle`` and the
    imaging run), checked in order:

    1. no imaging run in progress — waived when *imaging_run* is True, i.e. the
       caller IS the imaging run and already claimed the imaging state at entry;
    2. the observatory is marked safe (``safe!``);
    3. the telescope mount is powered off (:func:`_mount_power_blocked_reason`).
    """
    if not imaging_run and (is_imaging() or is_nina_running()):
        return "an imaging run is in progress (imaging state is not none)"
    if not is_safe():
        return "observatory is not marked safe — issue safe! first"
    return _mount_power_blocked_reason(dev_map)


def _roof_status_blocked_reason() -> str | None:
    """Return why ``roof!! status`` must be ignored, or None if it may run.

    Status is read-only, so unlike movement it does NOT require ``safe!`` —
    you want to see the roof/scope state *before* deciding to arm the
    observatory. It is still skipped while imaging is in progress and while
    the mount is powered on (fail-safe if power can't be confirmed).
    """
    if is_imaging() or is_nina_running():
        return "an imaging run is in progress (imaging state is not none)"
    return _mount_power_blocked_reason()


# Posted when a cancel lands after the relay has fired: the roof cannot be
# stopped mid-travel, so cancelling then only abandons the confirmation wait.
_ROOF_CANCEL_AFTER_FIRE_MSG = (
    "Cancel received after the roof relay fired — the move itself already "
    "completed; use roof!! status to verify the roof position"
)


def _roof_cancel_point(msg: str, imaging_run: bool = False) -> None:
    """Honour a pending job cancel at a point where the roof is NOT moving.

    Only for roof!!-initiated moves: the imaging run is excluded because it has
    its own abort machinery (safe!/unsafe!, is_aborting) and must not unwind via
    Cancelled while the imaging state is claimed. Posts *msg* so the user knows
    whether the roof moved, then raises jobs.Cancelled (the job worker turns
    that into the terminal CANCELLED state).
    """
    if imaging_run:
        return
    if jobs.is_cancelled(jobs.get_current_job()):
        social_server.post_social_message(msg)
        raise jobs.Cancelled()


def _roof_confirm_wait(seconds: float, imaging_run: bool) -> None:
    """Post-move confirm-loop sleep that wakes ~1×/s to honour a roof!! cancel.

    Never used while the roof is travelling — toggle_roof owns that window
    (stall watchdog + motor power-off must always run to completion).
    """
    if imaging_run:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + seconds
    while True:
        _roof_cancel_point(_ROOF_CANCEL_AFTER_FIRE_MSG)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def open_roof(force: bool = False, imaging_run: bool = False) -> bool:
    """Open the observatory roof with full gating — the single open path for
    both ``roof!! open`` and the imaging run.

    ABSOLUTE RULE: the roof must never move unless the scope is confirmed
    parked — a tracking/slewing scope can intersect the roof's travel path.

    Gate order (any failure posts a refusal to the web chat and returns False):
    1. Kasa device discovery (needed to drive the roof motor).
    2. :func:`_roof_move_blocked_reason` — no imaging run (waived only by
       *imaging_run*, passed by the imaging run itself which claimed the
       imaging state at entry), observatory marked safe via ``safe!``, and
       mount powered off.
    3. ``_roof_lock``, non-blocking — refuses if another roof command is
       running; held through the confirm loop so movements can never overlap.
    4. Vision precondition: scope parked AND roof closed. *force* waives ONLY
       this step, never gates 1–3.

    Then announces via Sonos (always — forced moves included; anyone inside
    should hear it), fires the relay, and — unless forced — confirms via
    vision. Returns True when the roof is confirmed open with the scope still
    parked, or when a forced relay fire went out (gates passed, movement
    unverified); False on any refusal or failed confirmation.
    """
    try:
        dev_map = asyncio.run(ku.make_discovery_map())
    except Exception as exc:
        _logger.warning("open_roof: Kasa discovery failed: %s", exc)
        social_server.post_social_message(f"Roof will not open: Kasa device discovery failed ({exc})")
        return False
    blocked = _roof_move_blocked_reason(imaging_run=imaging_run, dev_map=dev_map)
    if blocked is not None:
        _logger.warning("open_roof refused: %s", blocked)
        social_server.post_social_message(f"Roof will not open: {blocked}")
        return False
    if not _roof_lock.acquire(blocking=False):
        social_server.post_social_message("Roof will not open: another roof command is already running")
        return False
    try:
        _roof_cancel_point("Roof open cancelled — the roof was not moved", imaging_run)
        if not force:
            parked, closed, is_open, mod_date = get_status_with_lights()
            if not parked:
                social_server.post_social_message(
                    f"Vision Safety says Scope is NOT parked, therefore will not open{_vision_fail_reason('parked')}"
                )
                return False
            if not closed:
                social_server.post_social_message(
                    f"Vision Safety says roof is NOT closed, therefore will not open{_vision_fail_reason('closed')}"
                )
                return False
            social_server.post_social_message("Vision Safety says roof is closed, opening roof")
        # Last safe abort point: past here the relay fires and the move cannot
        # be stopped (toggle_roof owns the travel window uninterrupted).
        _roof_cancel_point("Roof open cancelled — the roof was not moved", imaging_run)
        announce_roof_movement("The roof will be opening in one minute")
        toggle_roof(dev_map, capture_direction="open")
        if force:
            social_server.post_social_message("Roof open relay fired (forced, unverified)")
            return True
        _roof_confirm_wait(30, imaging_run)
        MAX_ROOF_CHECKS = 5
        for attempt in range(MAX_ROOF_CHECKS):
            _roof_cancel_point(_ROOF_CANCEL_AFTER_FIRE_MSG, imaging_run)
            parked, closed, is_open, mod_date = get_status_with_lights()
            if is_open and parked:
                return True
            if attempt < MAX_ROOF_CHECKS - 1:
                msg = (
                    f"Roof open not confirmed (attempt {attempt + 1}/{MAX_ROOF_CHECKS})"
                    f"{_vision_fail_reason('open')}, waiting 5 min"
                )
                social_server.post_social_message(msg)
                _logger.warning(msg)
                _roof_confirm_wait(5 * 60, imaging_run)
        social_server.post_social_message(f"Roof could not be confirmed open after {MAX_ROOF_CHECKS} attempts, stopping")
        _logger.warning("Roof open check failed after %d attempts", MAX_ROOF_CHECKS)
        return False
    finally:
        _roof_lock.release()


def close_roof(force: bool = False, imaging_run: bool = False) -> bool:
    """Close the observatory roof with full gating — the single close path for
    ``roof!! close`` (the end-of-night close in ``end.py`` is separate).

    ABSOLUTE RULE: the roof must never move unless the scope is confirmed
    parked — a tracking/slewing scope can intersect the roof's travel path.

    Same gate order as :func:`open_roof` (discovery → imaging/safe!/mount gate
    → ``_roof_lock`` → vision), with the same *force* / *imaging_run* semantics:
    *force* waives only the vision check, *imaging_run* waives only the
    imaging-in-progress gate. Vision precondition here is scope parked; an
    already-closed roof posts a note and returns True without moving anything.

    Announces via Sonos (always — forced moves included), fires the relay, and
    — unless forced — confirms via vision. Returns True when the roof is
    confirmed closed (or was already closed), or when a forced relay fire went
    out (gates passed, movement unverified); False on any refusal or failed
    confirmation. Refusals are posted to the web chat.
    """
    try:
        dev_map = asyncio.run(ku.make_discovery_map())
    except Exception as exc:
        _logger.warning("close_roof: Kasa discovery failed: %s", exc)
        social_server.post_social_message(f"Roof will not close: Kasa device discovery failed ({exc})")
        return False
    blocked = _roof_move_blocked_reason(imaging_run=imaging_run, dev_map=dev_map)
    if blocked is not None:
        _logger.warning("close_roof refused: %s", blocked)
        social_server.post_social_message(f"Roof will not close: {blocked}")
        return False
    if not _roof_lock.acquire(blocking=False):
        social_server.post_social_message("Roof will not close: another roof command is already running")
        return False
    try:
        _roof_cancel_point("Roof close cancelled — the roof was not moved", imaging_run)
        if not force:
            parked, closed, is_open, mod_date = get_status_with_lights()
            if not parked:
                social_server.post_social_message(
                    f"Vision Safety says Scope is NOT parked, therefore will not close{_vision_fail_reason('parked')}"
                )
                return False
            if closed:
                social_server.post_social_message("Vision Safety says roof is already closed")
                return True
            social_server.post_social_message("Vision Safety says scope is parked, closing roof")
        # Last safe abort point: past here the relay fires and the move cannot
        # be stopped (toggle_roof owns the travel window uninterrupted).
        _roof_cancel_point("Roof close cancelled — the roof was not moved", imaging_run)
        announce_roof_movement("The roof will be closing in one minute")
        toggle_roof(dev_map, capture_direction="close")
        if force:
            social_server.post_social_message("Roof close relay fired (forced, unverified)")
            return True
        _roof_confirm_wait(30, imaging_run)
        MAX_ROOF_CHECKS = 5
        for attempt in range(MAX_ROOF_CHECKS):
            _roof_cancel_point(_ROOF_CANCEL_AFTER_FIRE_MSG, imaging_run)
            parked, closed, is_open, mod_date = get_status_with_lights()
            if closed:
                return True
            if attempt < MAX_ROOF_CHECKS - 1:
                msg = (
                    f"Roof close not confirmed (attempt {attempt + 1}/{MAX_ROOF_CHECKS})"
                    f"{_vision_fail_reason('closed')}, waiting 5 min"
                )
                social_server.post_social_message(msg)
                _logger.warning(msg)
                _roof_confirm_wait(5 * 60, imaging_run)
        social_server.post_social_message(f"Roof could not be confirmed closed after {MAX_ROOF_CHECKS} attempts, stopping")
        _logger.warning("Roof close check failed after %d attempts", MAX_ROOF_CHECKS)
        return False
    finally:
        _roof_lock.release()


def roof_cmd(words: list[str], account: str) -> None:
    """Move or report the observatory roof. Command: ``roof!! open|close|toggle|status [force]``

    The ``!!`` flags that this command can move hardware (the roof, and indirectly
    the scope's collision envelope), mirroring ``image!!``.

    Subcommands:
        ``roof!! status``  — report scope/roof position via vision safety (no
                             movement; requires no imaging + mount off, but NOT
                             ``safe!`` — it's read-only).
        ``roof!! open``    — fully gated open via :func:`open_roof`.
        ``roof!! close``   — fully gated close via :func:`close_roof`.
        ``roof!! toggle``  — single relay toggle (the hardware just toggles;
                             direction depends on current position). Same
                             movement gate; parked-checked; when the current
                             position is known the travel direction is announced.
        append ``force`` to ``open``/``close``/``toggle`` to skip the vision
        (scope-parked) check (DANGEROUS — collision risk; use only when you can
        physically see the scope is parked). ``force`` never skips the
        imaging/safe!/mount-power gates.

    SAFETY: every movement variant requires the observatory to be marked safe
    (``safe!``), no imaging run in progress, and the mount powered off, then
    verifies the scope is parked via vision (unless ``force``). Movement runs
    on a background thread so the chat stays responsive; for open/close the
    gating lives in open_roof/close_roof, so refusal messages arrive from that
    thread (after device discovery) rather than synchronously.
    """
    sub = words[2] if len(words) >= 3 else ""
    force = len(words) >= 4 and words[3] == "force"

    if sub not in ("status", "open", "close", "toggle"):
        social_server.post_social_message("Usage: roof!! open|close|toggle|status [force]")
        return

    if sub == "status":
        blocked = _roof_status_blocked_reason()
        if blocked is not None:
            social_server.post_social_message(f"Ignoring roof!! status: {blocked}")
            return
        # Read-only snapshot. The process-wide camera lock in
        # inside_camera_server serializes it against any in-flight job.
        parked, closed, is_open, mod_date = get_status_with_lights()
        roof_state = "closed" if closed else ("open" if is_open else "ambiguous")
        social_server.post_social_message(
            f"Roof: {roof_state}; scope: {'parked' if parked else 'NOT parked'} "
            f"(vision @ {mod_date})"
        )
        return

    if force:
        social_server.post_social_message(
            f"⚠️ roof!! {sub} FORCE — skipping the scope-parked safety check"
        )

    if sub == "toggle":
        # toggle keeps its gate + lock here in the dispatch thread —
        # _toggle_roof_cmd itself does neither. Never let a roof movement
        # overlap an imaging run or another roof command: the original
        # incident was a `close` issued while an `open` was still in its
        # confirm loop; both jobs raced the single USB camera and the close
        # crashed.
        blocked = _roof_move_blocked_reason()
        if blocked is not None:
            social_server.post_social_message(f"Ignoring roof!! toggle: {blocked}")
            return
        if not _roof_lock.acquire(blocking=False):
            social_server.post_social_message("Cannot move the roof: another roof command is already running")
            return

        def _run_toggle() -> None:
            try:
                _toggle_roof_cmd(force)
            finally:
                _roof_lock.release()

        try:
            jobs.spawn(_run_toggle)
        except Exception:
            # spawn failed before the worker took ownership of the lock — release
            # it here so a roof command can never be permanently wedged.
            _roof_lock.release()
            raise
        return

    # open/close: gating, locking, and refusal messages all live inside
    # open_roof/close_roof — the worker just reports the outcome.
    def _run() -> None:
        if sub == "open":
            ok = open_roof(force=force)
            if not ok:
                social_server.post_social_message("❌ Roof failed to open")
            elif not force:
                # forced success already posted "relay fired (forced, unverified)"
                social_server.post_social_message("✅ Roof successfully opened")
        else:  # close
            ok = close_roof(force=force)
            if not ok:
                social_server.post_social_message("❌ Roof failed to close")
            elif not force:
                social_server.post_social_message("✅ Roof successfully closed")

    jobs.spawn(_run)


def _toggle_roof_cmd(force: bool) -> None:
    """Body of ``roof toggle``: a single parked-checked relay toggle.

    The caller (roof_cmd) enforces the movement gate (imaging/safe!/mount
    power) and holds ``_roof_lock`` for the duration — this function does
    neither itself. The roof relay only toggles, so the travel direction
    depends on the current position. When that position is known (vision), the
    direction is inferred, announced, and used to label the banked motor
    current signature.
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
    # Last safe abort point: past here the relay fires and the move cannot be
    # stopped (toggle_roof owns the travel window uninterrupted).
    _roof_cancel_point("Roof toggle cancelled — the roof was not moved")
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
            # Mount power must stay ON until do_main's own parked check — its
            # get_is_parked needs mount telemetry; do_main then cuts mount power
            # before it moves the roof.
            social_server.post_social_message("Scope parked — closing roof and shutting down")
            end.do_main()
            # Backstop: if do_main failed before its power-off step (its blanket
            # except only logs), the mount would stay powered all night. Parked
            # is confirmed here, so force the Kasa switch off.
            try:
                if dev_map is None:
                    dev_map = asyncio.run(ku.make_discovery_map())
                asyncio.run(ku.kasa_do(dev_map, {"Telescope mount": 'off'}))
                _logger.info("emergency: mount power confirmed off")
            except Exception:
                _logger.exception("emergency: mount power-off backstop failed")
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

# Serializes roof movement (open/close/toggle). Always acquired non-blocking so
# a second roof command is refused rather than queued. For open/close it is
# taken inside open_roof/close_roof (before the vision check, so state can't
# change between check and toggle — this also serializes roof!! against the
# imaging run's open). For toggle, roof_cmd acquires it in the dispatch thread
# and the worker releases it; a threading.Lock has no owner thread, so
# cross-thread release is legal.
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


def live_cmd(words: list[str], account: str) -> None:
    """Post a live, no-light view of the sky from the scope-top webcam.

    Takes TWO dark-sky passes and posts both, because one exposure can't serve
    both goals: a low-gain, long-exposure pass records STARS (at max gain the
    longest sub blows out and the scorer falls to a ~15 ms starless frame), and a
    high-gain pass favours diffuse SKYGLOW / clouds. Both use the same USB camera
    as the park/roof vision-safety check but leave the inside light OFF. Safe to
    run while imaging: read-only, serialized against the safety snapshot by the
    camera lock, never moves hardware or touches the lights. Runs in a background
    job (two sweeps ~ 30-40 s).

    An optional frame count averages that many frames at the chosen exposure to
    pull faint sky detail out of the noise; omit it for a single frame.
    examples:  live   |   live 8
    """
    # Optional trailing integer = frames to average-stack (default 1 = one frame).
    stack_frames = 1
    if len(words) > 2:
        try:
            stack_frames = int(words[2])
        except ValueError:
            social_server.post_social_message(
                "Usage: live [frames] — frames must be a positive integer (e.g. live 8)")
            return
        if stack_frames < 1:
            social_server.post_social_message("Usage: live [frames] — frames must be >= 1")
            return
        stack_frames = min(stack_frames, 25)  # cap so a typo can't tie up the camera

    def _run() -> None:
        from sentry import inside_camera_server  # local import: avoids import cycle
        cfg = config.data()
        cams = cfg["camera safety"]
        note = f", stacking {stack_frames} frames each" if stack_frames > 1 else ""
        stacked = f" — {stack_frames} frames stacked" if stack_frames > 1 else ""
        # (tag, output path, no-light gain, caption) — low gain for stars, high for skyglow.
        passes = [
            ("stars", cams["sky_view_stars"], cams.get("sky_stars_gain", 30),
             "Live sky — stars (low gain, long exposure)"),
            ("skyglow", cams["sky_view"], cams.get("sky_skyglow_gain", 100),
             "Live sky — skyglow / clouds (high gain)"),
        ]
        social_server.post_social_message(
            f"Capturing sky view — two passes (stars + skyglow), lights stay off{note}…")
        posted = 0
        for tag, out_path, gain, caption in passes:
            try:
                ok = inside_camera_server.take_snapshot(
                    light=False,
                    out_path=out_path,
                    scorer=inside_camera_server.dark_sky_score,
                    stack_frames=stack_frames,
                    gain=gain,
                )
            except Exception as e:  # noqa: BLE001
                _logger.exception("live %s capture failed", tag)
                social_server.post_social_message(f"Sky view [{tag}] capture failed: {e}")
                continue
            if not ok or not os.path.exists(out_path):
                social_server.post_social_message(
                    f"Sky view [{tag}] capture failed (no frame from the camera).")
                continue
            social_server.post_social_message(caption + stacked, image=out_path)
            posted += 1
        if posted == 0:
            social_server.post_social_message("Sky view capture failed for both passes.")

    jobs.spawn(_run)


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
    # flat_root (the key shipped in the config template): a single root whose
    # per-filter subdirs (Ha/, OIII/, …) hold the flats. Explicit
    # flat_dir/flat_dirs keys win if both are configured.
    if cal.get("flat_root") and not (_flat_dir or _flat_dirs):
        _flat_dir, _flat_dirs = stacker.flat_dirs_from_root(Path(cal["flat_root"]))
    # N.I.N.A writes one FLAT dir per session with every filter mixed in, which
    # the root/<FILTER>/ convention above cannot see. Group by FITS header too.
    _flats_by_header = stacker.flats_by_filter(
        Path(cal["flat_root"]) if cal.get("flat_root") else None)

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
            flat_paths = stacker._resolve_flat_paths(
                filter_name, _flat_dir, _flat_dirs, _flats_by_header)
            # Reuse the FWHM/star measurements stats/bad/snr already wrote to
            # frame_stats.json instead of redoing detection on every frame — that
            # was several minutes of duplicated work per stack.
            precomputed = _load_precomputed_fwhm_stars(dso_dir, paths, arcsec_per_pixel)
            if flat_paths:
                _progress(f"using {len(flat_paths)} flats")
            result, info = stacker.stack(
                paths,
                method=stacker.StackMethod.SIGMA_CLIP_FWHM,
                bias_paths=_bias_paths,
                dark_paths=_dark_paths,
                flat_paths=flat_paths,
                progress_cb=_progress,
                cancel_cb=_cancel,
                precomputed_fwhm_stars=precomputed,
            )
        except jobs.Cancelled:
            raise
        except Exception as exc:
            social_server.post_social_message(
                f"{dso_dir.name} {filter_name}: stack failed — {exc}"
            )
            continue

        safe_filter = filter_name.replace(" ", "_")
        # Results live with the data, not in scratch — see stacker.results_dir.
        out_dir = stacker.results_dir(dso_dir.name)
        fits_path = out_dir / f"stack_{dso_dir.name}_{safe_filter}.fits"
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

        out_path = out_dir / f"stack_{dso_dir.name}_{safe_filter}.jpg"
        jpg = stacker._save_jpg(
            result, out_path,
            title=(
                f"{dso_dir.name}  {filter_name}  {info['n_frames']} frames "
                f"({info['method']})\n{metrics}"
            ),
        )
        # Plain full-resolution copy with no title/axes, for actually using.
        plain_path = out_dir / f"stack_{dso_dir.name}_{safe_filter}_plain.jpg"
        try:
            stacker.save_plain_jpg(result, plain_path)
            saved = (f"\nSaved:\n  {fits_path}\n  {plain_path}"
                     f"  ({result.shape[1]}x{result.shape[0]}, no text)")
        except Exception as exc:
            _logger.warning("stack: plain JPEG failed for %s: %s", plain_path.name, exc)
            saved = f"\nSaved:\n  {fits_path}"
        social_server.post_social_message(
            f"{dso_dir.name} {filter_name}: {info['n_frames']} frames stacked — {metrics}"
            + saved,
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

    Moves that classify cleanly are auto-filed to good/ by the roof flow, so this
    command is now mainly for the leftovers — captures that came back "bad" or
    "unknown" and were parked in unlabeled/ for a human call. Labeling moves the
    capture's spectrogram + WAV from roof_audio/unlabeled/ to the good/bad library
    that classify() judges future moves against, and files the motor-current
    signature from the same move (same direction, within ±10 minutes) under the
    same verdict — one verdict per roof move.

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
        "live": live_cmd,
        "optics": optics_cmd,
        "drift": drift_cmd,
        "stack": stack_cmd,
        "process": process_cmd,
        "purge": purge_cmd,
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

    The roof open goes through the unified open_roof(imaging_run=True), which
    additionally enforces mount power off (expected at run start — the mount
    is powered on later by the NINA prelude via start.bat) and the roof lock
    (so a concurrent roof!! command can never overlap this open).
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
    # Safety gate 1: check before the initial wait.                        #
    # The user must have previously issued "safe!" via the web chat.       #
    # ------------------------------------------------------------------ #
    pushover.push_message("Starting run in 1 min", inside_view)
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

    # Fully gated open: open_roof re-checks safe!, requires the mount off
    # (expected at run start — power comes on later in the NINA prelude),
    # takes the roof lock, verifies parked+closed via vision, announces via
    # Sonos, then opens and confirms. imaging_run=True waives only its
    # imaging-in-progress gate, since image_cmd already claimed the run.
    ok = open_roof(imaging_run=True)
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

        # Report on what was actually imaged: the DSO the grid published when it
        # planned the night. Re-running the selector here would rank against the
        # COMING night's dark hours and weather and could name a different DSO
        # than the one just shot. Falls back to the live queue if unpublished.
        from control import tonight_target
        eon_dso = tonight_target.read()
        if not eon_dso:
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


def _newest_fits_mtime(image_dir: Path) -> float:
    """Return the newest *.fits modification time under image_dir, or 0.0 if none.

    Used as a liveness signal for the flats run: while NINA keeps writing frames
    this climbs; if it stops advancing the sequence has stalled.
    """
    newest = 0.0
    try:
        for f in image_dir.rglob("*.fits"):
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            if mt > newest:
                newest = mt
    except OSError:
        _logger.exception("Failed to scan %s for newest flat frame", image_dir)
    return newest


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

    # Stall out on inactivity, not wall-clock: a long-but-healthy flats run
    # (many filters/exposures) can exceed 30 min while still writing frames.
    # Terminate only if no new FITS frame has landed under image_dir for 10 min.
    idle_limit = 600  # seconds without a new frame ⇒ flats are stuck
    image_dir = Path(config.data()["nina"]["image_dir"])
    last_mtime = _newest_fits_mtime(image_dir)
    last_activity = time.time()
    _logger.info("Waiting for flats to complete (state = DONE_FLATS, idle timeout 10 min)")
    while get_imaging_state() != ImagingState.DONE_FLATS:
        time.sleep(30)
        mtime = _newest_fits_mtime(image_dir)
        if mtime > last_mtime:
            last_mtime = mtime
            last_activity = time.time()
        elif time.time() - last_activity > idle_limit:
            _logger.warning("Flats stalled — no new frame in 10 min, terminating NINA + PWI4")
            social_server.post_social_message(
                "WARNING: Flats stalled (no new frame in 10 min), terminating NINA + PWI4"
            )
            # Killing flats early skips the sequence's own final step, which
            # shuts down PWI4 — so PWI4 would be left orphaned (as happened
            # last night). Kill both here to mirror a normal flats finish.
            subprocess.run(
                [os.path.join(_SCRIPTS_DIR, "kill_nina_pwi4.bat")],
                shell=True,
            )
            break

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

    # Reuse the FWHM/star measurements frame_watcher already cached in
    # frame_stats.json during the night. Without this the run re-measures every
    # sub from scratch: on sh2-92 (2026-08-01) that was ~41 min of the 79 min
    # Ha convergence, redoing a detection pass whose answers were already on
    # disk for all 294 frames. Cached FWHM comes from header HFR when present,
    # so it is close to but not identical with _measure_fwhm_and_stars — the
    # quality-gate cut and reference pick can shift slightly. Same tradeoff the
    # hr/stack path already takes.
    precomputed = _load_precomputed_fwhm_stars(
        dso_dir, fits_files, float(cfg["nina"]["arc_sec_per_pixel"]))
    if precomputed:
        social_server.post_social_message(
            f"Convergence: reusing {len(precomputed)}/{len(fits_files)} cached "
            "FWHM/star measurements")

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scratch_dir = os.path.join(_project_root, cfg["scratch"]["directory"])

    from fits_processing import convergence as _conv

    from datetime import date as _date

    # Each filter's convergence is fully independent — different frames, its own
    # uniquely-named output JPGs, no shared cache writes — so run them in
    # parallel. The dominant cost (per-frame astroalign registration) is
    # single-core per filter, so on this multi-core box N filters finish in
    # roughly 1/N the wall-clock of the old sequential loop. Cap the pool so peak
    # RAM (~one downscaled frame cube per active filter) stays bounded and the
    # brief inner 4-thread bursts (FWHM measure / Fibonacci sampling) aren't
    # badly oversubscribed.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    saved: dict[str, dict] = {}
    _saved_lock = threading.Lock()

    # Built once (a master is ~1 s/frame) and cached to scratch against the
    # calibration frames' mtimes; the workers mmap it rather than each holding a
    # 245 MB copy. Without it the curve's y-axis is a percentage of the bias
    # pedestal rather than of sky.
    _calibration = stacker.calibration_from_config()      # bias+dark, for the probe below
    if _calibration is None:
        social_server.post_social_message(
            "No bias/dark configured — convergence RMSE will be a percentage of "
            "the bias pedestal, not of sky signal."
        )

    def _run_filter(fn: str, paths: list) -> None:
        jobs.raise_if_cancelled(_job_id)
        out = Path(scratch_dir) / f"convergence_{fn}.jpg"
        gold = Path(scratch_dir) / f"golden_{fn}.jpg"
        def _progress(msg: str, _fn: str = fn) -> None:
            social_server.post_social_message(f"Convergence [{_fn}]: {msg}")
        _t0 = time.perf_counter()
        try:
            counts, resid, slope_pct, final_rmse_pct = stacker.convergence_curve(
                paths,
                filter_name=fn,
                output_path=out,
                golden_output_path=gold,
                progress_cb=_progress,
                cancel_cb=_cancel,
                precomputed_fwhm_stars=precomputed,
                # Per filter, so this filter's flat comes along too.
                calibration=stacker.calibration_from_config(fn),
            )
        except jobs.Cancelled:
            raise
        except Exception as exc:
            # Log too: a filter that dies here used to leave nothing in iris.log,
            # so a half-finished `snr` looked like a hang rather than a failure.
            _logger.exception("Convergence [%s] failed after %.1fs",
                              fn, time.perf_counter() - _t0)
            social_server.post_social_message(f"Convergence [{fn}]: failed — {exc}")
            return
        # The curve's last point is the all-frames stack, so counts[-1] is exactly
        # how many frames went into the golden — which is what these numbers
        # describe. len(paths) is every LIGHT frame on disk, before the quality
        # cut and any registration failures; report both so the gap is visible
        # rather than showing a count the plots disagree with.
        stacked = counts[-1]
        n_frames = f"{stacked} of {len(paths)} frames" if stacked != len(paths) \
            else f"{stacked} frames"
        _logger.info("Convergence [%s]: %s in %.1fs", fn, n_frames, time.perf_counter() - _t0)
        with _saved_lock:
            saved[fn] = {
                "tail_slope_pct": round(slope_pct, 6),
                "final_rmse_pct": round(final_rmse_pct, 4),
                "frame_count": stacked,
                "total_frames": len(paths),
                "calibrated": _calibration is not None,
                "updated": _date.today().isoformat(),
            }
        social_server.post_social_message(
            f"Stack convergence vs golden — {fn}  ({n_frames})  slope {slope_pct:+.4f}%/frame  RMSE {final_rmse_pct:.2f}%",
            str(out),
        )
        social_server.post_social_message(
            f"Golden stack — {fn}  ({n_frames})",
            str(gold),
        )
        social_server.post_social_message(
            _conv.progress_summary(fn, counts, resid, slope_pct, final_rmse_pct, len(paths))
        )

    max_workers = min(len(by_filter), 4)
    _t_all = time.perf_counter()
    # Bind the job to each pool thread. jobs holds the current job in a
    # threading.local, so a worker thread starts with no job and every
    # post_social_message it makes — the progress lines and both plots — lands
    # on the system feed instead of this command's card. Only the final summary,
    # posted from the calling thread, was arriving in the right place.
    with ThreadPoolExecutor(max_workers=max_workers,
                            initializer=jobs.set_current_job,
                            initargs=(_job_id,)) as pool:
        futs = [pool.submit(_run_filter, fn, paths) for fn, paths in by_filter.items()]
        for fut in as_completed(futs):
            # Propagate a cancellation from any worker; still-running workers see
            # the same cancel flag at their next checkpoint and wind down too.
            fut.result()
    _logger.info(
        "Convergence: %d filter(s) done in %.1fs wall-clock (%d workers)",
        len(by_filter), time.perf_counter() - _t_all, max_workers,
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


# dip/bump ratios beyond this mean the mirrored (upward) search found nothing:
# transit.py floors the bump score at 1e-6, so "no upward signal" divides into
# the millions. Report that as fully one-sided rather than an absurd number.
_DIP_BUMP_ONE_SIDED_CAP = 1000.0


def _transit_confidence_text(sig: dict) -> str:
    """Plain-language readout of the three transit significance tests.

    One line per test: the number, then what it means for this candidate. The
    tests are computed in transit_search.transit._transit_significance; the
    verdict thresholds here are display-only guidance, not gates.
    """
    lines = []

    # Permutation false-alarm probability: fraction of random time-shuffles of
    # this star's own curve that score as well as the real detection.
    nperm = sig.get("n_permutations") or 0
    fap = sig.get("perm_fap")
    if fap is not None:
        floor = 1.0 / (nperm + 1) if nperm else 0.0
        if nperm and fap <= floor:
            lines.append(
                f"• noise test: none of {nperm} random shuffles of this curve "
                f"scored as high (FAP <{floor:.3f}) — strong"
            )
        else:
            verdict = ("strong" if fap <= 0.01 else
                       "borderline" if fap <= 0.05 else
                       "weak, consistent with random noise")
            pct = f"{fap * 100:.1f}" if fap < 0.095 else f"{fap * 100:.0f}"
            lines.append(
                f"• noise test: {pct}% of random shuffles of this "
                f"curve score as high (FAP {fap:.3f}) — {verdict}"
            )
    elif not nperm:
        lines.append("• noise test: not run")
    else:
        lines.append("• noise test: skipped — fewer than 30 usable epochs")

    # Field outlier: this star's dip score vs every other searched star in the
    # frame. Field stars share the night's systematics (clouds, focus drift),
    # so a real transit should stand out from them.
    fz = sig.get("field_z")
    if fz is not None:
        verdict = ("far outside the field, not a shared systematic" if fz >= 5
                   else "a moderate outlier" if fz >= 3
                   else "within the field's normal spread — could be a "
                        "systematic the whole field shows")
        lines.append(f"• vs other field stars: z={fz:.1f} — {verdict}")
    elif sig.get("field_fap") is not None:
        lines.append("• vs other field stars: n/a — the comparison stars' "
                     "scores are nearly all identical (zero spread)")
    else:
        lines.append("• vs other field stars: n/a — fewer than 5 comparison stars")

    # One-sidedness: a transit only dips; symmetric noise bumps up as often
    # as down, so the dip score should dwarf the best upward score.
    db = sig.get("dip_bump")
    if db is None:
        lines.append("• one-sidedness: n/a")
    elif db > _DIP_BUMP_ONE_SIDED_CAP:
        lines.append("• one-sidedness: no upward counterpart at all — fully "
                     "one-sided, as a transit should be")
    else:
        verdict = ("one-sided, as a transit should be" if db >= 3 else
                   "only weakly one-sided — could be symmetric noise")
        lines.append(f"• one-sidedness: dip outscores the best upward bump "
                     f"{db:.1f}× — {verdict}")

    return ("Confidence (a 2nd transit at the same period is still needed "
            "to confirm):\n" + "\n".join(lines))


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
        sig = top.get("significance") or {}
        conf = f"\n{_transit_confidence_text(sig)}" if sig else ""
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


def _save_frame_stats_cache(cache_path: Path, cached_by_path: dict[str, dict]) -> None:
    """Write the frame_stats cache atomically, leaving the old file intact on error."""
    import json as _json
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


def image_stats_cmd(words: list[str], account: str) -> None:
    """Post per-frame FWHM/eccentricity graph in a background thread (non-blocking)."""
    jobs.spawn(_image_stats_run, args=(words, account))


def _image_stats_run(words: list[str], account: str) -> None:
    """Worker for image_stats_cmd.

    Usage:
        stats                — latest session for the DSO currently being imaged
        stats <dso>          — latest session for the named DSO
        stats <dso> all      — full multi-night history for the named DSO
        stats <dso> resky    — recompute only the sky brightness on cached frames
        stats <dso> rebuild  — discard the cache and re-analyse every frame

    By default only the most recent observing session is plotted (see
    imaging_artifacts.gather_dso_frames) so a single big night can't dominate
    the frame-count x-axis. A trailing "all" opts into the full history; the
    option tokens combine in any order (e.g. `stats sh2-92 all resky`).

    For each FITS file in scope the path is looked up in <dso_dir>/frame_stats.json.
    Cached entries are used as-is; only files missing from the cache are opened and
    analysed.  Newly analysed frames are written back to the cache so subsequent
    runs skip them too.

    Because entries are reused verbatim, a cache written before a change to the
    analysis code keeps the old numbers forever. "rebuild" is the blunt fix and
    re-runs star detection on everything, which is slow. "resky" is the cheap one:
    it re-reads just the four corner blocks to refresh the sky fields, which is
    what goes stale when the bias/dark pedestal calibration changes
    (see fits_processing/sky_pedestal.py).
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

    # Trailing option tokens, in any order:
    #   all      — full multi-night history (default: latest session only, so one
    #              big night can't dominate the frame-count x-axis)
    #   rebuild  — ignore the cache and re-analyse every frame from scratch
    #   resky    — keep cached FWHM/ecc/star counts, recompute only the sky fields
    # Anything left over is the DSO name.
    extra = list(words[2:])
    latest_session_only = True
    rebuild = False
    resky = False
    while extra and extra[-1].lower() in ("all", "rebuild", "resky"):
        tok = extra.pop().lower()
        if tok == "all":
            latest_session_only = False
        elif tok == "rebuild":
            rebuild = True
        else:
            resky = True
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
    if cache_path and cache_path.exists() and not rebuild:
        try:
            with open(cache_path) as f:
                existing = _json.load(f)
            if isinstance(existing, list):
                for entry in existing:
                    if "path" in entry:
                        cached_by_path[str(Path(entry["path"]))] = entry
        except Exception:
            pass
    elif rebuild:
        social_server.post_social_message(
            "Rebuild: ignoring the cache, re-analysing every frame from scratch…"
        )

    # ── resky: refresh only the sky fields on cached entries ────────────────
    # Cheap compared to a full rebuild — reads the four corner blocks instead of
    # running star detection over the whole frame. Use this after changing the
    # pedestal calibration, which is what makes cached sky values stale.
    if resky and cached_by_path:
        from fits_processing import sky_pedestal as _sp
        from astropy.io import fits as _fits_r

        wanted = {str(f) for f in fits_files}
        targets = [e for p, e in cached_by_path.items() if p in wanted]
        social_server.post_social_message(
            f"Resky: recomputing sky brightness for {len(targets)} cached frames…"
        )

        def _refresh_sky(entry: dict) -> tuple[dict, bool]:
            try:
                lvl = _sp.corner_level(Path(entry["path"]))
                if lvl is None:
                    return entry, False
                h = _fits_r.getheader(entry["path"])
                exp = float(h.get("EXPTIME", h.get("EXPOSURE", 1.0)))
                ped = _sp.lookup(h.get("GAIN"), h.get("OFFSET"), h.get("CCD-TEMP"), exp)
                if ped is None:
                    entry["sky_adu_per_s"] = None
                    entry["pedestal_source"] = "uncalibrated"
                    return entry, False
                entry["sky_adu_per_s"] = round(
                    max(lvl - ped["pedestal_adu"], 0.0) / max(exp, 1e-6), 5
                )
                entry["pedestal_source"] = (
                    "extrapolated" if ped["extrapolated"] else "measured"
                )
                return entry, True
            except Exception:
                return entry, False

        n_ok = 0
        n_extrap = 0
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(targets)))) as pool:
            jobs.register_resource(_job_id, pool)
            try:
                for entry, ok in pool.map(_refresh_sky, targets):
                    jobs.raise_if_cancelled(_job_id)
                    cached_by_path[str(Path(entry["path"]))] = entry
                    n_ok += bool(ok)
                    n_extrap += entry.get("pedestal_source") == "extrapolated"
            finally:
                jobs.unregister_resource(_job_id, pool)

        if cache_path:
            _save_frame_stats_cache(cache_path, cached_by_path)
        msg = f"Resky: {n_ok}/{len(targets)} frames updated"
        if n_extrap:
            msg += (f" — {n_extrap} used an extrapolated pedestal "
                    "(no BIAS/DARK at that sensor temperature)")
        if n_ok < len(targets):
            msg += (f"; {len(targets) - n_ok} had no pedestal calibration — run "
                    "`python fits_processing/sky_pedestal.py --build`")
        social_server.post_social_message(msg)

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
                # 5 dp, not 2: pedestal-corrected sky runs ~0.00-0.05 ADU/s.
                "sky_adu_per_s":   round(sky["sky_adu_per_s"], 5)
                                   if sky and sky.get("sky_adu_per_s") is not None else None,
                "sky_mag_arcsec2": round(sky["sky_mag_arcsec2"], 2)
                                   if sky and sky.get("sky_mag_arcsec2") is not None else None,
                "pedestal_source": sky.get("pedestal_source") if sky else None,
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
            _save_frame_stats_cache(cache_path, cached_by_path)
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

def process_cmd(words: list[str], account: str) -> None:
    """Stack a DSO's filters and combine them into a colour image.

    Usage:
        process <dso> <recipe>                  — recipe is LRGB, HOO or SHO
        process <dso> <recipe> noflat           — bias+dark only, no flats
        process <dso> <recipe> reuse black=50   — re-render from cached channels

    Display options (all optional, any order):
        black=65    percentile of each channel sent to black; lower = brighter
        white=99    percentile taken as full white
        soft=0.025  asinh softening; smaller lifts the faint end harder
        mesh=4      background mesh boxes across the short axis; higher removes
                    gradients harder but eats large nebulosity
        nobg        skip background subtraction entirely
        scale=N     bin the output N-fold (quick look)
        reuse       skip stacking and re-render the cached channels — seconds
                    instead of ~20 minutes, which is the only sane way to tune
                    the options above

    Recipes:
        LRGB   R->red, G->green, B->blue, with L substituted as luminance
        HOO    Ha->red, O-III->green and blue
        SHO    S-II->red, Ha->green, O-III->blue   (the "Hubble" palette)

    Every filter is registered to one shared reference so the channels land on
    the same pixels, then combined on a shared brightness scale so the ratio
    between channels — which is the whole point of a palette — survives into
    the picture. Frames are bias/dark/flat calibrated, quality-gated and stacked
    sigma-clipped at full resolution; expect ~20 minutes on a few hundred frames.

    `noflat` is there because a flat from the wrong epoch can be worse than
    none — dust migrates and focus shifts, so flats shot months after the lights
    may stamp in a mote the data never had. Worth trying both ways when the only
    flats available come from a different run.

    Writes two files to <image_dir>/Iris/<dso>/ and names both in its reply:
        process_<dso>_<recipe>.jpg           full resolution, no text
        process_<dso>_<recipe>_preview.jpg   2200 px, posted to the card

    Process-isolated (jobs.spawn_process) like stack, for the same reason.
    """
    jobs.spawn_process(_process_run, args=(words,))


def _process_run(words: list[str]) -> None:
    _job_id = jobs.get_current_job()
    _cancel = jobs.cancel_cb_for(_job_id)

    # This module imports numpy and stacker per-function, not at module scope.
    import numpy as np
    from stacking import color_process, stacker

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])

    extra = [w for w in (words[2:] if len(words) > 2 else []) if w]

    # Display knobs as key=value, plus bare flags. The stretch is a second and
    # the stack is 17 minutes, so `reuse` re-renders from the cached channels.
    _FLAGS = {"noflat": ("use_flats", False), "no-flat": ("use_flats", False),
              "noflats": ("use_flats", False), "no-flats": ("use_flats", False),
              "nobg": ("subtract_background", False),
              "no-bg": ("subtract_background", False),
              "reuse": ("reuse", True)}
    _KEYS = {"black": ("black_pct", float), "white": ("white_pct", float),
             "soft": ("softening", float), "mesh": ("mesh", int),
             "scale": ("scale", int)}
    opts = {"use_flats": True, "reuse": False}
    rest = []
    bad = []
    for w in extra:
        lw = w.lower()
        if lw in _FLAGS:
            k, v = _FLAGS[lw]; opts[k] = v; continue
        if "=" in lw:
            k, _, v = lw.partition("=")
            if k in _KEYS:
                name, cast = _KEYS[k]
                try:
                    opts[name] = cast(v)
                except ValueError:
                    bad.append(w)
                continue
            bad.append(w); continue
        rest.append(w)
    if bad:
        social_server.post_social_message(
            f"Unrecognised option(s): {', '.join(bad)}. "
            f"Known: {', '.join(sorted(_KEYS))}=value, "
            f"{', '.join(sorted(set(k for k in _FLAGS)))}")
        return
    extra = rest
    use_flats = opts.pop("use_flats")
    reuse = opts.pop("reuse")
    scale = opts.pop("scale", 1)
    recipe = None
    for i, w in enumerate(extra):
        if w.upper() in color_process.RECIPES:
            recipe = w.upper()
            extra = extra[:i] + extra[i + 1:]
            break
    dso_arg = " ".join(extra).strip()
    if not dso_arg or recipe is None:
        social_server.post_social_message(
            "Usage: process <dso> <recipe> [noflat], recipe one of "
            f"{', '.join(sorted(color_process.RECIPES))}  e.g. `process sh2-92 hoo`")
        return

    def _find_dso_dir_by_name(name: str) -> Optional[Path]:
        target = name.lower().replace(" ", "").replace("_", "")
        candidates = [d for d in image_dir.iterdir()
                      if d.is_dir()
                      and target in d.name.lower().replace(" ", "").replace("_", "")]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        def _latest(d: Path) -> float:
            try:
                return max(f.stat().st_mtime for f in d.rglob("*.fits"))
            except ValueError:
                return 0.0
        return max(candidates, key=_latest)

    dso_dir = _find_dso_dir_by_name(dso_arg)
    if dso_dir is None:
        social_server.post_social_message(f"No image directory found for '{dso_arg}'")
        return

    def _progress(msg: str) -> None:
        social_server.post_social_message(f"{dso_dir.name} {recipe}: {msg}")

    # Resolve where the output goes BEFORE the half-hour of stacking. A mistake
    # in this block used to surface only after the work was finished — an
    # undefined name here threw away a complete 33-minute abell2151 run — and
    # anything wrong with the destination is knowable up front.
    out_dir = stacker.results_dir(dso_dir.name)
    # Keep the no-flat render under its own name: the whole reason to run one is
    # to compare it against the flat-corrected version, and sharing a filename
    # would overwrite the thing being compared against.
    tag = recipe if use_flats else f"{recipe}_noflat"
    full_path = out_dir / f"process_{dso_dir.name}_{tag}.jpg"
    prev_path = out_dir / f"process_{dso_dir.name}_{tag}_preview.jpg"

    social_server.post_social_message(
        f"Processing {dso_dir.name} as {recipe} at full resolution"
        f"{'' if use_flats else ' (no flats)'} — "
        f"this takes a while on a few hundred frames.")
    _t0 = time.perf_counter()
    try:
        rgb, info = color_process.process_dso(
            dso_dir, recipe, progress_cb=_progress, cancel_cb=_cancel,
            use_flats=use_flats, scale=scale, cache_dir=out_dir, reuse=reuse,
            **opts)
    except jobs.Cancelled:
        raise
    except Exception as exc:
        _logger.exception("process %s %s failed", dso_dir.name, recipe)
        social_server.post_social_message(f"{dso_dir.name} {recipe}: failed — {exc}")
        return

    # The channel stacks are cached by process_dso, which is both the salvage
    # path and what makes `reuse` fast — no need for a separate RGB dump.
    color_process.save_rgb(rgb, full_path)
    color_process.save_rgb(rgb, prev_path, max_px=2200)

    chans = "  ".join(f"{c}={info['channels'][c]}" for c in ("R", "G", "B", "L")
                      if c in info["channels"])
    frames = "  ".join(f"{f}:{n}" for f, n in sorted(info["frames"].items()))
    h, w = info["shape"]
    social_server.post_social_message(
        f"{dso_dir.name} — {recipe}"
        f"{'' if info.get('flats', True) else '  (no flats)'}   {chans}\n"
        f"frames stacked: {frames}   ({time.perf_counter() - _t0:.0f}s)\n"
        f"shared reference: {info['reference']}\n"
        f"Saved:\n  {full_path}  ({w}x{h}, no text)\n  {prev_path}  (preview)",
        str(prev_path),
    )


def purge_cmd(words: list[str], account: str) -> None:
    """Delete superseded flat frames, keeping the most recent set per filter.

    Usage:
        purge            — dry run: list what would be deleted
        purge go         — actually delete

    Flats accumulate fast: 20 frames per filter per night at 122 MB each is
    2.45 GB a night for two filters, and this observatory has ~1000 of them.
    They are also highly redundant — masters built from 2026-07-14 and
    2026-07-31 agreed to 0.2-0.3% RMS, so the optical train had not moved and
    every set but the newest was contributing noise reduction at best.

    Dry run by default. This is the only command that deletes data outright
    (`bad` renames), so it will not act without `go`.
    """
    jobs.spawn(_purge_run, args=(words,))


def _purge_run(words: list[str]) -> None:
    _job_id = jobs.get_current_job()

    from stacking import stacker

    cfg = config.data()
    cal = cfg.get("calibration", {}) or {}
    root = cal.get("flat_root")
    if not root:
        social_server.post_social_message(
            "No calibration.flat_root configured — nothing to purge.")
        return
    root = Path(root)

    do_delete = any(w.lower() == "go" for w in (words[2:] if len(words) > 2 else []))

    by_filter = stacker.flats_by_filter(root)
    if not by_filter:
        social_server.post_social_message(f"No flats found under {root}")
        return

    # A session's flats live in one FLAT directory, and one directory holds
    # several filters, so "newest" is decided per filter: the most recent
    # session that actually contains that filter.
    keep: set[Path] = set()
    kept_desc: list[str] = []
    doomed: list[Path] = []
    for filt, paths in sorted(by_filter.items()):
        sessions: dict[Path, list[Path]] = {}
        for p in paths:
            sessions.setdefault(p.parent, []).append(p)
        newest = max(sessions, key=lambda d: max(f.stat().st_mtime for f in sessions[d]))
        keep.update(sessions[newest])
        kept_desc.append(f"{filt}: {len(sessions[newest])} from {newest.parent.name}"
                         f" ({len(sessions)} sets on disk)")
        for d, fs in sessions.items():
            if d != newest:
                doomed.extend(fs)

    doomed = sorted(set(doomed) - keep)
    freed = sum(f.stat().st_size for f in doomed) / 1e9
    total = sum(len(v) for v in by_filter.values())

    lines = [f"Flat purge under {root.name} — {total} flats on disk",
             "Keeping the newest set per filter:"]
    lines += [f"  {d}" for d in kept_desc]
    if not doomed:
        lines.append("Nothing to purge — every filter already has one set.")
        social_server.post_social_message("\n".join(lines))
        return
    lines.append(f"{'Deleting' if do_delete else 'Would delete'} "
                 f"{len(doomed)} older flats, freeing {freed:.1f} GB")
    if not do_delete:
        lines.append("Re-run as `purge go` to actually delete. "
                     "Note this is irreversible: older flats are the only way to "
                     "calibrate older lights if the optics were ever disturbed.")
        social_server.post_social_message("\n".join(lines))
        return

    removed, failed = 0, []
    for f in doomed:
        jobs.raise_if_cancelled(_job_id)
        try:
            f.unlink()
            removed += 1
        except Exception as exc:
            failed.append((f, str(exc)))
    # Tidy up FLAT directories left empty, and their session dir if it is now bare.
    for d in {f.parent for f in doomed}:
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                if d.parent.is_dir() and not any(d.parent.iterdir()):
                    d.parent.rmdir()
        except Exception:
            pass
    # The cached masters were keyed on the deleted files' paths and mtimes, so
    # the next calibration rebuilds from what remains rather than reusing them.
    stacker._FLATS_BY_FILTER_CACHE.clear()

    lines.append(f"Deleted {removed}/{len(doomed)}, freed {freed:.1f} GB")
    if failed:
        lines.append(f"Failed on {len(failed)}; first: {failed[0][0].name} — {failed[0][1]}")
    _logger.info("purge: deleted %d flats, freed %.1f GB", removed, freed)
    social_server.post_social_message("\n".join(lines))
