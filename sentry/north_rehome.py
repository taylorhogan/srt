"""Re-home Iris North by cycling its power when its aim has slipped.

Iris North's pan/tilt counts motor steps and has no encoder. On 2026-09-25 and
09-26/27 it lost steps: the camera still reported its home pose (102, -9) while
aiming ~95 px off, both roof references drifted together, and by 10-01 a
correct OPEN read 99 px off against a 100 px tolerance. A power cycle makes it
re-find its end stops, which fixed it (10-02: back to 8.7 px).

Measured 2026-10-05 with the new Kasa plug ("Iris north camera"): the stream
drops when the plug goes off and is back ~27 s after it comes on; WITHIN a boot
the aim repeats to 0.2 px from all four approach directions (42 px, four
times); ACROSS boots it lands a few tens of px apart (8.7, 26, 42). So a cycle
fixes a slip of ~95 px but leaves normal scatter of up to ~40 px, and the
trigger sits above that: DRIFT_PX.

Drift is told apart from a roof caught part way by the tag's apparent size: at
an end of travel the tag is the size it was when that end was recorded (72 px
shut, 350 px open); part way it is neither.

What acts on it:
  * the 09:00 morning check, with the observatory at rest, cycles and re-reads
  * `rehome north` in the chat, same guard
  * any other read only WARNS (once a day): cutting the camera during a roof
    operation would blind the roof decision.
"""
import json
import logging
import math
import os
import statistics
import time
from datetime import date

_logger = logging.getLogger(__name__)

DRIFT_PX = 60.0            # above boot-to-boot scatter (<= ~42 px), below the 100 px tolerance
ELSEWHERE_MAX_PX = 250.0   # beyond this from the matching end, call it unclear, not drift
SIZE_TOL = 0.12            # tag side within 12% of an end's recorded size = the roof is at that end
PLUG_DEFAULT = "Iris north camera"
OFF_S = 10
STREAM_WAIT_S = 120
WARN_MARKER = "local/north_drift_warned.txt"


def _side(corners):
    return math.hypot(corners[0][0] - corners[1][0], corners[0][1] - corners[1][1])


def drift_verdict(per_frame, ref):
    """{"state": ok|drift|unclear|no_tag, "end", "offset_px"} from one north read. Pure.

    *per_frame* is north_roof.last_detail["per_frame"]; *ref* the reference file.
    """
    tag = str(ref.get("tag_id", 2))
    sides = {end: _side(ref[end]["markers"][tag]) for end in ("shut", "open")}
    votes = []
    for f in per_frame or []:
        s = f.get("side_px")
        if not s:
            continue
        for end in ("shut", "open"):
            if abs(s / sides[end] - 1) <= SIZE_TOL and f.get("off_%s_px" % end) is not None:
                votes.append((end, f["off_%s_px" % end]))
    if not votes:
        decoded = any(f.get("side_px") for f in per_frame or [])
        return {"state": "unclear" if decoded else "no_tag", "end": None, "offset_px": None}
    ends = [e for e, _ in votes]
    end = max(set(ends), key=ends.count)
    off = statistics.median(o for e, o in votes if e == end)
    if off > ELSEWHERE_MAX_PX:
        state = "unclear"
    elif off > DRIFT_PX:
        state = "drift"
    else:
        state = "ok"
    return {"state": state, "end": end, "offset_px": round(off, 1)}


def rehome_refusal(night, nina_running, imaging_state, scheduler_state, roof_moving=False):
    """Why Iris North must NOT be power-cycled now, or None. Pure; unknown refuses."""
    if night is not False:
        return "it is not confirmed daytime"
    if nina_running is not False:
        return "N.I.N.A may be running"
    if imaging_state != "NONE":
        return "imaging state is %s" % (imaging_state or "unknown")
    if scheduler_state is None or scheduler_state == "IMAGING":
        return "the scheduler is %s" % (scheduler_state or "unknown")
    if roof_moving:
        return "the roof is moving"
    return None


def _plug_ip(alias):
    import asyncio
    from hardware_control import kasa_utils as ku
    return asyncio.run(ku.make_discovery_map(expect=(alias,))).get(alias)


def rehome(alias=None):
    """Cycle the camera's plug, wait for its stream, drive home the standard way,
    re-read. Returns a result dict; never raises. Attempted ONCE per call."""
    from configs import config
    from hardware_control import kasa_utils as ku
    from scripts import kasa_ptz
    from sentry import north_roof
    from sentry.sky_camera import credentials

    cfg = config.data()
    alias = alias or (cfg.get("camera safety") or {}).get("north_power_plug", PLUG_DEFAULT)
    out = {"plug": alias, "ok": False}
    try:
        ref = north_roof.reference()
        if ref is None:
            out["error"] = "no north reference on file"
            return out
        ip = _plug_ip(alias)
        if not ip:
            out["error"] = "plug %r not found on the LAN" % alias
            return out
        if ku.legacy_relay(ip, 0) != 0:
            out["error"] = "plug did not read back OFF"
            ku.legacy_relay(ip, 1)
            return out
        _logger.warning("north_rehome: %r off for %d s", alias, OFF_S)
        time.sleep(OFF_S)
        on = None
        for _ in range(3):
            on = ku.legacy_relay(ip, 1)
            if on == 1:
                break
            time.sleep(2)
        if on != 1:
            out["error"] = "plug did NOT come back ON -- Iris North is unpowered"
            _logger.error("north_rehome: %s", out["error"])
            return out
        user, pw = credentials(cfg)
        probe = os.path.join("local", "north_rehome_probe.jpg")
        t0 = time.monotonic()
        while time.monotonic() - t0 < STREAM_WAIT_S:
            time.sleep(5)
            if north_roof._grab(ref["host"], user, pw, probe, retries=0) is not None:
                out["boot_s"] = round(time.monotonic() - t0)
                break
        else:
            out["error"] = "camera stream not back %d s after power-on" % STREAM_WAIT_S
            return out
        # Standard approach: within a boot this repeats to 0.2 px from every side.
        home = tuple(ref["ptz"])
        dev = kasa_ptz._device(ref.get("camera", "Iris North"))
        kasa_ptz.step(dev, "left", 1)
        kasa_ptz.step(dev, "bottom", 1)
        out["pose"] = list(kasa_ptz.goto(dev, home[0], home[1], verbose=False, settle=True))
        state, det = north_roof.read(frames=3)
        out["state"] = state
        out["after"] = drift_verdict(det.get("per_frame"), ref)
        out["ok"] = out["after"]["state"] == "ok"
        _logger.warning("north_rehome: after the cycle north reads %s, %s", state, out["after"])
        return out
    except Exception as exc:  # noqa: BLE001
        _logger.exception("north_rehome failed")
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out


def describe(before, result):
    """One line for chat/push. Pure."""
    b = before or {}
    head = "Iris North had slipped %s px off its %s reference" % (b.get("offset_px"), b.get("end"))
    if result.get("error"):
        return "%s; re-home FAILED: %s" % (head, result["error"])
    a = result.get("after") or {}
    tail = "now %s px off %s" % (a.get("offset_px"), a.get("end")) if a.get("offset_px") is not None \
        else "tag not read after the cycle (%s)" % a.get("state")
    return "%s; power-cycled %r (back in %s s) and re-homed: %s%s" % (
        head, result.get("plug"), result.get("boot_s"), tail,
        "" if result.get("ok") else " -- STILL OUT, check the camera")


def note_read(per_frame, ref):
    """Called after every north read: warn (once a day) if the aim has slipped.
    Never acts and never raises -- the morning check is what re-homes."""
    try:
        v = drift_verdict(per_frame, ref)
        if v["state"] != "drift":
            return v
        today = date.today().isoformat()
        try:
            if open(WARN_MARKER).read().strip() == today:
                return v
        except OSError:
            pass
        os.makedirs(os.path.dirname(WARN_MARKER), exist_ok=True)
        with open(WARN_MARKER, "w") as fh:
            fh.write(today)
        msg = ("Iris North's aim may have slipped: tag %.0f px off its %s reference (re-home above %d px). "
               "The 09:00 morning check will power-cycle it; or run `rehome north` when the observatory is idle."
               % (v["offset_px"], v["end"], DRIFT_PX))
        _logger.warning(msg)
        from utils import pushover
        pushover.push_message(msg)
        return v
    except Exception:  # noqa: BLE001
        _logger.exception("north drift note failed")
        return None
