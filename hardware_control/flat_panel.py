"""The flat panel's light, read and switched through its ASCOM driver.

The Optec Alnitak Flat-Man on the observatory PC holds its last state: a
light left on stays on until something says otherwise. On 2026-11-25 the
flats run lost its filter wheel, every narrowband Auto Exposure Flat
saturated and failed, N.I.N.A skipped each failed block's own "light off",
and the sequence's closing steps and the stall kill touch PWI4 and the
Pegasus ports but never the panel: it sat lit at full brightness for a day
and a half until the owner walked in. So the flats routine now switches the
panel off itself, whichever way the run ended, and the morning check reads
it and says so.

Everything goes through the driver (``ASCOM.OptecAlnitak.CoverCalibrator``)
from a PowerShell child, because the driver is a COM object and the one
thing on the machine that speaks the panel's dialect: the controller
answered none of the classic Alnitak serial commands on its port. The
driver caches what it last did, and on 2026-11-26 it reported the light
off while the panel was lit, so :func:`ensure_off` does not trust a
reading of "off": it switches the light on at the lowest level and off
again, so a real "off" reaches the hardware every time.

A busy port (N.I.N.A still closing, or open by hand) reads as an error
from the driver; :func:`ensure_off` waits and tries again for a minute.
Nothing here runs anywhere but Windows; elsewhere every call answers
"unreadable" and nothing raises.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import time

_logger = logging.getLogger(__name__)

PROGID = "ASCOM.OptecAlnitak.CoverCalibrator"
STATE_NAMES = {0: "not present", 1: "off", 2: "not ready", 3: "on",
               4: "unknown", 5: "error"}
TIMEOUT_S = 45

# The driver's CalibratorState: 1 is Off, 3 is Ready (lit and stable), 2 is
# NotReady (lit, settling). 0 means the driver sees no calibrator at all.
_LIT = {2, 3}

_PS = r'''
$ErrorActionPreference = "Stop"
try {
  $c = New-Object -ComObject "%(progid)s"
  $c.Connected = $true
  %(action)s
  $o = @{ state = [int]$c.CalibratorState; brightness = [int]$c.Brightness; max = [int]$c.MaxBrightness }
  $c.Connected = $false
  $o | ConvertTo-Json -Compress
} catch {
  @{ error = $_.Exception.Message } | ConvertTo-Json -Compress
}
'''
_ACTION_OFF = ("$c.CalibratorOn(1); Start-Sleep -Milliseconds 800; "
               "$c.CalibratorOff(); Start-Sleep -Milliseconds 800")


def progid(cfg=None) -> str:
    """The driver's ProgID, from ``cfg["nina"]["flat_panel_progid"]`` when set."""
    try:
        if cfg is None:
            from configs import config
            cfg = config.data()
        return str(cfg.get("nina", {}).get("flat_panel_progid") or PROGID)
    except Exception:  # noqa: BLE001
        return PROGID


def parse(text: str) -> dict:
    """The driver's answer as a status dict. Pure.

    Returns {"ok", "light": "on"|"off"|None, "state", "state_name",
    "brightness", "max", "why"}; the last JSON line of ``text`` is the answer,
    anything before it is PowerShell chatter.
    """
    answer = None
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                answer = json.loads(line)
            except ValueError:
                continue
            break
    if answer is None:
        return _unreadable("driver gave no answer")
    if "error" in answer:
        return _unreadable(str(answer["error"]).strip() or "driver error")
    try:
        state = int(answer["state"])
        bright = int(answer.get("brightness", 0))
        mx = int(answer.get("max", 0)) or None
    except (KeyError, TypeError, ValueError):
        return _unreadable("driver answer malformed: %r" % (answer,))
    name = STATE_NAMES.get(state, "state %d" % state)
    if state in _LIT:
        light = "on"
    elif state == 1:
        light = "off"
    else:
        return {"ok": False, "light": None, "state": state, "state_name": name,
                "brightness": bright, "max": mx,
                "why": "driver reports calibrator %s" % name}
    return {"ok": True, "light": light, "state": state, "state_name": name,
            "brightness": bright, "max": mx, "why": None}


def _unreadable(why: str) -> dict:
    return {"ok": False, "light": None, "state": None, "state_name": None,
            "brightness": None, "max": None, "why": why}


def describe(status: dict) -> str:
    """One line for a log or a push. Pure."""
    if not status or not status.get("ok"):
        return "flat panel unreadable: %s" % ((status or {}).get("why") or "no reading")
    if status["light"] == "off":
        return "flat panel light off"
    mx = status.get("max")
    return "flat panel light ON at %s%s" % (
        status.get("brightness"), (" of %d" % mx) if mx else "")


def port_busy(status: dict) -> bool:
    """True when the driver's error reads like a port another program holds."""
    why = ((status or {}).get("why") or "").lower()
    return any(w in why for w in ("denied", "in use", "busy", "port", "timed out"))


def _run(action: str = "", cfg=None, timeout_s: float = TIMEOUT_S) -> dict:
    if platform.system() != "Windows":
        return _unreadable("no ASCOM driver on this machine")
    script = _PS % {"progid": progid(cfg), "action": action}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", script],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _unreadable("driver did not answer in %.0f s" % timeout_s)
    except OSError as e:
        return _unreadable("powershell could not start: %s" % e)
    return parse(out.stdout + "\n" + out.stderr)


def status(cfg=None) -> dict:
    """What the driver says the light is doing now. Never raises."""
    return _run("", cfg)


def ensure_off(cfg=None, retries: int = 6, pause_s: float = 10.0, run=_run,
               sleep=time.sleep) -> dict:
    """Switch the light off and return the reading after. Never raises.

    A busy port is retried ``retries`` times ``pause_s`` apart (N.I.N.A takes
    a few seconds to let go of the port after its sequence ends). Any other
    failure is returned at once.
    """
    st = None
    for attempt in range(retries + 1):
        st = run(_ACTION_OFF, cfg)
        if st.get("ok") or not port_busy(st):
            break
        if attempt < retries:
            _logger.info("flat panel: port busy (%s); retry in %.0f s", st.get("why"), pause_s)
            sleep(pause_s)
    return st


if __name__ == "__main__":
    import sys
    st = ensure_off() if "off" in sys.argv[1:] else status()
    print(describe(st))
    sys.exit(0 if st.get("ok") else 1)


# --- the camera's word -------------------------------------------------------
# Iris cam sees the whole panel square-on from its reference pose, with the
# inside light on (every gating read switches it on). Measured on the frames of
# 2026-11-26: the lit panel's median grey was 251-255 against the pine wall
# beside it at 121-134; off, 185 against 150. The driver had reported "off"
# while the panel was lit, so the picture, not the driver, decides "lit".

GLOW_MIN = 70          # panel median minus wall median, grey levels, lit above it
PANEL_BOX = (590, 100, 110, 90)   # x, y, w, h on the 1280 x 720 frame: the panel's
                                  # upper-left quadrant, clear of the tube end and
                                  # of the annotation text
WALL_BOX = (860, 100, 120, 90)    # the pine wall at the same height


def _box(cfg, key, default):
    try:
        b = (cfg or {}).get("camera safety", {}).get(key)
        if b and len(b) == 4:
            return tuple(int(v) for v in b)
    except Exception:  # noqa: BLE001
        pass
    return default


def seen_lit(frame, cfg=None):
    """What Iris cam's frame says about the panel. Pure.

    frame : the raw BGR frame of a gating read (kasa_state.last_frame), or
            None. Returns {"ok", "lit", "panel", "wall", "excess", "why"};
            ``ok`` False when there is no frame or the boxes fall outside it.
    """
    if frame is None:
        return {"ok": False, "lit": None, "panel": None, "wall": None,
                "excess": None, "why": "no camera frame"}
    try:
        import numpy as np
        g = frame if frame.ndim == 2 else frame.mean(axis=2)
        px, py, pw, ph = _box(cfg, "flat_panel_box", PANEL_BOX)
        wx, wy, ww, wh = _box(cfg, "flat_panel_wall_box", WALL_BOX)
        if py + ph > g.shape[0] or px + pw > g.shape[1] or wy + wh > g.shape[0] \
                or wx + ww > g.shape[1]:
            return {"ok": False, "lit": None, "panel": None, "wall": None,
                    "excess": None, "why": "frame %dx%d smaller than the boxes"
                    % (g.shape[1], g.shape[0])}
        panel = float(np.median(g[py:py + ph, px:px + pw]))
        wall = float(np.median(g[wy:wy + wh, wx:wx + ww]))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "lit": None, "panel": None, "wall": None,
                "excess": None, "why": "frame unreadable: %s" % exc}
    glow_min = float(((cfg or {}).get("camera safety", {}) or {}).get("flat_panel_glow_min", GLOW_MIN))
    excess = panel - wall
    return {"ok": True, "lit": excess > glow_min, "panel": round(panel), "wall": round(wall),
            "excess": round(excess), "why": None}


def describe_seen(seen):
    """One line for the camera's reading. Pure."""
    if not seen or not seen.get("ok"):
        return "camera: no reading (%s)" % ((seen or {}).get("why") or "no frame")
    return "camera sees it %s (panel %d, wall %d)" % (
        "LIT" if seen["lit"] else "dark", seen["panel"], seen["wall"])
