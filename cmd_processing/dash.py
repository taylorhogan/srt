"""A very short observatory status for the Core2 desk display (2026-10-22).

The Core2 (M5Stack, 320x240, ESP32) shows a glanceable summary and nothing
else: no buttons act, nothing here moves or powers anything. Every input is a
file read or a read-only status call:

    roof    the newest north-camera read in local/north_archive (its verdict is
            in the file name); never a fresh read, which would drive the camera
    scope   PWI4's GET /status (3 s timeout); never get_is_parked(), which
            CONNECTS the mount when it is not connected
    night   the conductor's /v1/state and /v1/plan
    frames  frame_ticker.json and the generated sequence (read_back)

The display is dumb on purpose: the colors are decided here, where the plan
is known, so the sketch only paints what it is given. "c" is g (fine),
a (look at it), r (act now) or x (no data).
"""
import json
import logging
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NORTH_ARCHIVE = ROOT / "local" / "north_archive"
NIGHT_STATUS = ROOT / "local" / "night_plan_status.json"
HOLD_STATES = ("SAFE_HOLD", "FAULT_ROOF_UNKNOWN", "ESTOP")
# The top band has room for 14 characters beside the clock.
SHORT_STATE = {"FAULT_ROOF_UNKNOWN": "ROOF FAULT", "ESTOP": "E-STOP"}


def _state_label(state: str) -> str:
    return SHORT_STATE.get(state, state.replace("_", " "))[:14]
DISK_WARN_GB, DISK_ALERT_GB = 100.0, 25.0        # the morning check / image!! thresholds
_NAME = re.compile(r"^(\d{6})_\d+_([a-z]+)(_ir)?\.jpg$")


# ----------------------------------------------------------------- pure parts

def roof_from_names(day: str, names) -> Optional[dict]:
    """{"v", "at"} from one archive day's file names (HHMMSS_i_verdict[_ir].jpg). Pure.

    The newest read is every frame with the newest stamp. It counts as SHUT or
    OPEN only when all of its frames agree, the same "every frame" bar the
    gate uses; anything else is "?"."""
    reads = {}
    for n in names:
        m = _NAME.match(n)
        if m:
            reads.setdefault(m.group(1), []).append(m.group(2))
    if not reads:
        return None
    stamp = max(reads)
    verdicts = set(reads[stamp])
    v = verdicts.pop().upper() if len(verdicts) == 1 and verdicts <= {"shut", "open"} else "?"
    at = datetime.strptime(day + stamp, "%Y-%m-%d%H%M%S")
    return {"v": v, "at": at}


def scope_from_status(status) -> str:
    """OFF / SLEWING / TRACKING / STOPPED from a PWI4 status object, or "?" without one. Pure."""
    if status is None:
        return "?"
    m = status.mount
    if not m.is_connected:
        return "OFF"
    if m.is_slewing:
        return "SLEWING"
    return "TRACKING" if m.is_tracking else "STOPPED"


def roof_window(summary: str) -> tuple:
    """("17:52", "06:05") from night_plan_status's "Roof opens ~17:52, closes ~06:05". Pure."""
    m = re.search(r"opens ~(\d{1,2}:\d{2}), closes ~(\d{1,2}:\d{2})", summary or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def _in_window(now: datetime, opens: Optional[str], closes: Optional[str]) -> bool:
    """True when *now*'s clock time is inside opens..closes (which runs past midnight)."""
    if not opens or not closes:
        return False
    t = now.strftime("%H:%M")
    return t >= opens or t < closes if opens > closes else opens <= t < closes


def _plus(hhmm: str, minutes: int) -> str:
    h, m = map(int, hhmm.split(":"))
    t = (h * 60 + m + minutes) % (24 * 60)
    return "%02d:%02d" % (t // 60, t % 60)


def next_slot(slots, now: datetime) -> Optional[dict]:
    """{"dso", "at"} of the first slot that has not started yet. Pure."""
    for s in slots or []:
        start = datetime.fromisoformat(s["start"])
        if start.replace(tzinfo=None) > now.replace(tzinfo=None):
            return {"dso": s["dso"], "at": start.strftime("%H:%M")}
    return None


def summarize(now: datetime, roof: Optional[dict], scope: str, conductor: Optional[dict],
              plan: Optional[dict], night_summary: str, live: Optional[dict],
              disk_gb: Optional[float]) -> dict:
    """The display's whole payload, colors decided. Pure.

    *conductor* is /v1/state, *plan* /v1/plan, *live* {"dso", "filter", "n",
    "of"} when frames are coming in, else None."""
    opens, closes = roof_window(night_summary)
    alerts = []

    if roof is None:
        roof_out = {"v": "?", "age": None, "c": "a"}
    else:
        age = int((now - roof["at"]).total_seconds() // 60)
        c = "a" if roof["v"] == "?" else "g"
        if roof["v"] == "OPEN" and not _in_window(now, opens, closes):
            # Open outside tonight's planned window: worth a look, and red once
            # it is half an hour past the planned close (the dawn close that an
            # internet outage can stop, 2026-10-20).
            # A roof the operator opened by hand (MANUAL_*) is never red.
            manual = str((conductor or {}).get("state") or "").startswith("MANUAL")
            late = closes and not manual and _in_window(now, _plus(closes, 30), opens or "12:00")
            c = "r" if late else "a"
            alerts.append("roof open late" if late else "roof open off-plan")
        roof_out = {"v": roof["v"], "age": age, "c": c}

    # PWI4 is not running by day (the mount is powered off), so no answer is
    # normal then; at night, inside the roof window, it is worth a look.
    night_now = _in_window(now, opens, closes)
    if scope == "?":
        scope_out = {"v": "PWI4 OFF", "c": "a" if night_now else "g"}
    else:
        scope_out = {"v": scope, "c": "g"}

    state = (conductor or {}).get("state")
    if state is None:
        night = {"v": "NO CONDUCTOR", "c": "r"}
        alerts.append("conductor down")
    elif state in HOLD_STATES:
        night = {"v": _state_label(state), "c": "r"}
        alerts.append(_state_label(state).lower())
    else:
        night = {"v": _state_label(state), "c": "g"}

    disk = None
    if disk_gb is not None:
        c = "r" if disk_gb < DISK_ALERT_GB else "a" if disk_gb < DISK_WARN_GB else "g"
        disk = {"gb": int(disk_gb), "c": c}      # its own colored line; not repeated as an alert

    rec = (plan or {}).get("sunset") or (plan or {}).get("noon") or {}
    p = rec.get("plan") or {}
    slots = p.get("slots") or []
    tonight = {"image": bool(p.get("will_image")),
               "slots": [{"dso": s["dso"], "at": datetime.fromisoformat(s["start"]).strftime("%H:%M")}
                         for s in slots][:4],
               "opens": opens, "closes": closes}

    # A hold does not carry on by itself, so it promises no next slot.
    upcoming = None if state in HOLD_STATES else next_slot(slots, now)
    level = "r" if any(x["c"] == "r" for x in (night, roof_out, disk or {"c": "g"})) else \
        "a" if alerts or "a" in (roof_out["c"], scope_out["c"], (disk or {}).get("c")) else "g"
    return {"ok": True, "t": now.strftime("%H:%M"), "level": level,
            "roof": roof_out, "scope": scope_out,
            "night": night, "now": live, "next": upcoming,
            "tonight": tonight, "disk": disk, "alerts": alerts[:3]}


# ----------------------------------------------------------------- the reads

def read_roof() -> Optional[dict]:
    try:
        days = sorted(d for d in os.listdir(NORTH_ARCHIVE) if re.match(r"\d{4}-\d{2}-\d{2}$", d))
        for day in reversed(days[-2:]):
            r = roof_from_names(day, os.listdir(NORTH_ARCHIVE / day))
            if r:
                return r
    except OSError:
        pass
    return None


def read_scope() -> str:
    try:
        from hardware_control.pwi4_client import PWI4
        return scope_from_status(PWI4().status())
    except Exception:  # noqa: BLE001 -- PWI4 not running is a normal daytime answer
        return "?"


_plan_cache = {"mtime": None, "rows": []}


def _planned(seq_path: Path) -> dict:
    """{(dso, filter): frames per pass} from tonight's sequence file, cached on its mtime."""
    try:
        mtime = seq_path.stat().st_mtime
        if _plan_cache["mtime"] != mtime:
            from nina_gen.nina_sequence_gen import read_back
            _plan_cache["rows"] = read_back(seq_path)
            _plan_cache["mtime"] = mtime
    except Exception:  # noqa: BLE001
        return {}
    return {(name.lower(), f): n for name, counts, _e in _plan_cache["rows"] for f, n in counts.items()}


def read_live(cfg: dict) -> Optional[dict]:
    """{"dso", "filter", "n", "of"} while frames are coming in, else None.

    n counts this target's frames in this filter in the newest frame's own
    LIGHT folder; "of" is one pass from the sequence file (passes repeat, so n
    can pass it)."""
    try:
        from cmd_processing.web_server import _live_imaging_target
        root = Path(cfg["nina"]["image_dir"])
        ticker = json.loads((root / "frame_ticker.json").read_text())
        dso = _live_imaging_target(ticker)
        if not dso:
            return None
        last = Path(ticker["last_frame"]["path"])
        filt = ticker["last_frame"].get("filter") or "?"
        n = sum(1 for f in last.parent.glob("*.fits") if "_%s_" % filt in f.name)
        of = _planned(Path(cfg["nina"]["sequence_output"])).get((dso, filt))
        return {"dso": dso, "filter": filt, "n": n, "of": of}
    except Exception:  # noqa: BLE001
        return None


def build(cfg: dict) -> dict:
    """Gather everything and summarize. Blocking: call it off the event loop."""
    from iris import client as conductor
    try:
        night_summary = json.loads(NIGHT_STATUS.read_text()).get("summary", "")
    except (OSError, ValueError):
        night_summary = ""
    try:
        drive = Path(cfg["nina"]["image_dir"]).drive or "C:"
        disk_gb = shutil.disk_usage(drive + "\\").free / 1024 ** 3
    except Exception:  # noqa: BLE001
        disk_gb = None
    now = datetime.now()
    # The night's date is yesterday before noon, as get_scheduler_state reads it,
    # so at 02:00 the display still shows the plan that started the night.
    night_of = (now - timedelta(hours=12)).strftime("%Y-%m-%d")
    return summarize(now, read_roof(), read_scope(), conductor.state(2.0),
                     conductor.get_plan(night_of, 3.0), night_summary, read_live(cfg), disk_gb)
