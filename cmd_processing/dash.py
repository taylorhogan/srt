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


# ----------------------------------------------------- the night's frames (2026-10-25)
#
# Pages 2-4 of the display: tonight's FWHM chart, the newest frame as a small
# picture, and last night's summary. All read frame_stats.json, which the
# frame watcher already keeps per target; nothing is measured here.

_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_")
FILTER_ORDER = ["L", "R", "G", "B", "Ha", "O-III", "S-II"]
MAX_POINTS = 150


def night_frames(rows_by_dso: dict, night: str) -> list:
    """[(local datetime, dso, filter, fwhm, stars)] of *night*'s frames, oldest first. Pure.

    *rows_by_dso* is {dso: frame_stats rows}. A frame belongs to the night of
    the session folder N.I.N.A wrote it to (.../<rig>/<night>/LIGHT/x.fits);
    its local time comes from the file name, which N.I.N.A stamps in local
    time (the rows' own "time" is UTC)."""
    out = []
    for dso, rows in rows_by_dso.items():
        for r in rows:
            if not isinstance(r, dict) or not r.get("fwhm_arcsec"):
                continue
            p = Path(str(r.get("path", "")))
            if p.parent.parent.name != night:
                continue
            m = _STAMP.search(p.name)
            if m:
                out.append((datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S"), dso,
                            r.get("filter") or "?", float(r["fwhm_arcsec"]), int(r.get("star_count") or 0)))
    return sorted(out)


def fwhm_series(frames: list) -> Optional[dict]:
    """The chart's data, compact for the ESP32. Pure.

    {"start": "HH:MM", "span": minutes, "f": [filters], "p": [[minute, fwhm x100,
    filter index], ...], "slots": [[minute, dso], ...], "med": x100,
    "last": [fwhm x100, filter]} -- at most MAX_POINTS points, thinned evenly."""
    if not frames:
        return None
    t0 = frames[0][0]
    filters = [f for f in FILTER_ORDER if any(x[2] == f for x in frames)]
    filters += sorted({x[2] for x in frames} - set(filters))
    step = max(1, -(-len(frames) // MAX_POINTS))
    pts = [[int((t - t0).total_seconds() // 60), int(round(fw * 100)), filters.index(f)]
           for t, _d, f, fw, _n in frames[::step]]
    slots, prev = [], None
    for t, d, *_ in frames:
        if d != prev:
            slots.append([int((t - t0).total_seconds() // 60), d])
            prev = d
    fws = sorted(x[3] for x in frames)
    return {"start": t0.strftime("%H:%M"),
            "span": max(60, int((frames[-1][0] - t0).total_seconds() // 60) + 1),
            "f": filters, "p": pts, "slots": slots,
            "med": int(round(fws[len(fws) // 2] * 100)),
            "last": [int(round(frames[-1][3] * 100)), frames[-1][2]]}


def imaging_filters(frames: list, dso: str) -> list:
    """[{"f", "n", "med", "last"}] for *dso*'s frames tonight, in filter order. Pure.

    The big-type imaging pages (2026-10-26, owner: "simpler information while
    imaging"): page 2 is the counts, page 3 the median and last FWHM."""
    mine = [x for x in frames if x[1] == dso]
    by = {}
    for t, _d, f, fw, _n in mine:
        by.setdefault(f, []).append((t, fw))
    order = [f for f in FILTER_ORDER if f in by] + sorted(set(by) - set(FILTER_ORDER))
    out = []
    for f in order:
        fws = sorted(fw for _t, fw in by[f])
        out.append({"f": f, "n": len(by[f]), "med": round(fws[len(fws) // 2], 2),
                    "last": round(max(by[f])[1], 2)})
    return out


def night_summary(frames: list, night: str) -> Optional[dict]:
    """Per target: frames per filter and median FWHM; plus the night's sharpest frame. Pure."""
    if not frames:
        return None
    targets = []
    for dso in dict.fromkeys(x[1] for x in frames):             # running order
        mine = [x for x in frames if x[1] == dso]
        counts = {}
        for x in mine:
            counts[x[2]] = counts.get(x[2], 0) + 1
        fws = sorted(x[3] for x in mine)
        targets.append({"dso": dso, "n": counts, "fwhm": round(fws[len(fws) // 2], 2)})
    best = min(frames, key=lambda x: x[3])
    return {"night": night, "total": len(frames), "targets": targets[:5],
            "best": {"fwhm": round(best[3], 2), "dso": best[1], "filter": best[2],
                     "at": best[0].strftime("%H:%M")}}


_stats_cache: dict = {}


def _rows(stats: Path) -> list:
    """frame_stats.json rows, cached on the file's mtime."""
    try:
        mtime = stats.stat().st_mtime
    except OSError:
        return []
    hit = _stats_cache.get(stats)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        d = json.loads(stats.read_text())
        rows = list(d.values()) if isinstance(d, dict) else list(d)
    except (OSError, ValueError):
        rows = []
    _stats_cache[stats] = (mtime, rows)
    return rows


def read_night(cfg: dict, now: datetime):
    """(night, frames) for the night under way, or the last night that has frames."""
    root = Path(cfg["nina"]["image_dir"])
    for back in (0, 1, 2):
        night = (now - timedelta(hours=12) - timedelta(days=back)).strftime("%Y-%m-%d")
        dsos = {p.parent.parent.parent.name for p in root.glob("*/*/%s/LIGHT" % night)}
        if dsos:
            frames = night_frames({d: _rows(root / d / "frame_stats.json") for d in dsos}, night)
            if frames:
                return night, frames
    return None, []


# The newest frame as a 240 x 160 picture: the centre third of the field
# (targets are centred), stretched on its own sky. Baseline JPEG, which is
# what the ESP32's decoder reads. Rendered once per new frame.
THUMB_W, THUMB_H = 240, 160
_thumb = {"path": None, "jpeg": None}


def thumbnail(fits_path: Path) -> Optional[bytes]:
    if _thumb["path"] == fits_path:
        return _thumb["jpeg"]
    try:
        import io
        import numpy as np
        from astropy.io import fits
        from PIL import Image
        with fits.open(fits_path, memmap=True, do_not_scale_image_data=True) as h:
            hdr, data = h[0].header, h[0].data
            ny, nx = data.shape
            cw, ch = nx // 3, ny // 3
            x0, y0 = (nx - cw) // 2, (ny - ch) // 2
            step = max(1, cw // (THUMB_W * 2))
            a = (np.asarray(data[y0:y0 + ch:step, x0:x0 + cw:step], np.float32)
                 * float(hdr.get("BSCALE", 1.0)) + float(hdr.get("BZERO", 0.0)))
        med = float(np.median(a))
        sig = 1.4826 * float(np.median(np.abs(a - med))) or 1.0
        v = np.clip((a - (med - 1.5 * sig)) / (40.0 * sig), 0, 1)
        v = np.arcsinh(v * 12.0) / np.arcsinh(12.0)
        im = Image.fromarray((v * 255).astype(np.uint8)).resize((THUMB_W, THUMB_H), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=80, progressive=False)
        _thumb.update(path=fits_path, jpeg=buf.getvalue())
        return _thumb["jpeg"]
    except Exception:  # noqa: BLE001 -- no picture is a normal answer
        _logger.debug("dash thumbnail failed for %s", fits_path, exc_info=True)
        return None


def newest_frame(cfg: dict) -> Optional[dict]:
    """{"path", "dso", "filter", "fwhm", "id"} of the newest LIGHT frame, or None."""
    try:
        root = Path(cfg["nina"]["image_dir"])
        last = json.loads((root / "frame_ticker.json").read_text()).get("last_frame") or {}
        p = Path(last["path"])
        if not p.exists():
            return None
        dso = p.relative_to(root).parts[0].lower()
        m = _STAMP.search(p.name)
        return {"path": p, "dso": dso, "filter": last.get("filter") or "?",
                "fwhm": last.get("fwhm_arcsec"),
                "at": m.group(1)[11:16].replace("-", ":") if m else "",
                "id": p.stem[-24:]}
    except Exception:  # noqa: BLE001
        return None


# The sky camera's clarity (2026-10-25): plate-solve matches over the catalogue
# stars in its field, as a fraction of a clear sky -- the dusk check's reading
# (sentry/dusk_check, clear sky = 0.75), but from the newest 5-minute frame
# only. Moonlight only ever lowers it, so a moonlit low reading is not cloud.
SKY_LOG = ROOT / "local" / "sky_log.jsonl"
SKY_MAX_AGE_MIN = 15


def sky_label(row: Optional[dict], now_utc: datetime, in_field: Optional[int],
              clear: float, moonlit: bool) -> Optional[dict]:
    """{"v", "c", "moon"} for the status page, or None when there is nothing to say. Pure.

    g at 65% of a clear sky or more, a below it, x (grey) under 20%: cloud is
    worth a look, never an alarm. Under a bright moon the reading cannot tell
    cloud from moonlight (10-24, a clear night, read 35-67%), so it is shown
    in neutral w with the moon named rather than amber half of every month."""
    if not row:
        return None
    try:
        t = datetime.fromisoformat(row.get("captured") or row["generated"])
    except (KeyError, ValueError):
        return None
    if t.tzinfo is None:
        from datetime import timezone
        t = t.replace(tzinfo=timezone.utc)
    if (now_utc - t).total_seconds() > SKY_MAX_AGE_MIN * 60:
        return {"v": "no sky cam", "c": "x", "moon": False}
    if not row.get("night"):
        return {"v": "day", "c": "x", "moon": False}
    if row.get("solve_matches") is None or not in_field:
        return {"v": "?", "c": "a", "moon": moonlit}
    pct = int(round(100 * row["solve_matches"] / in_field / clear))
    c = "w" if moonlit else ("g" if pct >= 65 else ("x" if pct < 20 else "a"))
    return {"v": "%d%%" % pct, "c": c, "moon": moonlit}


_sky_inputs = {"key": None, "val": (None, 0.75, False)}


def read_sky() -> Optional[dict]:
    """sky_label() for the newest sky-camera frame.

    The catalogue count and the moon cost astropy work, so they are worked out
    once per frame; the label (and its age check) is recomputed every call."""
    from datetime import timezone
    try:
        with open(SKY_LOG, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            row = json.loads(fh.read().decode("utf-8", "replace").strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
    key = row.get("generated")
    if _sky_inputs["key"] != key:
        in_field, clear, moonlit = None, 0.75, False
        try:
            if row.get("night") and row.get("solve_matches") is not None:
                from sentry import dusk_check, plate_solve
                from scripts.sky_transparency_report import in_field as _in_field
                st = dusk_check._settings()
                clear = float(st["dusk_clear_transparency"])
                t = datetime.fromisoformat(row.get("captured") or row["generated"])
                sol = plate_solve.load()
                in_field = _in_field(sol, t) if sol is not None else None
                malt, mill = dusk_check.moon_at(t)
                moonlit = bool(malt > 0 and mill >= float(st["dusk_moon_max_illum"]))
        except Exception:  # noqa: BLE001 -- the sky line is optional
            _logger.debug("dash sky reading failed", exc_info=True)
        _sky_inputs.update(key=key, val=(in_field, clear, moonlit))
    in_field, clear, moonlit = _sky_inputs["val"]
    return sky_label(row, datetime.now(timezone.utc), in_field, clear, moonlit)


def build(cfg: dict) -> dict:
    """Gather everything and summarize. Blocking: call it off the event loop."""
    from iris import client as conductor
    try:
        plan_text = json.loads(NIGHT_STATUS.read_text()).get("summary", "")
    except (OSError, ValueError):
        plan_text = ""
    try:
        drive = Path(cfg["nina"]["image_dir"]).drive or "C:"
        disk_gb = shutil.disk_usage(drive + "\\").free / 1024 ** 3
    except Exception:  # noqa: BLE001
        disk_gb = None
    now = datetime.now()
    # The night's date is yesterday before noon, as get_scheduler_state reads it,
    # so at 02:00 the display still shows the plan that started the night.
    night_of = (now - timedelta(hours=12)).strftime("%Y-%m-%d")
    live = read_live(cfg)
    out = summarize(now, read_roof(), read_scope(), conductor.state(2.0),
                    conductor.get_plan(night_of, 3.0), plan_text, live, disk_gb)
    night, frames = read_night(cfg, now)
    out["fwhm"] = fwhm_series(frames)
    # While imaging, the display switches to three big-type pages: the time
    # and temperature, then the target's counts per filter, then its median
    # and last FWHM per filter (owner, 2026-10-26). "sec" lets the display
    # tick its own clock between polls.
    out["sec"] = now.hour * 3600 + now.minute * 60 + now.second
    out["im"] = ({"dso": live["dso"], "filters": imaging_filters(frames, live["dso"])}
                 if live else None)
    try:
        from cmd_processing.web_server import _read_outside_temp_f
        out["temp_f"] = _read_outside_temp_f()
    except Exception:  # noqa: BLE001 -- no temperature is a normal answer
        out["temp_f"] = None
    # The summary is for the morning after: shown while nothing is imaging.
    out["summary"] = night_summary(frames, night) if live is None else None
    f = newest_frame(cfg)
    out["img"] = ({k: f[k] for k in ("id", "dso", "filter", "fwhm", "at")} if f else None)
    out["sky"] = read_sky()
    out["fw"] = firmware_info()
    return out



# Over-the-air firmware (2026-10-26): scripts/core2_publish.py compiles and
# signs it; the Core2 compares "fw"["v"] with its own build and fetches
# ?part=firmware when they differ, booting it only if the HMAC matches.
FW_MANIFEST = ROOT / "local" / "core2_fw" / "manifest.json"
FW_BINARY = ROOT / "local" / "core2_fw" / "firmware.bin"


def firmware_info(manifest: Path = FW_MANIFEST) -> Optional[dict]:
    """{"v", "size", "hmac"} of the published firmware, or None when nothing is published."""
    try:
        m = json.loads(manifest.read_text())
        return {k: m[k] for k in ("v", "size", "hmac")}
    except (OSError, ValueError, KeyError):
        return None


# The build the Core2 last said it runs (it adds ?fw=<version> to every poll).
RUNNING = {"v": None, "at": None}


def note_running(version: str, now: Optional[datetime] = None) -> None:
    """Remember the display's reported build; anything odd is cut short, not trusted."""
    import re
    v = re.sub(r"[^A-Za-z0-9._+-]", "", str(version))[:48]
    if v:
        RUNNING.update(v=v, at=now or datetime.now())


def firmware_status(published: Optional[dict], running: dict, now: datetime) -> str:
    """The `core2 status` line: published vs running, and how long since it was heard. Pure."""
    pub = published["v"] if published else None
    run, at = running.get("v"), running.get("at")
    if run is None:
        heard = "not heard from since the chat server started"
    else:
        mins = int((now - at).total_seconds() // 60)
        heard = "running %s (heard %s)" % (run, "just now" if mins < 1 else "%d min ago" % mins)
    if pub is None:
        return "core2: nothing published; %s" % heard
    state = ("up to date" if run == pub else "update pending" if run else "unknown")
    return "core2: published %s; %s -- %s" % (pub, heard, state)


def firmware(binary: Path = FW_BINARY) -> Optional[bytes]:
    try:
        return binary.read_bytes()
    except OSError:
        return None


def image(cfg: dict) -> Optional[bytes]:
    """The newest frame's thumbnail JPEG (GET /api/dash?part=image)."""
    f = newest_frame(cfg)
    return thumbnail(f["path"]) if f else None
