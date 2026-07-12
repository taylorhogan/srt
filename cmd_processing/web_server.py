"""FastAPI web server for the SRT chat interface.

Provides a WebSocket endpoint for real-time command/response interaction,
a REST endpoint for cross-process message injection, and static file serving
for the chat UI and response images.
"""

import asyncio
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from cmd_processing import message_bus, jobs

_logger = logging.getLogger(__name__)

app = FastAPI()

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.on_event("startup")
async def _on_startup():
    """Send a Pushover notification when the server is ready to accept connections."""
    try:
        from configs import config
        from utils.pushover import push_message
        cfg = config.data()
        version = cfg.get("version", {}).get("date", "unknown")
        await asyncio.to_thread(push_message, f"Iris online — v{version}")
    except Exception:
        _logger.exception("Failed to send startup Pushover notification")

# Images dir is set at startup via init()
_images_dir: Optional[str] = None


def _read_cpu_percent() -> Optional[float]:
    """Best-effort system CPU utilisation (%).

    Prefers psutil; falls back to a no-dependency Windows CIM query so the
    metric still works where psutil isn't installed. Returns None on failure.
    Blocking — call via asyncio.to_thread.
    """
    try:
        import psutil
        return psutil.cpu_percent(interval=0.3)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | "
             "Measure-Object -Property LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=8,
        )
        val = out.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


# Outside-temperature cache (Open-Meteo current_weather), refreshed every 5 min.
_outside_temp_f: Optional[float] = None
_outside_temp_fetched: float = 0.0
_OUTSIDE_TEMP_TTL = 300.0


def _read_outside_temp_f() -> Optional[float]:
    """Current outside temperature in °F via Open-Meteo (no key). Cached ~5 min.

    Blocking (uses requests) — call via asyncio.to_thread. Returns the last
    cached value on failure, or None if never fetched.
    """
    global _outside_temp_f, _outside_temp_fetched
    import time as _time
    if _outside_temp_f is not None and (_time.time() - _outside_temp_fetched) < _OUTSIDE_TEMP_TTL:
        return _outside_temp_f
    try:
        import requests
        from configs import config
        loc = config.data()["location"]
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current_weather": True, "timezone": "auto"},
            timeout=10,
        )
        r.raise_for_status()
        temp_c = float(r.json()["current_weather"]["temperature"])
        _outside_temp_f = round(temp_c * 9 / 5 + 32, 1)
        _outside_temp_fetched = _time.time()
    except Exception:
        _logger.exception("outside temperature fetch failed")
    return _outside_temp_f


def init(images_dir: str) -> None:
    global _images_dir
    _images_dir = images_dir
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=images_dir), name="images")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/chat.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = message_bus.subscribe()

    try:
        # Send the current jobs snapshot on connect so the client can rehydrate
        # every card (and its isolated log) after a fresh load or a reconnect.
        await websocket.send_json({"type": "history", "jobs": jobs.snapshot()})

        async def send_loop():
            while True:
                # Queues now carry fully-typed envelopes (job_event, etc.).
                envelope = await queue.get()
                await websocket.send_json(envelope)

        async def recv_loop():
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "command":
                    cmd_text = data["command"]
                    # Dispatch in background thread
                    threading.Thread(
                        target=_dispatch_command, args=(cmd_text,), daemon=True
                    ).start()

        await asyncio.gather(send_loop(), recv_loop())

    except WebSocketDisconnect:
        pass
    except Exception:
        _logger.exception("WebSocket error")
    finally:
        message_bus.unsubscribe(queue)


def _dispatch_command(cmd_text: str):
    """Run a command as a job through the social_server.do_command dispatch.

    A job is created and bound to this dispatch thread, so every message the
    command posts lands in that job's isolated log. Long commands hand off to a
    worker thread via ``jobs.spawn`` and own their own terminal transition; for
    plain synchronous commands we finalize here on return/exception.
    """
    cmd_text = cmd_text.strip()
    job_id = jobs.create(cmd_text)
    jobs.set_current_job(job_id)
    jobs.transition(job_id, jobs.RUNNING)
    try:
        from configs import config
        from cmd_processing import social_server, cmd_history

        cfg = config.data()
        super_users = cfg.get("Super Users", {})
        account = next(iter(super_users), "web_user")

        # !n — expand to the nth history entry and re-run it
        if cmd_text.startswith("!"):
            raw = cmd_text[1:]
            try:
                n = int(raw)
            except ValueError:
                message_bus.post_message(f"history: '{raw}' is not a valid number")
                jobs.finalize(job_id, jobs.ERROR)
                return
            resolved = cmd_history.get_nth(n)
            if resolved is None:
                message_bus.post_message(f"history: no entry #{n}")
                jobs.finalize(job_id, jobs.ERROR)
                return
            message_bus.post_message(f"!{n} → {resolved}")
            cmd_text = resolved

        cmd_history.record(cmd_text)
        social_server.do_command(f"@iris {cmd_text}", None, account)

        # If a worker thread adopted the job (jobs.spawn), it owns the terminal
        # state; otherwise this synchronous command is now complete.
        if not jobs.is_async(job_id):
            jobs.finalize(job_id, jobs.DONE)
    except Exception:
        _logger.exception("Command dispatch failed for: %s", cmd_text)
        message_bus.post_message(f"Error processing command: {cmd_text}")
        jobs.finalize(job_id, jobs.ERROR)
    finally:
        jobs.clear_current_job()


@app.post("/api/post")
async def api_post(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    audio_path: Optional[str] = Form(None),
    html: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
):
    """Cross-process message injection endpoint.

    Used by the scheduler process, _preview_worker, process-isolated command
    workers, and standalone scripts that cannot access the in-process message
    bus directly. A ``job_id`` (sent by process-isolated workers) routes the
    entry to that job's card; without one it lands in the system feed.
    """
    actual_image_path = None

    saved_filename = None
    if image and image.filename:
        # File was uploaded directly
        dest = Path(_images_dir) / image.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(image.file, f)
        actual_image_path = str(dest)
        saved_filename = image.filename
    elif image_path:
        actual_image_path = image_path

    message_bus.post_message(message, actual_image_path, html=html or None,
                             job_id=job_id or None, audio_path=audio_path or None)
    return {"ok": True, "filename": saved_filename}


@app.post("/api/rename_upload")
async def api_rename_upload(old_filename: str = Form(...), new_name: str = Form(...)):
    """Rename a previously uploaded file to a normalised DSO name."""
    normalized = new_name.strip().lower().replace(" ", "") + ".jpg"
    old_path = Path(_images_dir) / old_filename
    new_path = Path(_images_dir) / normalized
    if not old_path.exists():
        return JSONResponse(content={"ok": False, "error": "file not found"}, status_code=404)
    old_path.rename(new_path)
    return JSONResponse(content={"ok": True, "filename": normalized})


@app.get("/api/history")
async def api_history():
    return {"messages": message_bus.get_history()}


@app.get("/api/jobs")
async def api_jobs():
    """Return the current jobs snapshot (rehydration / debugging)."""
    return {"jobs": jobs.snapshot()}


# Cancelling an imaging run is not cooperative — nothing in doit_cmd checks the
# cancel flag, and stopping the night means real hardware action. The client
# must re-POST with ?confirm=1 after showing this to the user; confirming runs
# the full stop! emergency sequence.
_IMAGING_CANCEL_CONFIRM_MSG = (
    "EMERGENCY STOP the imaging run? This kills NINA, parks the scope, and "
    "closes the roof only if the scope confirms parked. The night ends here."
)


def _is_imaging_job(job: Optional[dict]) -> bool:
    words = ((job or {}).get("command") or "").split()
    return bool(words) and words[0].lower() == "image!!"


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str, confirm: bool = False):
    """Request cooperative cancellation of a running job.

    Sets a cancel flag the worker observes at its next checkpoint; the terminal
    CANCELLED event is emitted by the worker, not optimistically here.

    An imaging job (``image!!``) is the exception: no worker checkpoint exists,
    so cancelling it means an emergency stop. Without ``confirm`` the request
    is refused with ``requires_confirm`` so the client can ask the user; with
    ``confirm`` the cancel flag is set (labels the card CANCELLED when the run
    unwinds) and a real ``stop!`` command is dispatched, which owns the safe
    shutdown (park-gated roof close) on its own job card.
    """
    job = jobs.get_job(job_id)
    if _is_imaging_job(job) and job["status"] == jobs.RUNNING:
        if not confirm:
            return JSONResponse(content={
                "ok": False,
                "requires_confirm": True,
                "message": _IMAGING_CANCEL_CONFIRM_MSG,
            })
        ok = jobs.request_cancel(job_id)
        threading.Thread(target=_dispatch_command, args=("stop!",),
                         daemon=True).start()
        return JSONResponse(content={"ok": ok, "emergency_stop": True})
    ok = jobs.request_cancel(job_id)
    return JSONResponse(content={"ok": ok})


@app.post("/api/jobs/{job_id}/remove")
async def api_remove_job(job_id: str):
    """Remove a finished job card; broadcasts a 'removed' event to all clients."""
    ok = jobs.remove(job_id)
    return JSONResponse(content={"ok": ok})


@app.get("/api/ticker")
async def api_ticker():
    """Return current observatory status metrics for the header ticker."""
    try:
        from cmd_processing import super_user_commands as su
        from control import instructions

        mode = su.get_mode()
        safe = "Safe" if su.is_safe() else "Unsafe"
        imaging = su.get_imaging_state().value.replace("_", " ").title()
        sched = su.get_scheduler_state()

        sched_state = sched.get("state", "—")
        if isinstance(sched_state, str):
            sched_state = sched_state.replace("_", " ").title()

        tonight = sched.get("will image tonight", "—")
        if isinstance(tonight, bool):
            tonight = "Yes" if tonight else "No"

        # Read the live queue so reprioritisation is reflected immediately.
        try:
            dso = instructions.get_dso_object_tonight().get("dso", "—")
        except Exception:
            dso = sched.get("dso") or "—"

        metrics = [
            {"label": "Scheduler", "value": sched_state},
            {"label": "Target",    "value": str(dso)},
            {"label": "Mode",      "value": mode.title()},
            {"label": "Safety",    "value": safe},
            {"label": "State",     "value": imaging},
        ]

        # Append Pegasus (inside-observatory) environment data if available
        try:
            from hardware_control.pegasus import get_temperature_humidity
            env = await asyncio.to_thread(get_temperature_humidity)
            if env:
                metrics.append({"label": "Inside Temp", "value": f"{env['temperature_f']}°F"})
                metrics.append({"label": "Humidity",    "value": f"{env['humidity']}%"})
        except Exception:
            pass  # Unity not running or device unavailable

        # Outside temperature via Open-Meteo (no key needed)
        try:
            outside_f = await asyncio.to_thread(_read_outside_temp_f)
            if outside_f is not None:
                metrics.append({"label": "Outside Temp", "value": f"{outside_f}°F"})
        except Exception:
            pass

        # Moon phase
        try:
            import math
            from datetime import date
            from astral import moon
            phase_days = moon.phase(date.today())   # 0–29.53 days
            illumination = round(50 * (1 - math.cos(2 * math.pi * phase_days / 29.53)))
            direction = "+" if phase_days < 14.765 else "-"
            metrics.append({"label": "Moon", "value": f"{illumination}% {direction}"})
        except Exception:
            pass

        # Free space on the drive hosting NINA images
        try:
            import shutil
            from configs import config as _cfg
            image_drive = Path(_cfg.data()["nina"]["image_dir"]).drive or "C:"
            free_gb = shutil.disk_usage(image_drive + "\\").free / 1024 ** 3
            metrics.append({"label": f"{image_drive} Free", "value": f"{free_gb:.1f} GB"})
        except Exception:
            pass

        # System CPU utilisation
        try:
            cpu = await asyncio.to_thread(_read_cpu_percent)
            if cpu is not None:
                metrics.append({"label": "CPU", "value": f"{cpu:.0f}%"})
        except Exception:
            pass

        payload = {"ok": True, "metrics": metrics}
    except Exception:
        _logger.exception("api_ticker error")
        payload = {"ok": False, "metrics": []}
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.get("/api/imaging_ticker")
async def api_imaging_ticker():
    """Return live per-frame imaging stats for the second ticker bar."""
    try:
        from configs import config
        from datetime import datetime
        cfg = config.data()
        ticker_path = Path(cfg["nina"]["image_dir"]) / "frame_ticker.json"
        if not ticker_path.exists():
            return JSONResponse(
                content={"ok": True, "active": False, "frame_count": 0, "last_frame": None},
                headers={"Cache-Control": "no-store"},
            )
        with open(ticker_path) as f:
            data = json.load(f)
        # Treat as inactive if the heartbeat is stale (watcher died without calling stop)
        if data.get("active"):
            hb = data.get("last_heartbeat")
            if hb:
                try:
                    age = (datetime.utcnow() - datetime.fromisoformat(hb)).total_seconds()
                    if age > 120:
                        data["active"] = False
                except Exception:
                    pass
        remaining_seconds = None
        # Time left above the custom horizon for the active target. Gate on the
        # live `active` heartbeat: the instruction status lifecycle is
        # waiting→completed and never sets "in process", so the old in_proc gate
        # was always None and this stayed blank. RA/Dec come from the generated
        # NINA sequence (no Simbad call on every poll).
        if data.get("active"):
            try:
                import json as _json
                import datetime as _dt
                import pytz
                import astropy.units as _u
                from astropy.coordinates import SkyCoord as _SkyCoord
                from astroplan import FixedTarget as _FixedTarget
                from iris_astronomy import astro_dso_visibility
                from astropy.time import Time as _AstroTime

                cfg2 = config.data()
                dso = None
                seq_path = Path(cfg2["nina"]["sequence_output"])
                if seq_path.exists():
                    def _walk_seq(obj):
                        if isinstance(obj, dict):
                            if "NINA.Astrometry.InputCoordinates" in obj.get("$type", ""):
                                ra_h = obj["RAHours"] + obj["RAMinutes"] / 60 + obj["RASeconds"] / 3600
                                dec_abs = obj["DecDegrees"] + obj["DecMinutes"] / 60 + obj["DecSeconds"] / 3600
                                return ra_h, -dec_abs if obj.get("NegativeDec") else dec_abs
                            for v in obj.values():
                                r = _walk_seq(v)
                                if r:
                                    return r
                        elif isinstance(obj, list):
                            for item in obj:
                                r = _walk_seq(item)
                                if r:
                                    return r
                        return None
                    with open(seq_path) as f:
                        seq_data = _json.load(f)
                    coords = _walk_seq(seq_data)
                    if coords:
                        ra_h, dec_d = coords
                        sc = _SkyCoord(ra=ra_h * _u.hour, dec=dec_d * _u.deg)
                        dso = _FixedTarget(coord=sc, name=data.get("target") or "target")
                if dso:
                    finish = astro_dso_visibility.get_finish_time(dso, _AstroTime.now())
                    if finish is not None:
                        local_tz = pytz.timezone("America/New_York")
                        now_local = _dt.datetime.now(local_tz)
                        if finish.tzinfo is None:
                            finish = local_tz.localize(finish)
                        secs = int((finish - now_local).total_seconds())
                        if secs > 0:
                            remaining_seconds = secs
            except Exception:
                _logger.exception("remaining_seconds calculation failed")
        data["remaining_seconds"] = remaining_seconds

        return JSONResponse(
            content={"ok": True, **data},
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        _logger.exception("api_imaging_ticker error")
        return JSONResponse(
            content={"ok": False, "active": False, "frame_count": 0, "last_frame": None},
            headers={"Cache-Control": "no-store"},
        )


def _render_latest_image() -> dict:
    """Render the newest LIGHT FITS to a JPEG in the served images dir (blocking).

    On-demand fallback for `/api/latest_image`; the frame_watcher also writes
    the same artifact eagerly via fits_processing.imaging_artifacts.
    """
    from configs import config
    from fits_processing import imaging_artifacts as _ia

    if _images_dir is None:
        return {"ok": False}
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec = cfg["nina"]["arc_sec_per_pixel"]
    out = Path(_images_dir) / _ia.LATEST_IMAGE_NAME
    rendered = _ia.render_latest_image(image_dir, arcsec, out)
    if rendered is None:
        return {"ok": False}
    return {"ok": True, "image_url": f"/images/{_ia.LATEST_IMAGE_NAME}?t={int(rendered.stat().st_mtime)}"}


@app.get("/api/latest_image")
async def api_latest_image():
    """Lazily render the newest frame to a JPEG and return its served URL."""
    try:
        result = await asyncio.to_thread(_render_latest_image)
    except Exception:
        _logger.exception("api_latest_image error")
        result = {"ok": False}
    return JSONResponse(content=result, headers={"Cache-Control": "no-store"})


@app.get("/api/imaging_artifacts")
async def api_imaging_artifacts():
    """Return URLs for the pre-rendered latest-image + stats-plot artifacts.

    The frame_watcher rebuilds both whenever a new sub lands; this endpoint
    just reports what is currently on disk (no rendering).  Each URL carries
    the file's mtime as a cache buster so the UI sees fresh content.
    """
    from fits_processing import imaging_artifacts as _ia
    if _images_dir is None:
        return JSONResponse({"ok": False}, headers={"Cache-Control": "no-store"})
    out = {"ok": True}
    img = Path(_images_dir) / _ia.LATEST_IMAGE_NAME
    stats = Path(_images_dir) / _ia.STATS_PLOT_NAME
    if img.exists():
        out["image_url"] = f"/images/{_ia.LATEST_IMAGE_NAME}?t={int(img.stat().st_mtime)}"
    if stats.exists():
        out["stats_url"] = f"/images/{_ia.STATS_PLOT_NAME}?t={int(stats.stat().st_mtime)}"
    return JSONResponse(out, headers={"Cache-Control": "no-store"})
