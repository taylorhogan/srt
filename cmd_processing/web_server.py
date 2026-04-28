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

from cmd_processing import message_bus

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
        # Send message history on connect
        history = message_bus.get_history()
        await websocket.send_json({"type": "history", "messages": history})

        async def send_loop():
            while True:
                msg = await queue.get()
                await websocket.send_json({"type": "message", "data": msg})

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
    """Run a command through the existing social_server.do_command dispatch."""
    try:
        from configs import config
        from cmd_processing import social_server

        cfg = config.data()
        # Use the first super user account for web commands
        super_users = cfg.get("Super Users", {})
        account = next(iter(super_users), "web_user")
        sentence = f"@iris {cmd_text}"
        social_server.do_command(sentence, None, account)
    except Exception:
        _logger.exception("Command dispatch failed for: %s", cmd_text)
        message_bus.post_message(f"Error processing command: {cmd_text}")


@app.post("/api/post")
async def api_post(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    html: Optional[str] = Form(None),
):
    """Cross-process message injection endpoint.

    Used by the scheduler process, _preview_worker, and standalone scripts
    that cannot access the in-process message bus directly.
    """
    actual_image_path = None

    if image and image.filename:
        # File was uploaded directly
        dest = Path(_images_dir) / image.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(image.file, f)
        actual_image_path = str(dest)
    elif image_path:
        actual_image_path = image_path

    message_bus.post_message(message, actual_image_path, html=html or None)
    return {"ok": True}


@app.get("/api/history")
async def api_history():
    return {"messages": message_bus.get_history()}


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

        # Append Pegasus environment data if available
        try:
            from hardware_control.pegasus import get_temperature_humidity
            env = await asyncio.to_thread(get_temperature_humidity)
            if env:
                metrics.append({"label": "Temp",     "value": f"{env['temperature_f']}°F"})
                metrics.append({"label": "Humidity", "value": f"{env['humidity']}%"})
        except Exception:
            pass  # Unity not running or device unavailable

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
