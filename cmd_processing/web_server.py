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
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from cmd_processing import message_bus

_logger = logging.getLogger(__name__)

app = FastAPI()

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

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
                    # Post the user's command as a message so it shows in chat
                    message_bus.post_message(f"> {cmd_text}")
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

    message_bus.post_message(message, actual_image_path)
    return {"ok": True}


@app.get("/api/history")
async def api_history():
    return {"messages": message_bus.get_history()}
