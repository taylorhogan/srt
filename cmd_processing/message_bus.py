"""Thread-safe in-memory message bus for the SRT web chat interface.

Any thread (command handlers, scheduler, background workers) can call
``post_message()`` to broadcast a message to all connected WebSocket clients.
Messages are stored in a bounded deque so new clients can receive recent history.
"""

import asyncio
import os
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

_messages: deque = deque(maxlen=500)
_subscribers: set = set()
_lock = threading.Lock()
_initialized = False
_images_dir: Optional[str] = None
_counter = 0


def init(images_dir: str, max_history: int = 500) -> None:
    global _messages, _initialized, _images_dir
    _images_dir = images_dir
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    _messages = deque(maxlen=max_history)
    _initialized = True


def is_initialized() -> bool:
    return _initialized


def make_entry(text: str, image_url: Optional[str] = None, html: Optional[str] = None) -> dict:
    """Build a message/log entry dict (id + timestamp) without broadcasting."""
    global _counter
    with _lock:
        _counter += 1
        return {
            "id": _counter,
            "timestamp": time.time(),
            "text": text,
            "image_url": image_url,
            "html": html,
        }


def broadcast(envelope: dict) -> None:
    """Push a typed envelope (e.g. ``{"type": "job_event", ...}``) to all clients."""
    with _lock:
        for queue in list(_subscribers):
            try:
                queue.put_nowait(envelope)
            except asyncio.QueueFull:
                pass


def post_message(text: str, image_path: Optional[str] = None, html: Optional[str] = None) -> dict:
    image_url = None

    if image_path and _images_dir:
        src = Path(image_path)
        if src.is_file():
            global _counter
            with _lock:
                _counter += 1
                seq = _counter
            unique_name = f"{int(time.time())}_{seq}_{src.name}"
            dest = Path(_images_dir) / unique_name
            shutil.copy2(str(src), str(dest))
            image_url = f"/images/{unique_name}"

    entry = make_entry(text, image_url=image_url, html=html)

    # Resolve the active job (falls back to the pinned system feed) and route the
    # entry into that job's isolated log; the registry broadcasts the job event.
    from cmd_processing import jobs
    job_id = jobs.get_current_job() or jobs.SYSTEM_JOB_ID
    entry["job_id"] = job_id

    with _lock:
        _messages.append(entry)

    jobs.append_log(job_id, entry)
    return entry


def subscribe() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    with _lock:
        _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    with _lock:
        _subscribers.discard(queue)


def get_history() -> list[dict]:
    with _lock:
        return list(_messages)
