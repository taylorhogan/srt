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


def _stage_media(src_path: str) -> Optional[str]:
    """Copy a file into the served images dir under a unique name.

    Returns the ``/images/<name>`` URL, or None if there's nowhere to serve
    from or the source doesn't exist. Used for both image and audio attachments
    (StaticFiles serves whatever mime type the extension implies).
    """
    if not (src_path and _images_dir):
        return None
    src = Path(src_path)
    if not src.is_file():
        return None
    global _counter
    with _lock:
        _counter += 1
        seq = _counter
    unique_name = f"{int(time.time())}_{seq}_{src.name}"
    dest = Path(_images_dir) / unique_name
    shutil.copy2(str(src), str(dest))
    return f"/images/{unique_name}"


def post_message(text: str, image_path: Optional[str] = None, html: Optional[str] = None,
                 job_id: Optional[str] = None, audio_path: Optional[str] = None) -> dict:
    image_url = _stage_media(image_path) if image_path else None

    # An audio clip is delivered as an inline <audio> player via the html field,
    # followed by explicit download links for the clip (and the attached image,
    # if any) so the raw files can be saved for offline analysis.
    if audio_path and html is None:
        audio_url = _stage_media(audio_path)
        if audio_url:
            links = f' <a href="{audio_url}" download>⬇ wav</a>'
            if image_url:
                links += f' <a href="{image_url}" download>⬇ png</a>'
            html = (f'<audio controls preload="none" src="{audio_url}"></audio>'
                    f'{links}')

    entry = make_entry(text, image_url=image_url, html=html)

    # Resolve the target job and route the entry into its isolated log; the
    # registry broadcasts the job event. An explicit job_id (e.g. forwarded from
    # a process-isolated worker via /api/post) wins; otherwise fall back to the
    # calling thread's current job, then the pinned system feed.
    from cmd_processing import jobs
    job_id = job_id or jobs.get_current_job() or jobs.SYSTEM_JOB_ID
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
