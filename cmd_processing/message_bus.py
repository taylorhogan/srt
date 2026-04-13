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


def post_message(text: str, image_path: Optional[str] = None, html: Optional[str] = None) -> dict:
    global _counter
    image_url = None

    if image_path and _images_dir:
        src = Path(image_path)
        if src.is_file():
            _counter += 1
            unique_name = f"{int(time.time())}_{_counter}_{src.name}"
            dest = Path(_images_dir) / unique_name
            shutil.copy2(str(src), str(dest))
            image_url = f"/images/{unique_name}"

    with _lock:
        _counter += 1
        msg = {
            "id": _counter,
            "timestamp": time.time(),
            "text": text,
            "image_url": image_url,
            "html": html,
        }
        _messages.append(msg)

        for queue in list(_subscribers):
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    return msg


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
