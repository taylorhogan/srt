"""Post messages and images to the web chat from a remote machine.

The web chat's FastAPI server (cmd_processing/web_server.py) exposes
POST /api/post, which the same-machine processes already use via
post_social_message's HTTP fallback. This client is the cross-machine
version: an offline analysis node (the DGX Spark) posts its results into
the one chat everyone reads, over the Tailnet, instead of running a
second chat instance.

Images are sent as real multipart uploads (the endpoint saves them into
its upload dir), so the caller's filesystem never needs to be visible to
the chat server.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

_logger = logging.getLogger(__name__)


def _base_url() -> str:
    cfg = config.data().get("web_chat", {})
    url = cfg.get("remote_url")
    if not url:
        # Same-machine fallback, matching post_social_message's behaviour.
        url = f"http://localhost:{cfg.get('port', 8095)}"
    return url.rstrip("/")


def post_to_webchat(
    message: str,
    image_path: Optional[Path] = None,
    timeout: float = 30.0,
) -> bool:
    """Post ``message`` (and optionally one image) to the web chat.

    Returns True on success. Never raises — a chat outage must not kill the
    analysis job whose results are being announced; failures are logged.
    """
    url = f"{_base_url()}/api/post"
    # job_id must be explicit: the chat UI renders job cards (the system feed
    # is the pinned "Observatory" card), and un-tagged posts fall into the
    # legacy /api/history feed on older server builds — stored but invisible.
    data = {"message": message, "job_id": "system"}
    try:
        if image_path is not None:
            with open(image_path, "rb") as f:
                resp = requests.post(
                    url, data=data,
                    files={"image": (Path(image_path).name, f, "image/png")},
                    timeout=timeout,
                )
        else:
            resp = requests.post(url, data=data, timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception:
        _logger.exception("Failed to post to web chat at %s", url)
        return False
