"""Pushover notifications.

Delivery is *verified*, not assumed: Pushover answers every POST with a JSON
body carrying ``status`` (1 = accepted), and a wrong token, a malformed
priority-2 request, or a monthly-quota overrun all come back as a perfectly
healthy-looking HTTP response with ``status: 0``. Ignoring the body means a
silently undelivered alert, which for the roof-stall emergency is the whole
notification path failing quietly.

So ``push_message`` returns True only when Pushover confirms acceptance, and
callers on the safety path are expected to check it. It never raises — an
alerting failure must not take down the caller that is trying to alert.
"""

import http.client        # `import http` alone does NOT bind http.client
import json
import os
import random
import sys
import time
import urllib.parse       # likewise urllib.parse

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from configs import config
import utils.rate_limit as rate_limit
from utils import utils

cfg = config.data()

client_id = f'subscribe-{random.randint(0, 100)}'
token = cfg['pushover']['token']
user = cfg['pushover']['user']

_logger = utils.set_logger()

# Emergencies are retried; routine alerts are not worth blocking a caller for.
_EMERGENCY_ATTEMPTS = 3
_EMERGENCY_BACKOFF_S = (2.0, 4.0)
_TIMEOUT_S = 10.0

api_limiter = rate_limit.RateLimiter(max_calls=6, period=60.0)


def _say(message):
    """Log, and echo to the console without ever raising.

    The observatory's alert text carries emoji and em-dashes, and stdout here is
    cp1252 — printing one directly is an unhandled UnicodeEncodeError that would
    propagate out of the notifier. Encode through the console's own codec with
    backslash escapes so an unprintable character degrades to mojibake rather
    than to a crash inside the alerting path.
    """
    _logger.info(message)
    try:
        enc = sys.stdout.encoding or "utf-8"
        print(message.encode(enc, errors="backslashreplace").decode(enc, errors="replace"))
    except Exception:  # noqa: BLE001 — console output must never break alerting
        pass


def _accepted(status_code, body):
    """True when Pushover's response body confirms it accepted the message."""
    if status_code != 200:
        return False, f"HTTP {status_code}: {body[:200]}"
    try:
        payload = json.loads(body)
    except ValueError:
        return False, f"unparseable response: {body[:200]}"
    if payload.get("status") == 1:
        return True, ""
    return False, f"pushover rejected: {body[:200]}"


def _post_once(params):
    """One POST to the messages endpoint. Returns (accepted, detail)."""
    conn = None
    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443", timeout=_TIMEOUT_S)
        conn.request("POST", "/1/messages.json",
                     urllib.parse.urlencode(params),
                     {"Content-type": "application/x-www-form-urlencoded"})
        response = conn.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        return _accepted(response.status, body)
    except Exception as e:  # noqa: BLE001 — reported to the caller as a failure
        return False, f"{type(e).__name__}: {e}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def push_message(message, image=None, priority=0):
    """Send a Pushover notification. Returns True only if Pushover accepted it.

    priority=2 is a Pushover emergency: it repeats on the phone until
    acknowledged, bypasses the rate limiter — an emergency (e.g. roof motor
    stall) must never be dropped because routine alerts used the budget — and is
    retried, because a single transient network failure losing the one alert
    that matters is the failure mode this whole function exists to prevent.
    """
    if priority < 2 and not api_limiter():
        _say("Rate limit exceeded - ignoring this call")
        return False

    if image is not None:
        # Carry the priority through: attaching an image used to silently
        # downgrade an emergency to a normal notification.
        return push_message_with_picture(message, image, priority=priority)

    params = {
        "token": token,
        "user": user,
        "message": message,
    }
    if priority:
        params["priority"] = priority
        if priority == 2:
            # Pushover requires retry/expire for emergency priority:
            # re-alert every 60s for up to an hour until acknowledged.
            params["retry"] = 60
            params["expire"] = 3600

    attempts = _EMERGENCY_ATTEMPTS if priority == 2 else 1
    detail = ""
    for attempt in range(1, attempts + 1):
        accepted, detail = _post_once(params)
        if accepted:
            return True
        _logger.warning("pushover attempt %d/%d failed: %s", attempt, attempts, detail)
        if attempt < attempts:
            time.sleep(_EMERGENCY_BACKOFF_S[min(attempt - 1, len(_EMERGENCY_BACKOFF_S) - 1)])

    _logger.error("pushover DELIVERY FAILED (priority=%s) after %d attempt(s): %s | message: %s",
                  priority, attempts, detail, message)
    return False


def push_message_with_picture(message, image, priority=0):
    """Send a notification with an image attached. Returns True if accepted."""
    try:
        with open(image, "rb") as fh:
            data = {
                "token": token,
                "user": user,
                "message": message,  # Required: Your message text (up to 1024 chars)
                "title": "News From Iris",  # Optional: Notification title (up to 250 chars)
            }
            if priority:
                data["priority"] = priority
                if priority == 2:
                    data["retry"] = 60
                    data["expire"] = 3600
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files={"attachment": ("image.jpg", fh, "image/jpeg")},
                timeout=_TIMEOUT_S,
            )
        accepted, detail = _accepted(response.status_code, response.text)
        if not accepted:
            _logger.error("pushover image DELIVERY FAILED (priority=%s): %s | message: %s",
                          priority, detail, message)
        return accepted
    except Exception as e:  # noqa: BLE001 — alerting must not raise
        _logger.exception("pushover image send failed: %s", e)
        return False


def main():
    for i in range(10):
        print (i)
        push_message ("hi", "./base_images/inside.jpg")


if __name__ == '__main__':
    main()
