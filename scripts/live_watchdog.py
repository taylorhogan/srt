#!/usr/bin/env python3
"""Alert when the live sky chart stops being updated.

Runs as its own scheduled task, separately from the generator, and that
separation is the whole point: on 2026-08-07 the generator hung and every
subsequent run reported success, so nothing inside that path could have
noticed. A watchdog that shares a process with the thing it watches is not a
watchdog. This one asks the public URL, so it also proves the tunnel, Caddy and
the box are all still answering -- not merely that a file exists locally.

Sends at most one Pushover per stale episode, and one when it recovers. A
monitor that repeats every cycle gets muted, and a muted monitor is worse than
none because it looks like coverage.

Usage:  python scripts/live_watchdog.py [--check-only]
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

URL = "https://irisscience.org/live/status.json"
STALE_MINUTES = 20.0          # generator runs every 5; 20 tolerates a few misses
STATE = Path(__file__).resolve().parent.parent / "local" / "live_watchdog.json"


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"alerted": False}


def _save(d: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(d))
    except Exception:
        pass


def _push(msg: str) -> None:
    try:
        from utils import pushover
        pushover.push_message(msg)
        print("pushover:", msg)
    except Exception as exc:      # never let alerting be the thing that fails
        print("pushover failed:", exc)


def main() -> None:
    age = None
    problem = None
    try:
        # Cloudflare 403s the default python-urllib user-agent, so the check
        # must identify itself like an ordinary client or the watchdog reports
        # an outage that is really its own request being blocked.
        req = urllib.request.Request(URL, headers={
            "Cache-Control": "no-cache",
            "User-Agent": "iris-live-watchdog/1.0 (+https://irisscience.org)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        gen = datetime.fromisoformat(data["generated"])
        age = (datetime.now(timezone.utc) - gen).total_seconds() / 60.0
        if age > STALE_MINUTES:
            problem = ("live sky feed is stale: last chart %.0f minutes ago "
                       "(target was %s)" % (age, data.get("target")))
    except Exception as exc:
        # Unreachable is its own failure and worth alerting on: it means the
        # tunnel, Caddy or the box is down, not just the generator.
        problem = "live sky feed unreachable: %s" % exc

    st = _state()
    if problem:
        print("STALE:", problem)
        if not st.get("alerted") and "--check-only" not in sys.argv:
            _push("Iris: " + problem)
            _save({"alerted": True,
                   "since": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    else:
        print("ok: chart is %.1f minutes old" % age)
        if st.get("alerted") and "--check-only" not in sys.argv:
            _push("Iris: live sky feed recovered (chart %.0f min old)" % age)
            _save({"alerted": False})


if __name__ == "__main__":
    main()
