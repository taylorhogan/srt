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


GEN_ERR = Path(__file__).resolve().parent.parent / "local" / "live_skymap.err"
TAILSCALE = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         "Tailscale", "tailscale.exe")


def diagnose(err_text: str, tailscale_state) -> str:
    """Why the chart is stale and what to do, in plain words. Pure.

    *err_text*: the generator's last stderr (local/live_skymap.err).
    *tailscale_state*: Tailscale's BackendState ("Running", "NeedsLogin",
    "Stopped", ...) or None if it could not be asked.

    2026-10-05: "live sky feed is stale: last chart 21 minutes ago (target was
    ngc7320)" reached the owner's phone and meant nothing to him. The real
    story was that Tailscale had logged out (node key expired), so every
    upload to the web host failed -- that is what this now says.
    """
    upload_failed = ("live_push" in err_text or "scp" in err_text) and (
        "exit status 255" in err_text or "timed out" in err_text.lower())
    if tailscale_state and tailscale_state != "Running":
        return ("Tailscale on iris-pc is %s, so iris-pc cannot upload to the web server. "
                "Fix: log in from the Tailscale tray icon on iris-pc, then disable key expiry "
                "for iris-pc in the Tailscale admin console." % (
                    "logged out" if tailscale_state == "NeedsLogin" else tailscale_state.lower()))
    if upload_failed:
        return ("iris-pc renders the chart but cannot upload it to the web server "
                "(scp over Tailscale failed). Check that the web server (iris) is up and "
                "reachable on the tailnet.")
    if err_text.strip():
        last = err_text.strip().splitlines()[-1][:160]
        return "the chart generator on iris-pc is failing: %s (see local/live_skymap.err)" % last
    return "the cause is not visible from iris-pc (check the IrisLiveSkymap scheduled task)"


def _tailscale_state():
    try:
        import subprocess
        out = subprocess.run([TAILSCALE, "status", "--json"], capture_output=True,
                             text=True, timeout=20).stdout
        return json.loads(out).get("BackendState")
    except Exception:
        return None


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
            try:
                err = GEN_ERR.read_text(errors="replace")
            except OSError:
                err = ""
            problem = ("the website's live sky panel stopped updating %.0f minutes ago -- %s"
                       % (age, diagnose(err, _tailscale_state())))
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
            _push("Iris: the website's live sky panel is updating again (chart %.0f min old)" % age)
            _save({"alerted": False})


if __name__ == "__main__":
    main()
