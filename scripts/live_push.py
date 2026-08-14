#!/usr/bin/env python3
"""Copy readings to the web host that serves the lab site's live panel.

Deliberately not git: these are readings regenerated on a timer, not authored
content, and a commit per frame would grow the repo without bound. Caddy serves
/srv/iris-live at /live/.

Every file lands as a dotted .tmp and is renamed on the far side. Without that
the site eventually fetches a half-copied JPEG or, worse, a truncated JSON that
parses as valid but is missing fields.
"""
import os
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

HOST = "taylor@100.91.17.119"          # web host, over the tailnet
DEST = "/srv/iris-live"
KEY = str(Path.home() / ".ssh" / "id_ed25519_iris")

# Every option here bounds a different way the tailnet can fail, and all three
# are needed. ConnectTimeout only covers the INITIAL connect, so on its own it
# does nothing about the common case: a link that connects and then stalls
# mid-transfer, which ssh will otherwise wait on indefinitely. ServerAlive*
# bounds that established-but-dead state to ~15s.
#
# Both commands get the full set. They used to differ -- ssh was passed
# _SSH_OPTS[:2], which is just BatchMode, silently dropping its connect timeout
# -- and there is no reason for them to: it is the same host over the same link
# and it fails the same way.
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]

# Backstop for a child that ignores all of the above (a wedged ssh that never
# reaches its own timeouts). Chosen well under the 240s kill in
# live_skymap.bat/sky_monitor.bat so a stall surfaces as this script failing,
# not as the whole scheduled job being killed -- the second loses the log.
_PROC_TIMEOUT_S = 45


def push(pairs, host=HOST, dest=DEST, key=KEY):
    """Copy [(local_path, remote_name), ...]. Raises on failure.

    Raising is deliberate: the caller is a scheduled job whose exit code is the
    only thing anyone sees, and a push that silently half-succeeded would leave
    the site showing a star count next to last hour's picture.

    Failing FAST matters as much as failing loudly here. These jobs run every 5
    minutes, so a hiccup that makes one run fail in 30s is invisible -- the next
    run fixes it. The same hiccup that hangs until the 240s kill makes every run
    inside the outage fail, the chart goes stale, and the watchdog pages someone.
    That is what happened on 2026-08-14: three kills at 06:21, 07:06 and 07:16
    over one hour, and a stale-feed Pushover, on a link measured at 3ms and 0.28s
    round trip once anyone looked.
    """
    for src, name in pairs:
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(str(src))
        subprocess.run(["scp", "-i", key] + _SSH_OPTS
                       + [str(src), host + ":" + dest + "/." + name + ".tmp"],
                       check=True, timeout=_PROC_TIMEOUT_S)
        subprocess.run(["ssh", "-i", key] + _SSH_OPTS
                       + [host, "mv " + dest + "/." + name + ".tmp "
                          + dest + "/" + name],
                       check=True, timeout=_PROC_TIMEOUT_S)
    return len(pairs)
