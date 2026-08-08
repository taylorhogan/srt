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

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def push(pairs, host=HOST, dest=DEST, key=KEY):
    """Copy [(local_path, remote_name), ...]. Raises on failure.

    Raising is deliberate: the caller is a scheduled job whose exit code is the
    only thing anyone sees, and a push that silently half-succeeded would leave
    the site showing a star count next to last hour's picture.
    """
    for src, name in pairs:
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(str(src))
        subprocess.run(["scp", "-i", key] + _SSH_OPTS
                       + [str(src), host + ":" + dest + "/." + name + ".tmp"],
                       check=True)
        subprocess.run(["ssh", "-i", key] + _SSH_OPTS[:2]
                       + [host, "mv " + dest + "/." + name + ".tmp "
                          + dest + "/" + name], check=True)
    return len(pairs)
