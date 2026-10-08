"""Publish the lab site: push its repo, then make the web host pull NOW.

    python scripts/site_push.py

GitHub stays the site's source of truth and history (and taylorhogan.github.io
its mirror); the web host serves /srv/iris-site, which iris-site-pull.timer
refreshes every 5 minutes. This removes the wait: after the push it starts
that same pull service over the tailnet and reports the commit the host is
now serving, so "pushed" means "live".
"""
import os
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from scripts import live_push

SITE = Path.home() / "Documents" / "development" / "taylorhogan.github.io"
PULL_UNIT = "iris-site-pull.service"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(SITE)] + list(args), check=True,
                          capture_output=True, text=True, timeout=60).stdout.strip()


def main() -> int:
    local = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("site has uncommitted changes; commit first:\n" + dirty)
        return 2
    subprocess.run(["git", "-C", str(SITE), "push", "-q", "origin", "main"],
                   check=True, timeout=60)
    print("pushed", local)
    out = subprocess.run(
        ["ssh", "-i", live_push.KEY] + live_push._SSH_OPTS
        # The checkout is owned by `deploy`; reading it as `taylor` needs the
        # safe.directory override (read-only, so no sudo for the rev-parse).
        + [live_push.HOST, "sudo -n systemctl start %s && git -c safe.directory=/srv/iris-site "
           "-C /srv/iris-site rev-parse --short HEAD" % PULL_UNIT],
        capture_output=True, text=True, timeout=live_push._PROC_TIMEOUT_S)
    served = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode != 0 or not served:
        print("pull trigger failed (the 5-minute timer will pick it up):", out.stderr.strip())
        return 1
    if served != local:
        print("web host serves %s, local is %s -- pull did not land" % (served, local))
        return 1
    print("live: irisscience.org serves", served)
    return 0


if __name__ == "__main__":
    sys.exit(main())
