"""Publish the lab site: push its repo, make the web host pull NOW, announce.

    python scripts/site_push.py [--no-announce] [--yes] [--announce SLUG] [--to EMAIL]

GitHub stays the site's source of truth and history (and taylorhogan.github.io
its mirror); the web host serves /srv/iris-site, which iris-site-pull.timer
refreshes every 5 minutes. This removes the wait: after the push it starts
that same pull service over the tailnet and reports the commit the host is
now serving, so "pushed" means "live".

Then the subscribers (2026-10-08): the full list of notes and images on the
site (scripts/site_articles) goes to the host's `iris-subscribe announce`,
which knows what it has announced before. It prints a preview of every NEW
article and the recipient count; nothing is sent until you answer y (or pass
--yes). --announce SLUG re-sends one article on purpose (an edit never
announces by itself); --to EMAIL sends to that one address only, as a test.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from scripts import live_push
from scripts import site_articles

SITE = Path.home() / "Documents" / "development" / "taylorhogan.github.io"
PULL_UNIT = "iris-site-pull.service"
SUBSCRIBE_CLI = "sudo -n -u irismail iris-subscribe"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(SITE)] + list(args), check=True,
                          capture_output=True, text=True, timeout=60).stdout.strip()


def _ssh(command: str, stdin: str = None, timeout: int = 120):
    return subprocess.run(["ssh", "-i", live_push.KEY] + live_push._SSH_OPTS + [live_push.HOST, command],
                          input=stdin, capture_output=True, text=True, timeout=timeout)


def publish() -> int:
    local = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("site has uncommitted changes; commit first:\n" + dirty)
        return 2
    subprocess.run(["git", "-C", str(SITE), "push", "-q", "origin", "main"],
                   check=True, timeout=60)
    print("pushed", local)
    # The checkout is owned by `deploy`; reading it as `taylor` needs the
    # safe.directory override (read-only, so no sudo for the rev-parse).
    out = _ssh("sudo -n systemctl start %s && git -c safe.directory=/srv/iris-site "
               "-C /srv/iris-site rev-parse --short HEAD" % PULL_UNIT, timeout=live_push._PROC_TIMEOUT_S)
    served = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode != 0 or not served:
        print("pull trigger failed (the 5-minute timer will pick it up):", out.stderr.strip())
        return 1
    if served != local:
        print("web host serves %s, local is %s -- pull did not land" % (served, local))
        return 1
    print("live: irisscience.org serves", served)
    return 0


def wants_send(answer) -> bool:
    """Only an explicit y sends. Pure; EOF (no terminal) counts as no."""
    return (answer or "").strip().lower() in ("y", "yes")


def announce(yes: bool = False, again: str = None, to: str = None) -> int:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    payload = json.dumps({"articles": site_articles.articles(html)})
    extra = (" --again %s" % again if again else "") + (" --to %s" % to if to else "")
    out = _ssh(SUBSCRIBE_CLI + " announce --dry-run" + extra, stdin=payload)
    if out.returncode != 0:
        print("announcements unavailable (%s)" % (out.stderr.strip().splitlines() or ["?"])[-1])
        return 0                     # the publish itself succeeded
    print(out.stdout.rstrip())
    if "nothing new to announce" in out.stdout:
        return 0
    if not yes:
        try:
            answer = input("\nSend? [y/N] ")
        except EOFError:
            answer = ""
        if not wants_send(answer):
            print("nothing sent")
            return 0
    out = _ssh(SUBSCRIBE_CLI + " announce" + extra, stdin=payload, timeout=300)
    print(out.stdout.rstrip())
    if out.returncode != 0:
        print("send failed:", out.stderr.strip())
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-announce", action="store_true", help="publish only")
    ap.add_argument("--yes", action="store_true", help="send without asking")
    ap.add_argument("--announce", metavar="SLUG", help="re-send one article (no publish needed)")
    ap.add_argument("--to", metavar="EMAIL", help="test: send to this address only")
    a = ap.parse_args(argv)
    if a.announce:
        return announce(yes=a.yes, again=a.announce, to=a.to)
    rc = publish()
    if rc != 0 or a.no_announce:
        return rc
    return announce(yes=a.yes, to=a.to)


if __name__ == "__main__":
    sys.exit(main())
