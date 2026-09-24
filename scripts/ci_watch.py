"""ci_watch.py -- say when CI is green, and when it is not coming.

    python scripts/ci_watch.py             # one poll: announce transitions
    python scripts/ci_watch.py --report    # print the state, announce nothing

Runs every 5 minutes as the scheduled task IrisCIWatch (ci_watch.cmd).

WHY. The observatory deploys from `release`, which GitHub Actions fast-forwards
to `main` ONLY on a green run, and the `update` command pulls `release`. So
"is CI green?" is exactly "has release caught up with main?" -- a question the
two branch tips answer, with no GitHub token, no `gh` (not installed here) and
no access to the job logs (admin auth). Until 2026-09-24 the only way to know
was to look: CI was red from 09-10 to 09-13 and again 09-17 with nobody
noticing, and every day since someone has compared the two tips by hand
before typing `update`.

`git ls-remote` reads both tips in half a second without touching the working
tree. Two transitions matter and each is announced once:

  GREEN   release moved and now equals main -> feed + Pushover:
          "CI green: release at <sha> <subject> -- safe to update"
  STUCK   main has been ahead of release for longer than a run takes
          (STUCK_AFTER_S; a green run lands in 2-5 min) -> feed + Pushover.
          Red, or a run that never started; either way release is not coming.

The pure part (assess) is what CI tests; the git and the posting are around
it. State lives in local/ci_watch.json so a transition is announced once,
not every 5 minutes for the rest of the day.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATE_PATH = os.path.join(ROOT, "local", "ci_watch.json")
STUCK_AFTER_S = 12 * 60


# ------------------------------------------------------------------ pure

def assess(state, main, release, now, stuck_after_s=STUCK_AFTER_S):
    """(new_state, events) from the previous state and the two tips now. Pure.

    *state* is the dict last written (or {}); *now* a timezone-aware datetime.
    events is a list of ("green", sha) / ("stuck", main_sha, release_sha,
    minutes) tuples, each produced at most once per sha.
    """
    st = dict(state or {})
    events = []
    st["main"], st["release"] = main, release
    st["checked"] = now.isoformat(timespec="seconds")
    if main == release:
        if st.get("announced_green") != release and st.get("last_release") not in (None, release):
            events.append(("green", release))
            st["announced_green"] = release
        st["ahead_since"] = None
    else:
        since = st.get("ahead_since")
        if not since or st.get("ahead_main") != main:
            st["ahead_since"] = now.isoformat(timespec="seconds")
            st["ahead_main"] = main
        else:
            waited = (now - datetime.fromisoformat(since)).total_seconds()
            if waited >= stuck_after_s and st.get("announced_stuck") != main:
                events.append(("stuck", main, release, int(waited // 60)))
                st["announced_stuck"] = main
    st["last_release"] = release
    return st, events


# ------------------------------------------------------------------ io

def tips():
    """{'main': sha, 'release': sha} from origin, or None if it cannot be read."""
    try:
        out = subprocess.run(["git", "ls-remote", "origin", "refs/heads/main", "refs/heads/release"],
                             cwd=ROOT, capture_output=True, text=True, timeout=40).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            found[parts[1].rsplit("/", 1)[-1]] = parts[0]
    return found if {"main", "release"} <= set(found) else None


def subject(sha):
    """'<sha7> <first line>' for a commit, fetching it if it is not local yet."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%h %s", sha], cwd=ROOT,
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            subprocess.run(["git", "fetch", "-q", "origin", "main", "release"], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)
            r = subprocess.run(["git", "log", "-1", "--format=%h %s", sha], cwd=ROOT,
                               capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or sha[:7]
    except (OSError, subprocess.SubprocessError):
        return sha[:7]


def load_state():
    try:
        with open(STATE_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, STATE_PATH)


def announce(msg, push=True):
    try:
        from cmd_processing import social_server
        social_server.post_social_message(msg)
    except Exception:  # noqa: BLE001
        print("(feed post failed) " + msg)
    if push:
        try:
            from utils import pushover
            pushover.push_message(msg)
        except Exception:  # noqa: BLE001
            print("(pushover failed) " + msg)


def summary(st, t):
    """One line an operator can act on."""
    if t is None:
        return "CI: cannot reach origin"
    if t["main"] == t["release"]:
        return "CI green: release = main at %s" % subject(t["release"])
    since = st.get("ahead_since")
    mins = ""
    if since:
        mins = " for %d min" % int((datetime.now(timezone.utc)
                                    - datetime.fromisoformat(since)).total_seconds() // 60)
    return ("CI pending%s: main at %s, release still at %s"
            % (mins, subject(t["main"]), subject(t["release"])))


def check(announce_events=True):
    """One poll. Returns the summary line."""
    st = load_state()
    t = tips()
    if t is None:
        return summary(st, None)
    st, events = assess(st, t["main"], t["release"], datetime.now(timezone.utc))
    save_state(st)
    if announce_events:
        for ev in events:
            if ev[0] == "green":
                announce("CI green: release at %s -- safe to `update`" % subject(ev[1]))
            else:
                _, main, release, mins = ev
                announce("CI RED or stuck: main at %s has been ahead of release (%s) for %d min"
                         % (subject(main), subject(release), mins))
    return summary(st, t)


def report():
    """The summary line with NO side effects: nothing saved, nothing announced.

    The chat's `ci` command uses this rather than check(), because check()
    records what it has announced -- and a transition first seen by a chat
    command would then never be announced by the 5-minute watcher.
    """
    return summary(load_state(), tips())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="print the state; announce nothing")
    args = ap.parse_args()
    print(report() if args.report else check())
    return 0


if __name__ == "__main__":
    sys.exit(main())
