#!/home/taylor/Documents/srt/.venv/bin/python
"""Watch the Seestar's presence on the house wifi (2026-11-27).

The owner wants to know how long the device stays connected and when it
drops. Spark cron, every minute:

    * * * * * /home/taylor/Documents/srt/scripts/seestar_watch.py >> /tmp/seestar_watch.log 2>&1

One probe a minute: resolve ``cfg["seestar"]["host"]`` (seestar.local) and
open a TCP connection to its SMB port (``watch_port``, 445; the share the
pull script reads). Up is "the port answered"; a name that does not resolve
or a port that does not answer is a miss. The device is declared DOWN after
``down_after`` (2) consecutive misses, so a one-minute blip is a note in the
log and not a chat line, and UP again on the first answer. Every change is
appended to ``local/seestar_watch.jsonl`` with the time, the address and
how long the previous state lasted, and posted as one line to the
Observatory feed through utils/webchat_client. The current state lives in
``local/seestar_watch.json``. Nothing is written to the device; the probe
is a connection that is closed at once, independent of the one control
client the Seestar allows on port 4700.

``--status`` prints the state, ``--report [DAYS]`` summarises the log (the
sessions, their lengths, the gaps), ``--dry-run`` probes and prints the
verdict without touching the state or the chat. A chat outage is silence;
nothing here raises.
"""
import argparse
import json
import os
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if __package__ is None or __package__ == "":
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config  # noqa: E402

STATE = Path(_root) / "local" / "seestar_watch.json"
LOG = Path(_root) / "local" / "seestar_watch.jsonl"
DEFAULTS = {"host": "seestar.local", "watch_port": 445, "down_after": 2, "probe_timeout_s": 3.0}


# ------------------------------------------------------------------ the probe

def probe(host: str, port: int, timeout_s: float = 3.0) -> dict:
    """{"ok": bool, "ip": str|None, "why": str}. Never raises."""
    try:
        ip = socket.gethostbyname(host)
    except OSError as exc:
        return {"ok": False, "ip": None, "why": "name %s does not resolve (%s)" % (host, exc)}
    try:
        with socket.create_connection((ip, int(port)), timeout=timeout_s):
            pass
    except OSError as exc:
        return {"ok": False, "ip": ip, "why": "%s:%s not answering (%s)" % (ip, port, exc)}
    return {"ok": True, "ip": ip, "why": ""}


# --------------------------------------------------------------- the decision

def _parse(ts):
    return datetime.fromisoformat(ts) if ts else None


def decide(state: dict, ok: bool, now: datetime, ip=None, down_after: int = 2):
    """Fold one probe into the state. Returns (new_state, event) where event is
    None or {"event": "up"|"down", "at", "ip", "after_s", "misses"}.

    ``state`` keys: up (bool|None: never probed), since (iso: when the current
    state began), ip, last_seen (iso), misses (consecutive failed probes while
    up), first_miss (iso), checked (iso). A fresh state with no history takes
    the first probe's verdict without an event, so a restart never announces.
    """
    s = dict(state or {})
    s["checked"] = now.isoformat(timespec="seconds")
    was = s.get("up")
    event = None
    if ok:
        s["last_seen"] = now.isoformat(timespec="seconds")
        if ip:
            s["ip"] = ip
        if was is None:
            s.update(up=True, since=now.isoformat(timespec="seconds"), misses=0, first_miss=None)
        elif was is False:
            down_since = _parse(s.get("since")) or now
            event = {"event": "up", "at": s["checked"], "ip": s.get("ip"),
                     "after_s": int((now - down_since).total_seconds()), "misses": 0}
            s.update(up=True, since=now.isoformat(timespec="seconds"), misses=0, first_miss=None)
        else:
            if s.get("misses"):
                # a blip that healed: remembered in the state file, not announced
                s["blips"] = int(s.get("blips", 0)) + 1
            s["misses"], s["first_miss"] = 0, None
        return s, event
    # a miss
    if was is None:
        s.update(up=False, since=now.isoformat(timespec="seconds"), misses=0, first_miss=None)
        if ip:
            s["ip"] = ip
        return s, None
    if was is False:
        return s, None
    misses = int(s.get("misses", 0)) + 1
    s["misses"] = misses
    if misses == 1:
        s["first_miss"] = now.isoformat(timespec="seconds")
    if misses >= max(1, int(down_after)):
        first = _parse(s.get("first_miss")) or now
        up_since = _parse(s.get("since")) or first
        event = {"event": "down", "at": first.isoformat(timespec="seconds"), "ip": s.get("ip"),
                 "after_s": int((first - up_since).total_seconds()), "misses": misses}
        s.update(up=False, since=first.isoformat(timespec="seconds"), misses=0, first_miss=None)
    return s, event


def span(seconds) -> str:
    seconds = max(0, int(seconds or 0))
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    if d:
        return "%d d %d h" % (d, h)
    if h:
        return "%d h %02d min" % (h, m)
    if m:
        return "%d min" % m
    return "%d s" % seconds


def line_for(event: dict) -> str:
    ip = (" at %s" % event["ip"]) if event.get("ip") else ""
    if event["event"] == "down":
        return "Seestar off the wifi%s, after %s up" % (ip, span(event.get("after_s")))
    return "Seestar back on the wifi%s, after %s off" % (ip, span(event.get("after_s")))


# ------------------------------------------------------------------ the files

def load_state(path: Path = STATE) -> dict:
    try:
        return dict(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict, path: Path = STATE) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        pass


def append_log(event: dict, path: Path = LOG) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:  # noqa: BLE001
        pass


def read_log(path: Path = LOG) -> list:
    out = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except Exception:  # noqa: BLE001
        pass
    return out


# ----------------------------------------------------------------- the report

def report(events: list, state: dict, now: datetime, days: float = 7.0) -> str:
    """The sessions in the last ``days``: each 'down' event closes an up
    session of its after_s, each 'up' event closes a gap. The open session
    or gap, from the state, is counted to ``now``."""
    cut = now - timedelta(days=days)
    recent = [e for e in events if (_parse(e.get("at")) or cut) >= cut]
    ups = [e["after_s"] for e in recent if e.get("event") == "down"]
    gaps = [e["after_s"] for e in recent if e.get("event") == "up"]
    lines = ["Seestar, last %g days: %d drop%s, %d return%s"
             % (days, len(ups), "" if len(ups) == 1 else "s", len(gaps), "" if len(gaps) == 1 else "s")]
    if ups:
        lines.append("  up sessions: longest %s, shortest %s, median %s"
                     % (span(max(ups)), span(min(ups)), span(sorted(ups)[len(ups) // 2])))
    if gaps:
        lines.append("  gaps: longest %s, shortest %s" % (span(max(gaps)), span(min(gaps))))
    if state.get("up") is not None:
        since = _parse(state.get("since"))
        held = span((now - since).total_seconds()) if since else "?"
        lines.append("  now: %s for %s%s" % ("UP" if state.get("up") else "DOWN", held,
                                            (" at %s" % state["ip"]) if state.get("ip") else ""))
        if state.get("blips"):
            lines.append("  one-minute blips healed without a line: %d" % int(state["blips"]))
    for e in recent[-12:]:
        lines.append("  %s  %s" % (e.get("at", "?"), line_for(e)))
    return "\n".join(lines)


# ------------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="probe and print, keep no state, post nothing")
    ap.add_argument("--status", action="store_true", help="print the state file")
    ap.add_argument("--report", nargs="?", const=7.0, type=float, metavar="DAYS",
                    help="summarise the log over the last DAYS (7)")
    args = ap.parse_args(argv)
    scfg = dict(DEFAULTS)
    scfg.update(config.data().get("seestar", {}) or {})
    now = datetime.now().astimezone()
    if args.status:
        print(json.dumps(load_state(), indent=1))
        return 0
    if args.report is not None:
        print(report(read_log(), load_state(), now, args.report))
        return 0
    res = probe(scfg["host"], scfg["watch_port"], float(scfg.get("probe_timeout_s", 3.0)))
    if args.dry_run:
        print("%s: %s%s" % (now.isoformat(timespec="seconds"), "UP" if res["ok"] else "MISS",
                            (" at " + res["ip"]) if res["ip"] else "", ), res["why"] or "")
        return 0
    state, event = decide(load_state(), res["ok"], now, res["ip"], int(scfg.get("down_after", 2)))
    save_state(state)
    if event:
        append_log(event)
        line = line_for(event)
        print(now.isoformat(timespec="seconds"), line)
        try:
            from utils.webchat_client import post_to_webchat
            post_to_webchat(line)
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
