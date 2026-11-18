#!/home/taylor/Documents/srt/.venv/bin/python
"""Log the aircraft passing over the observatory in the web chat (2026-11-18).

Spark cron, every minute, around the clock (the owner's choice):

    * * * * * /home/taylor/Documents/srt/scripts/overhead_watch.py >> /tmp/overhead_watch.log 2>&1

One poll of the public ADS-B feed (control/overhead: adsb.lol, no key),
the passes inside the roof's circle picked by their closest approach, one
line each to the Observatory feed through utils/webchat_client, deduped by
aircraft for half an hour in local/overhead_state.json. A feed or chat
outage is silence; nothing here raises. `--dry-run` prints the lines and
posts nothing; `--now` prints what the feed hears right now, like the
chat's `overhead` command.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config  # noqa: E402
from control import overhead  # noqa: E402

STATE = Path(_root) / "local" / "overhead_state.json"


def load_state() -> dict:
    try:
        return {str(k): float(v) for k, v in json.loads(STATE.read_text()).items()}
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(overhead.prune(state), indent=0))
        tmp.replace(STATE)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the lines, post nothing, keep no state")
    ap.add_argument("--now", action="store_true", help="print what the feed hears right now and exit")
    args = ap.parse_args()
    cfg = config.data()
    if args.now:
        print(overhead.nearby_report(cfg))
        return 0
    state = {} if args.dry_run else load_state()
    stamp = datetime.now().strftime("%H:%M")

    def post(line: str) -> None:
        print(f"{datetime.now().isoformat(timespec='seconds')} {line}", flush=True)
        if args.dry_run:
            return
        from utils.webchat_client import post_to_webchat
        post_to_webchat(line)

    lines = overhead.poll_once(cfg, state, post, now=time.time(), when=stamp)
    if not args.dry_run:
        save_state(state)
    if not lines and args.dry_run:
        print("nothing passing over")
    return 0


if __name__ == "__main__":
    sys.exit(main())
