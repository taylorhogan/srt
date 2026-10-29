"""morning_watch.py -- did the Spark's morning run happen, and did it render?

    python scripts/morning_watch.py            # check, push only on a problem
    python scripts/morning_watch.py --dry-run  # check and print, never push

Spark cron, 10:00 daily, after morning_trigger's 09:30 deadline
(docs/AUTOMATION_PLAN.md B8, 2026-10-29). The 06:30 trigger logs to
/tmp/morning_trigger.log and nothing read it: a trigger that never ran (cron
off after a reboot, a venv gone) or a render that exited non-zero left the
owner waiting for a picture that was not coming. Like noon_watch on the PC,
this asks from outside the run and pushes through Pushover directly.

What must hold by 10:00: local/spark_render/last_run.json is dated today
(the trigger writes it at the end of every run, and when it refuses for the
disk), its render exited 0, and it was not skipped.
"""
import argparse
import json
import logging
import os
import sys
from datetime import date

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LAST_RUN = os.path.join(ROOT, "local", "spark_render", "last_run.json")
_logger = logging.getLogger(__name__)


def evaluate(last_run, today) -> list:
    """-> problems (empty = all clear). Pure. *last_run* is the record or None."""
    if not last_run:
        return ["the morning trigger left no record today (did cron run? is /tmp/morning_trigger.log "
                "moving?)"]
    if str(last_run.get("date")) != today.isoformat():
        return ["the morning trigger's last record is from %s, not today" % last_run.get("date")]
    problems = []
    if last_run.get("skipped"):
        problems.append("the morning run was skipped: %s" % last_run["skipped"])
    elif last_run.get("render_rc") not in (0, None):
        problems.append("the morning render exited %s; see local/spark_render/morning.log"
                        % last_run.get("render_rc"))
    if last_run.get("sync_ok") is False:
        problems.append("the Targets sync failed twice; last night's frames are still on iris-pc")
    return problems


def read_last_run(path=LAST_RUN):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    problems = evaluate(read_last_run(), date.today())
    if not problems:
        print("morning watch: all clear")
        return 0
    msg = "\n".join(["Iris morning watch (Spark) FAILED:"] + ["- " + p for p in problems])
    print(msg)
    if args.dry_run:
        return 1
    from utils import pushover
    if not pushover.push_message(msg, priority=1):
        _logger.error("morning watch: Pushover did not accept the alert")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
