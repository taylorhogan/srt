"""noon_watch.py -- did today's noon check finish, and is the web chat answering?

    python scripts/noon_watch.py            # check, push only on a problem
    python scripts/noon_watch.py --dry-run  # check and print, never push

Run daily at 12:30 by the scheduled task IrisNoonWatch (noon_watch.cmd).

WHY. On 2026-10-15 no noon report arrived and nothing said so. The web chat
server had stopped answering at ~09:43 (every post timed out), and the
scheduler entered NOON_CHECK at 12:00:07 and logged nothing after it. Both
processes were alive, holding their ports, so nothing that asks "is it up?"
would have noticed. Every report goes out through the web chat, so a silent
web chat cannot report its own failure: this check asks from outside and
pushes through Pushover directly.

Two things must hold by 12:30:

  noon check done   iris.log has, dated today, the scheduler leaving the noon
                    phase: "State: WAITING_FOR_PRE_SUNSET" (imaging planned and
                    the sequence written) or "Skipping tonight" (too cloudy).
                    Entering NOON_CHECK is not enough -- that is where it hung.
  web chat answers  an HTTP request to the chat server returns within
                    CHAT_TIMEOUT_S, any status.

Either failing is pushed. A push names what the log last said, so the operator
can tell a hang in the noon check from a scheduler that never woke up.
"""
import argparse
import logging
import os
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# utils / config / pushover are imported in main(): CI runs the tests with
# nothing but pytest installed (see the note in morning_check.py).
_logger = logging.getLogger(__name__)

LOG_FILE = os.path.join(ROOT, "iris.log")
TAIL_BYTES = 4 * 1024 * 1024   # a day of iris.log is well under this
CHAT_PORT = 8095
CHAT_TIMEOUT_S = 15
NOON_DONE = ("State: WAITING_FOR_PRE_SUNSET", "Skipping tonight")
NOON_START = "State: NOON_CHECK"
SCHED_ERROR = "Unhandled exception in scheduler loop"
LOG_DATE_FMT = "%m/%d/%Y"      # iris.log lines start "10/15/2026 12:00:07 PM ..."


def todays_lines(lines, today):
    """The log lines dated `today` (continuation lines of a traceback carry no
    date and are dropped)."""
    prefix = today.strftime(LOG_DATE_FMT) + " "
    return [ln for ln in lines if ln.startswith(prefix)]


def evaluate(lines, today, chat_ok, chat_why=""):
    """-> list of problems (empty = all clear). `lines` are iris.log lines."""
    today_lines = todays_lines(lines, today)
    problems = []
    if not any(any(m in ln for m in NOON_DONE) for ln in today_lines):
        started = [ln for ln in today_lines if NOON_START in ln]
        if started:
            last = today_lines[-1][:120] if today_lines else ""
            problems.append("noon check STARTED (%s) but never finished; last log line: %s"
                            % (started[0][11:22].strip(), last))
        else:
            problems.append("noon check never started today (scheduler not waking at noon?)")
    if any(SCHED_ERROR in ln for ln in today_lines):
        problems.append("scheduler logged an unhandled exception today")
    if not chat_ok:
        problems.append("web chat server not answering on port %d (%s)" % (CHAT_PORT, chat_why))
    return problems


def read_log_tail(path=LOG_FILE, nbytes=TAIL_BYTES):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - nbytes))
            return fh.read().decode("utf-8", errors="replace").splitlines()
    except OSError as exc:
        _logger.error("noon watch: cannot read %s: %s", path, exc)
        return []


def chat_alive(port=CHAT_PORT, timeout=CHAT_TIMEOUT_S):
    """(ok, why). Any HTTP response counts: the question is whether it answers."""
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:%d/" % port, timeout=timeout)
        return True, ""
    except urllib.error.HTTPError:
        return True, ""
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print; never push")
    args = ap.parse_args()
    from utils import utils
    utils.set_logger()

    chat_ok, chat_why = chat_alive()
    problems = evaluate(read_log_tail(), datetime.now().date(), chat_ok, chat_why)
    if not problems:
        _logger.info("noon watch: all clear (noon check finished, web chat answering)")
        print("noon watch: all clear")
        return 0

    msg = "\n".join(["Iris noon watch FAILED -- no noon report is coming:"]
                    + ["- " + p for p in problems]
                    + ["Fix: restart start_srt, then scripts/service_check.py"])
    _logger.warning(msg.replace("\n", " | "))
    print(msg)
    if args.dry_run:
        return 1
    from utils import pushover
    if not pushover.push_message(msg, priority=1):
        _logger.error("noon watch: Pushover did not accept the alert")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
