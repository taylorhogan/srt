#!/usr/bin/env python3
"""Wait for the observatory to finish its flats, then sync and render.

Replaces the pair of fixed-time cron entries (sync at 07:30, render at 08:00).
Those were a guess at when the panel flats finish, and the guess is fragile:
the flat run is not dawn-locked, it follows whenever imaging stops, and the
seven sessions on record finished anywhere between 04:43 and 07:19. A sync
that starts before the flats are written misses them for a whole day, which
is what the 06:00 schedule did every night -- the morning render was always
stacking against the *previous* session's flats.

So poll instead of guess. Two signals, because neither alone is enough:

  * the FLAT directory has gone QUIET -- the newest flat on the observatory is
    older than --quiet minutes. This is the primary test. It needs nothing from
    the observatory but a directory listing, it does not care how many flats a
    session took, and it is correct on a night that produced none (there is no
    new flat, so the newest one stays old and the run proceeds).

  * NINA_FLATS_DONE in today's conductor journal. Authoritative when present --
    on 2026-10-06/07/08 it landed within 90 s of the last flat -- but it only
    appeared on 14 of 42 days, so it can confirm and never gate.

Ages are computed on the OBSERVATORY's clock, not this one, so the two
machines' clocks never have to agree.

Deadline: at --deadline the run proceeds regardless and says so loudly. A
morning render against yesterday's flats beats no render at all, and a
silent hang beats nothing.

Cron (replaces the 07:30 sync and the 08:00 render):

    30 6 * * * /home/taylor/Documents/srt/scripts/morning_trigger.py
"""
import argparse
import base64
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

PC = "iriso@100.95.7.19"
SRT = r"C:\Users\iriso\Documents\development\srt"
TARGETS = r"C:\Users\iriso\Documents\N.I.N.A\Targets"
ROOT = Path(_root)
LOG = Path("/tmp/morning_trigger.log")
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def _ps(script: str, timeout: int = 60) -> str | None:
    """Run PowerShell on the observatory; None if it could not be reached.

    Sent as -EncodedCommand. ssh to this host lands in cmd.exe, which eats
    pipes and quotes before PowerShell ever sees them -- the first version of
    this failed with "'Sort-Object' is not recognized", because cmd had split
    the pipeline and tried to run the second half itself. Base64 UTF-16LE has
    no metacharacters, so nothing can be reinterpreted on the way.
    """
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    try:
        r = subprocess.run(_SSH + [PC, "powershell", "-NoProfile", "-EncodedCommand", enc],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def flats_quiet_for() -> float | None:
    """Minutes since the newest flat was written, on the observatory's clock.

    None if the observatory could not be reached. A very large number when it
    has no flats at all, which is the right answer for a clouded-out night.
    """
    out = _ps(
        "$f = Get-ChildItem -Path '%s\\cdk17\\*\\FLAT\\*.fits' -ErrorAction SilentlyContinue "
        "| Sort-Object LastWriteTime -Descending | Select-Object -First 1; "
        "if ($f) { [int]((Get-Date) - $f.LastWriteTime).TotalMinutes } else { 99999 }" % TARGETS)
    if out is None:
        return None
    try:
        return float(out.splitlines()[-1].strip())
    except (ValueError, IndexError):
        return None


def flats_done_in_journal() -> bool:
    """NINA_FLATS_DONE in today's conductor journal. Confirming only."""
    day = dt.date.today().isoformat()
    out = _ps("Select-String -Path '%s\\local\\journal\\%s.jsonl' -Pattern 'NINA_FLATS_DONE' "
              "-ErrorAction SilentlyContinue | Measure-Object | "
              "ForEach-Object { $_.Count }" % (SRT, day))
    try:
        return out is not None and int(out.splitlines()[-1].strip()) > 0
    except (ValueError, IndexError):
        return False


def run(script: str, label: str) -> bool:
    log(f"running {label}")
    t0 = time.time()
    r = subprocess.run(["bash", str(ROOT / "scripts" / script)],
                       capture_output=True, text=True)
    log(f"  {label} finished rc={r.returncode} in {time.time() - t0:.0f}s")
    if r.returncode != 0:
        log(f"  {label} stderr: {(r.stderr or '')[-400:]}")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", type=float, default=10.0,
                    help="minutes of no new flat before the session counts as finished")
    ap.add_argument("--every", type=float, default=5.0, help="minutes between polls")
    ap.add_argument("--deadline", default="09:30",
                    help="local time to give up waiting and run anyway (HH:MM)")
    ap.add_argument("--dry-run", action="store_true", help="poll and report, run nothing")
    args = ap.parse_args()

    hh, mm = (int(x) for x in args.deadline.split(":"))
    deadline = dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline < dt.datetime.now():
        deadline += dt.timedelta(days=1)
    log(f"=== waiting for flats (quiet {args.quiet:g} min, poll {args.every:g} min, "
        f"deadline {deadline:%H:%M})")

    reason = None
    while reason is None:
        quiet = flats_quiet_for()
        if quiet is None:
            log("  observatory unreachable; will retry")
        else:
            journal = flats_done_in_journal()
            log(f"  newest flat {quiet:.0f} min old"
                + ("  (journal says NINA_FLATS_DONE)" if journal else ""))
            if quiet >= args.quiet:
                reason = ("journal + quiet" if journal else "quiet")
                break
        if dt.datetime.now() >= deadline:
            reason = "DEADLINE -- proceeding without confirmation, flats may be missing"
            log(f"  {reason}")
            break
        time.sleep(args.every * 60)

    log(f"flats settled ({reason}); starting the morning run")
    if args.dry_run:
        log("dry run: not syncing, not rendering")
        return 0
    if not run("sync_nina_targets_to_spark.bsh", "Targets sync"):
        log("sync failed -- rendering anyway against whatever is on disk")
    run("spark_morning_render.bsh", "morning render")
    log("=== done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
