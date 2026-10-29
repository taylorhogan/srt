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

It then runs, in order: the Targets sync, the sentry library sync, and the
morning render. The sentry sync is here rather than on its own 06:00 entry
because the roof closes at 06:01-06:02 and that entry missed the morning's
roof-close signature every day.

Disk (docs/AUTOMATION_PLAN.md B6, 2026-10-29): before the sync the Spark's
own drive is read (utils/disk_space; floors from the machine block,
cfg["disk"]). In the refuse band nothing is pulled or rendered -- a sync into
a full disk is how a morning dies half-way -- and the phone rings (priority
2) once a day until it is dealt with; in the warn band it warns once a day
and scripts/spark_prune.py reports what it could clear. Raw lights are never
pruned.

Every run writes local/spark_render/last_run.json (date, sync, render rc,
disk) for scripts/morning_watch.py, the 10:00 outside witness; a render that
exits non-zero is pushed from here as well.

Cron (replaces the Targets sync, the sentry sync and the morning render):

    30 6 * * * /home/taylor/Documents/srt/scripts/morning_trigger.py
"""
import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

PC = "iriso@100.95.7.19"
STATE_DIR = Path(_root) / "local" / "spark_render"
DISK_ALERT = STATE_DIR / "disk_alert.json"
LAST_RUN = STATE_DIR / "last_run.json"
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


def warn_operator(message: str, priority: int = 1) -> None:
    """A Pushover warning; never lets a notification failure stop the run."""
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from utils import pushover
        pushover.push_message(message, priority=priority)
        log(f"  operator warned by Pushover (priority {priority})")
    except Exception as e:                      # noqa: BLE001
        log(f"  Pushover warning failed: {e}")


def once_today(path: Path, key: str, today: str) -> bool:
    """True the first time *key* is asked for on *today*; remembered in *path*."""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        state = {}
    if state.get(key) == today:
        return False
    state[key] = today
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    except OSError as e:
        log(f"  could not record {key}: {e}")
    return True


def disk_verdict(u, warn_gb: float, refuse_gb: float) -> tuple:
    """(band, message) for the Spark's drive. Pure."""
    from utils import disk_space
    b = disk_space.band(u, warn_gb, refuse_gb)
    if b == "refuse":
        return b, ("Iris Spark disk nearly FULL: %.0f GB free of %.0f GB. The morning sync and "
                   "render are SKIPPED below %.0f GB; last night's frames stay on iris-pc. "
                   "Free space (scripts/spark_prune.py reports what can go), then run "
                   "scripts/morning_trigger.py by hand." % (u["free_gb"], u["total_gb"], refuse_gb))
    if b == "warn":
        return b, ("Iris Spark disk getting full: %.0f GB free of %.0f GB (warns below %.0f GB, "
                   "the morning run stops below %.0f GB). scripts/spark_prune.py lists the "
                   "stale stack caches." % (u["free_gb"], u["total_gb"], warn_gb, refuse_gb))
    if b == "unknown":
        return b, "Iris Spark: the image drive's free space could not be read"
    return b, "disk ok: %.0f GB free of %.0f GB" % (u["free_gb"], u["total_gb"])


def check_disk(today: str) -> str:
    """Read the drive, log it, ring once a day in the warn and refuse bands. -> band."""
    try:
        from configs import config
        from utils import disk_space
        cfg = config.data()
        warn_gb, refuse_gb = disk_space.thresholds(cfg)
        u = disk_space.usage(disk_space.data_path(cfg))
    except Exception as e:  # noqa: BLE001
        log(f"  disk check failed: {e}")
        return "unknown"
    band, msg = disk_verdict(u, warn_gb, refuse_gb)
    log(f"  {msg}")
    if band in ("warn", "refuse") and once_today(DISK_ALERT, "disk:" + band, today):
        warn_operator(msg, priority=2 if band == "refuse" else 1)
    if band == "warn":
        run_py("spark_prune.py", "spark_prune report")
    return band


def write_last_run(**fields) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_RUN.write_text(json.dumps(dict(fields, written=dt.datetime.now().isoformat(
            timespec="seconds")), indent=1), encoding="utf-8")
    except OSError as e:
        log(f"  could not write {LAST_RUN}: {e}")


def run_py(script: str, label: str) -> int:
    """A repo script under the venv's python; -> its return code (logged)."""
    log(f"running {label}")
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / script)],
                       capture_output=True, text=True)
    for line in (r.stdout or "").splitlines()[-12:]:
        log(f"  | {line}")
    if r.returncode != 0:
        log(f"  {label} rc={r.returncode}: {(r.stderr or '')[-400:]}")
    return r.returncode


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
    today = dt.date.today().isoformat()
    band = check_disk(today)
    if band == "refuse":
        log("disk in the refuse band -- not syncing, not rendering")
        write_last_run(date=today, disk=band, sync_ok=None, render_rc=None,
                       skipped="disk nearly full")
        return 2
    if args.dry_run:
        log("dry run: not syncing, not rendering")
        return 0
    sync_ok = True
    if not run("sync_nina_targets_to_spark.bsh", "Targets sync"):
        # The script already retried four times over ~5 min. One more go a
        # quarter hour later covers a longer wobble on the link; after that
        # the operator hears about it, because the render below will say
        # "nothing new" about a night that may have 300 frames on iris-pc
        # (2026-10-21: 290 frames, a pull that died in under a second).
        log("sync failed -- retrying once in 15 min")
        time.sleep(15 * 60)
        if not run("sync_nina_targets_to_spark.bsh", "Targets sync (retry)"):
            sync_ok = False
            log("sync failed twice -- continuing against whatever is on disk")
            warn_operator("Iris morning sync failed twice; last night's frames are still on "
                          "iris-pc and the morning render will report nothing new. "
                          "Run scripts/sync_nina_targets_to_spark.bsh by hand.")
    # Moved off its own 06:00 cron entry on 2026-10-09: the roof closes at
    # 06:01-06:02, so that run missed the morning's roof-close signature every
    # single day and collected it 24 h late. By the time the flats are quiet
    # the roof has been shut for an hour, so here it is always in time. It is
    # ~10 kB and --ignore-existing, so a failure costs nothing and never
    # blocks the render.
    run("sync_sentry_library_to_spark.bsh", "sentry library sync")
    render_ok = run("spark_morning_render.bsh", "morning render")
    if not render_ok:
        # Until 2026-10-29 a failed render was a line in this log and nothing
        # else; the owner found out when no picture came (docs/AUTOMATION_PLAN.md B8).
        warn_operator("Iris morning render FAILED (non-zero exit). No pictures this morning; "
                      "see local/spark_render/morning.log on the Spark.")
    write_last_run(date=today, disk=band, sync_ok=sync_ok, render_rc=0 if render_ok else 1)
    log("=== done")
    return 0 if render_ok else 1


if __name__ == "__main__":
    sys.exit(main())
