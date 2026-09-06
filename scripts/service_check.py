"""service_check.py — are the observatory's services up, supervised, and current?

    python scripts/service_check.py

Three questions, because on 2026-09-06 all three had different answers and only
the third one mattered.

  UP?         is anything listening on the port
  SUPERVISED? does its parent process still exist
  CURRENT?    did it start AFTER the newest source file it depends on

THE FAILURE THIS EXISTS TO CATCH. The shadow conductor ran ORPHANED from
2026-08-30: its parent `start_srt` had died, it kept holding port 8096, and
every supervised conductor started afterwards failed to bind and exited
immediately. start_srt only joins the social server, so a dead third child is
silent -- nothing in iris.log, no error anywhere. Every `update` for a week
restarted two services out of three while appearing to restart everything, and
the conductor ran month-old code against a repo that kept moving.

Nothing about that was visible from the outside. "The service is up" was true.
"The restart worked" was true. Only "the running code is current" was false,
and nothing was checking it.

So CURRENT is compared against file mtimes rather than trusted: a process that
started before the newest file it imports is running something other than what
is on disk, whatever anyone believes about the last restart.
"""
import os
import subprocess
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# label -> directories whose newest .py that service is running
SOURCES = {
    "web chat / social server": ["cmd_processing", "sentry", "hardware_control"],
    "shadow conductor":         ["iris"],
    "scheduler":                ["end_points", "iris_astronomy"],
}
SOCIAL_PORT = 8095
CONDUCTOR_PORT = 8096
# The scheduler has no fixed port: PREFECT_API_URL is blank, so Prefect starts
# an ephemeral server on a random loopback port each run (8169 on 2026-09-06).
# It is found by family instead -- the sibling of the web chat under start_srt.

PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } |
  ForEach-Object { "P|$($_.ProcessId)|$($_.ParentProcessId)|$($_.CreationDate.ToString('yyyy-MM-dd HH:mm:ss'))" }
Get-NetTCPConnection -State Listen | ForEach-Object { "L|$($_.LocalPort)|$($_.OwningProcess)" }
"""


def snapshot():
    """({pid: (ppid, started)}, {port: pid}) for every python process."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True, timeout=60).stdout
    procs, ports = {}, {}
    for line in out.splitlines():
        f = line.strip().split("|")
        if f[0] == "P":
            procs[int(f[1])] = (int(f[2]), datetime.strptime(f[3], "%Y-%m-%d %H:%M:%S"))
        elif f[0] == "L":
            ports.setdefault(int(f[1]), int(f[2]))
    return procs, ports


def newest_source(dirs):
    """(mtime, path) of the newest .py under any of *dirs*."""
    best = (0.0, None)
    for d in dirs:
        for root, _, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in root:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(root, f)
                try:
                    m = os.path.getmtime(p)
                except OSError:
                    continue
                if m > best[0]:
                    best = (m, os.path.relpath(p, ROOT))
    return best


def main():
    procs, ports = snapshot()

    def info(pid):
        if pid is None or pid not in procs:
            return None
        ppid, started = procs[pid]
        return pid, started, ("alive" if ppid in procs else "ORPHAN"), ppid

    social = info(ports.get(SOCIAL_PORT))
    conductor = info(ports.get(CONDUCTOR_PORT))
    scheduler = None
    if social and social[2] == "alive":
        # start_srt's other children; the conductor (if any) is excluded by pid
        for pid, (ppid, _) in procs.items():
            if ppid == social[3] and pid != social[0] and pid != (conductor or (None,))[0]:
                scheduler = info(pid)
                break
    found = {"web chat / social server": social,
             "shadow conductor": conductor,
             "scheduler": scheduler}

    print("%-28s %-8s %-20s %-9s %s" % ("service", "pid", "started", "parent", "code"))
    bad = 0
    for label, dirs in SOURCES.items():
        mtime, newest = newest_source(dirs)
        hit = found[label]
        if hit is None:
            why = "NOT LISTENING" if label != "scheduler" else "NOT FOUND"
            print("%-28s %-8s %-20s %-9s %s" % (label, "-", why, "-", "-"))
            bad += 1
            continue
        pid, started, parent, _ = hit
        stale = started.timestamp() < mtime
        code = "STALE" if stale else "current"
        print("%-28s %-8d %-20s %-9s %s"
              % (label, pid, started.strftime("%Y-%m-%d %H:%M"), parent, code))
        if stale:
            print("%-28s %s" % ("", "   running code older than %s (%s)"
                                % (newest, datetime.fromtimestamp(mtime)
                                   .strftime("%Y-%m-%d %H:%M"))))
        if parent == "ORPHAN":
            print("%-28s %s" % ("", "   PARENT IS GONE -- this survives restarts and "
                                    "blocks the supervised one from binding"))
        bad += stale or parent == "ORPHAN"
    print()
    if bad:
        print("%d problem(s). An ORPHAN must be stopped before a restart can "
              "replace it; a STALE service needs a restart to load what is on disk; "
              "a service that is NOT LISTENING died at start -- check start_srt."
              % bad)
    else:
        print("all three up, supervised, and running the code that is on disk.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
