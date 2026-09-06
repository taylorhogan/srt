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

# port -> (label, directories whose newest file that service is running)
SERVICES = {
    8095: ("web chat / social server", ["cmd_processing", "sentry", "hardware_control"]),
    8096: ("shadow conductor",         ["iris"]),
    8442: ("scheduler (prefect)",      ["end_points", "iris_astronomy"]),
}

PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$c = Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq %d }
if (-not $c) { Write-Output "DOWN"; exit }
$pid_ = $c.OwningProcess | Select-Object -First 1
$p = Get-CimInstance Win32_Process -Filter "ProcessId=$pid_"
if (-not $p) { Write-Output "DOWN"; exit }
$par = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)"
$parent = if ($par) { "alive" } else { "ORPHAN" }
Write-Output "$pid_|$($p.CreationDate.ToString('yyyy-MM-dd HH:mm:ss'))|$parent"
"""


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


def probe(port):
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS % port],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    if not out or out.startswith("DOWN"):
        return None
    pid, started, parent = out.splitlines()[-1].split("|")
    return int(pid), datetime.strptime(started, "%Y-%m-%d %H:%M:%S"), parent


def main():
    print("%-28s %-8s %-20s %-9s %s" % ("service", "pid", "started", "parent", "code"))
    bad = 0
    for port, (label, dirs) in SERVICES.items():
        info = probe(port)
        mtime, newest = newest_source(dirs)
        if info is None:
            print("%-28s %-8s %-20s %-9s %s" % (label, "-", "NOT LISTENING", "-", "-"))
            bad += 1
            continue
        pid, started, parent = info
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
              "replace it; a STALE service needs a restart to load what is on disk." % bad)
    else:
        print("all three up, supervised, and running the code that is on disk.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
