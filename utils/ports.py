"""ports.py -- keep our listening ports ours.

    from utils import ports
    ports.evict_loopback_squatters((8095, 8096))

Our services bind 0.0.0.0. Windows lets another process bind 127.0.0.1 on the
same port at the same time, and then every localhost client reaches THAT
process instead of us. Prefect's per-flow-run API server did exactly this on
2026-09-07: it drew 8095 from its random range, tested it by binding
127.0.0.1 (free, as far as Windows is concerned), and sat on 127.0.0.1:8095
shadowing the web chat for eleven hours -- `show` previews, the station
recorder's notice and every in-process post went into its 404 while the
Tailscale address still reached the real server and the chat looked alive.

The kill has to happen from the services' own session: an operator shell
gets "Access is denied" for the same process. start_srt calls this before
every launch, and the web chat calls it before binding, because `update`
relaunches the web chat but not start_srt -- the supervisor loop that runs
an update is the OLD start_srt, so a fix placed only there waits for a
reboot.
"""
import os
import subprocess


def loopback_squatters(ports):
    """{port: pid} for processes listening on 127.0.0.1:<port>, port in *ports*."""
    ps = ("Get-NetTCPConnection -State Listen | Where-Object { $_.LocalAddress "
          "-eq '127.0.0.1' -and $_.LocalPort -in (%s) } | ForEach-Object { "
          "\"$($_.LocalPort)|$($_.OwningProcess)\" }" % ",".join(str(p) for p in ports))
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return {}
    found = {}
    for line in out.splitlines():
        try:
            port, pid = line.strip().split("|")
            found[int(port)] = int(pid)
        except ValueError:
            continue
    return found


def evict_loopback_squatters(ports, log=print):
    """Kill every loopback-only listener on *ports* (never ourselves). Returns the pids taken."""
    taken = []
    for port, pid in loopback_squatters(ports).items():
        if pid == os.getpid():
            continue
        log("evicting pid %d squatting on 127.0.0.1:%d" % (pid, port))
        try:
            r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                taken.append(pid)
            else:
                log("could not evict pid %d: %s" % (pid, (r.stderr or r.stdout).strip()[:120]))
        except Exception as exc:  # noqa: BLE001
            log("could not evict pid %d: %r" % (pid, exc))
    return taken
