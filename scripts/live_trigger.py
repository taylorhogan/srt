"""Push the site's live panel NOW, on an event, instead of at the next tick.

Until 2026-10-08 the live panel (status, latest frame, sky chart) was
refreshed only by the IrisLiveSkymap task every 5 minutes. The owner wants
the site to change when the observatory does, so three places call
``trigger``: the imaging state setter (roof, prelude, each slot, flats, done),
set_imaging_state.bat (the same transitions when N.I.N.A writes them), and
the frame watcher (every new light frame). The 5-minute task stays as the
floor and as the catch-up when a trigger is lost.

A trigger spawns one detached ``live_skymap.py --reason <why>`` (a render is
~30 s) and returns at once; the caller is never slowed or failed by it.

The lock. live_skymap.bat warns, from 2026-08-07, what two renders at once
do: they contend for the log and the second's push can clobber the first's
temp file mid-rename. So every render, scheduled or triggered, takes
``render_lock``; a second one finding it held simply exits ("another render
is in progress"), which is right because the one running will push the same
fresh state. The lock goes stale after STALE_S, the .bat's own kill time, so
a render killed mid-way cannot wedge the panel.
"""
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "local" / "live_render.lock"
LOG = ROOT / "local" / "live_trigger.log"
STALE_S = 240            # = the hard kill in live_skymap.bat
DEBOUNCE_S = 20          # per process: a burst of events is one render
LOG_MAX_BYTES = 1_000_000
_last = {"t": 0.0}


class RenderBusy(Exception):
    """Another render holds the lock."""


@contextmanager
def render_lock(path: Path = LOCK, stale_s: float = STALE_S):
    """Exclusive lock around one render+push. Raises RenderBusy if held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and time.time() - path.stat().st_mtime > stale_s:
            path.unlink()
    except OSError:
        pass
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RenderBusy(str(path))
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        # Only our own lock: a render that overran STALE_S has already had
        # its lock taken over, and must not remove the new holder's.
        try:
            if path.read_text() == str(os.getpid()):
                path.unlink()
        except OSError:
            pass


def _python() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def trigger(reason: str, debounce_s: float = DEBOUNCE_S) -> bool:
    """Start one detached render+push for *reason*. Never raises.

    False when debounced (another trigger from this process within
    *debounce_s*) or when the spawn itself failed.
    """
    now = time.monotonic()
    if now - _last["t"] < debounce_s:
        return False
    _last["t"] = now
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:
            LOG.unlink()
        flags = 0
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write("%s trigger: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), reason))
            subprocess.Popen([_python(), str(ROOT / "scripts" / "live_skymap.py"),
                              "--reason", reason],
                             cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, close_fds=True, creationflags=flags)
        return True
    except Exception as exc:  # noqa: BLE001 -- the site is never worth failing imaging
        try:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write("trigger failed: %s\n" % exc)
        except OSError:
            pass
        return False


if __name__ == "__main__":
    ok = trigger(" ".join(sys.argv[1:]) or "manual", debounce_s=0)
    print("triggered" if ok else "not triggered")
