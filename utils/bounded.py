"""Run a call under a wall-clock bound, for libraries that take no timeout.

astroquery's ``Simbad.TIMEOUT`` reaches its own requests, not pyvo's: the
capability fetch under ``add_votable_fields`` (pyvo.dal.vosi, a session.get
with no timeout) waits for the OS to give up on the TCP connect. During the
CDS outage of 2026-11-02 (SYN sent, nothing back) that was 18+ minutes per
lookup, in the test suite and in anything that classified a new name. From
outside the library the only handle on it is a thread with a join timeout.
"""
import threading


def call_bounded(fn, seconds, *args, **kwargs):
    """``fn(*args, **kwargs)`` on a daemon thread: its result, its exception
    re-raised here, or TimeoutError after *seconds*. The thread is left to
    finish on its own; it holds nothing the caller waits for."""
    box = {}

    def run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 -- re-raised on the caller's thread
            box["error"] = e

    name = getattr(fn, "__name__", "call")
    t = threading.Thread(target=run, daemon=True, name="bounded:%s" % name)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError("%s did not return within %.0f s" % (name, seconds))
    if "error" in box:
        raise box["error"]
    return box.get("value")
