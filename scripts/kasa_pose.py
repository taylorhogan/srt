"""Where a Kasa pan/tilt camera was last commanded to point, recorded locally.

This file exists so a safety decision never has to ask the cloud where the
camera is looking. The park verdict compares an AprilTag against corner
coordinates recorded at one camera pose; if the camera has been pointed
somewhere else since, those coordinates describe a different world and the
comparison is meaningless. Detecting that must not depend on an internet round
trip, for the same reason the roof gate reads the mount's plug over the LAN
first: an internet round trip has no business in the path of a decision about
whether the roof may move.

So kasa_ptz records every move it makes here, and readers consult this file
with nothing but the filesystem. That is sound because this camera does not
drift -- it moves only when something commands it, and the only thing that
commands it is kasa_ptz. A move made from the phone app is the known gap; the
positions recorded here are "last commanded", not "measured now", and the
docstring says so rather than the code pretending otherwise.

Deliberately dependency-free (json + os). It is imported by both a CLI that
talks to the cloud and a detector that must not, so it carries neither.
"""
import json
import os

POSE_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "local", "kasa_ptz_pose.json")


def record(name, position, path=POSE_PATH):
    """Note that *name* was commanded to *position* (an (x, y) pair).

    Best-effort: a failure to record must never break the move that just
    succeeded, because the move is the thing the operator asked for. A missing
    record degrades a later verdict to "unknown", which refuses, so the
    failure still lands on the safe side.
    """
    try:
        data = {}
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
        from datetime import datetime
        data[str(name)] = {
            "position": [int(position[0]), int(position[1])],
            "when": datetime.now().astimezone().isoformat(timespec="seconds")}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        return True
    except Exception:       # noqa: BLE001 -- see docstring
        return False


def last(name, path=POSE_PATH):
    """The last commanded position for *name* as a tuple, or None.

    A HINT, never an authority. It records what this software last commanded,
    which is not the same as where the camera is: on 2026-09-05 the file read
    (-123, 363) while the camera sat at (-666, 306), because it had been
    driven from the phone app. Anything that must KNOW the pose calls
    ensure_at() and moves the camera itself.

    None means "no idea", not "unmoved".
    """
    try:
        with open(path) as fh:
            entry = json.load(fh)[str(name)]
        return tuple(int(v) for v in entry["position"])
    except Exception:       # noqa: BLE001
        return None


def ensure_at(name, want, tolerate_missing_ptz=False):
    """Put the camera AT *want* and return where it actually is, or None.

    The rule this exists to enforce: before using a pan/tilt camera for a
    pose-dependent measurement, MOVE IT to the pose you need. Do not consult a
    record of where it was last put. A record describes this software's
    intentions; the app, a power cycle and a hand on the mount are all outside
    them, and the first of those was observed doing exactly this.

    Reads the true position first and only commands a move when it differs, so
    the common case costs one round trip and no motor travel -- worth avoiding
    because goto's backlash re-approach steps away and back even when the
    delta is zero, and that is wear for nothing.

    Returns the verified (x, y) on success, or None if the camera could not be
    reached or would not land. None means the pose is UNKNOWN and every
    pose-dependent verdict downstream must refuse; it never means "probably
    fine". This talks to the cloud, because for this camera pan/tilt is a
    cloud passthrough and there is no local way to ask -- which is why callers
    in a roof-decision path pass through here deliberately rather than by
    default.
    """
    try:
        from scripts import kasa_ptz
        dev = kasa_ptz._device(name)
        now = tuple(int(v) for v in kasa_ptz.position(dev))
        want = tuple(int(v) for v in want)
        if now == want:
            record(name, now)
            return now
        final = tuple(int(v) for v in kasa_ptz.goto(dev, want[0], want[1],
                                                    verbose=False))
        record(name, final)
        return final if final == want else None
    except Exception:       # noqa: BLE001 -- unreachable is UNKNOWN, not OK
        return None
