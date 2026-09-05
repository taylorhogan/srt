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

    None means "no idea", not "unmoved". Callers must treat it as ignorance:
    the whole point of this module is that an unverifiable pose invalidates a
    pose-dependent comparison rather than being assumed benign.
    """
    try:
        with open(path) as fh:
            entry = json.load(fh)[str(name)]
        return tuple(int(v) for v in entry["position"])
    except Exception:       # noqa: BLE001
        return None
