"""
kasa_ptz.py
Read and command the pan/tilt motor on a Kasa KC-series camera.

The motor is NOT on the LAN. Local ports carry the video stream and nothing
else, so every call here goes through the TP-Link cloud passthrough that
`hardware_control/kasa_cloud.py` already speaks. See docs/KASA_CAMERA_PTZ.md
for the port map behind that claim and the full method table.

Usage:
    python kasa_ptz.py list
    python kasa_ptz.py pos  "Iris cam"
    python kasa_ptz.py move "Iris cam" left [steps] [speed]
    python kasa_ptz.py goto "Iris cam" -123 363
    python kasa_ptz.py stop "Iris cam"
    python kasa_ptz.py on|off "Iris cam"   # the camera itself, not its plug

The sky camera is refused by default -- see PROTECTED below.
"""
import json
import os
import sys
import time
from itertools import product

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hardware_control import kasa_cloud as kc

PTZ = "smartlife.cam.ipcamera.ptz"

# The vertical words are "top" and "bottom". "up"/"down" are rejected, as are
# "Left" and "LEFT" -- matched lowercase and exactly. An unrecognised word
# answers -41203, the SAME code as a value out of range, so a refused move does
# not distinguish a typo from an end stop. Check the spelling before concluding
# an axis is at its limit.
DIRECTIONS = ("left", "right", "top", "bottom")
POSITIVE = ("right", "top")            # directions that increase the reading

# `speed` is step SIZE, not rate, and is limited to 1..10 (0 and 50 are
# rejected). Measured on Iris cam; 4 and 6-9 were never measured and are left
# out rather than guessed, because the solver below depends on these being
# exact.
STEP_UNITS = {1: 50, 2: 55, 3: 62, 5: 83, 10: 333}

SETTLE_S = 1.4

# Pointing this camera destroys work that cannot be recovered by pointing it
# back: it carries a plate solution (1.5 px residual, axis 4 deg off zenith,
# 104 deg FOV) and every alt/az mapping, the foliage mask and the limiting
# magnitude derive from it. If it genuinely must move, record the position
# first and RE-SOLVE afterwards rather than trusting a restored coordinate --
# see the backlash note below for why the coordinate is not enough.
PROTECTED = {"sky camera"}


class PTZError(RuntimeError):
    pass


def _device(name):
    for d in kc._call("getDeviceList").get("deviceList", []):
        if d.get("alias") == name:
            if not d.get("status"):
                raise PTZError("%r is offline" % name)
            return d
    raise PTZError("%r is not on the Kasa account" % name)


def _call(dev, method, args):
    res = kc._call("passthrough", {
        "deviceId": dev["deviceId"],
        "requestData": json.dumps({PTZ: {method: args}})})
    body = json.loads(res["responseData"])
    inner = body.get(PTZ, {}).get(method, body)
    if not isinstance(inner, dict):
        raise PTZError("%s returned %r, not an object" % (method, inner))
    return inner


def _ok(reply, method):
    """A command succeeded only if it said so.

    Non-object arguments make this firmware answer a bare {} carrying no
    err_code at all -- neither success nor a documented failure. Treating a
    missing code as success would report a move that never happened.
    """
    if reply.get("err_code") != 0:
        raise PTZError("%s rejected: %s" % (method, json.dumps(reply)))
    return reply


def position(dev):
    r = _ok(_call(dev, "get_position", {}), "get_position")
    return (r["x"], r["y"])


def capability(dev):
    r = _ok(_call(dev, "get_capability", {}), "get_capability")
    return (r["max_x"], r["max_y"])


def step(dev, direction, speed=1):
    if direction not in DIRECTIONS:
        raise PTZError("direction must be one of %s" % (DIRECTIONS,))
    _ok(_call(dev, "set_move", {"direction": direction, "speed": speed}),
        "set_move")
    time.sleep(SETTLE_S)
    return position(dev)


def stop(dev):
    return _ok(_call(dev, "set_stop", {}), "set_stop")


# The camera's own enable, independent of the Kasa plug it is powered from.
# Turning it off is a REAL disable, not a recording flag: the 19443 stream
# answers 503 with no parts at all, so the video and the microphone both stop.
# That matters for anything using this camera as a safety sensor -- see
# docs/ROOF_STATE_SENSING.md. Check `enabled()` before trusting a verdict from
# it, because a camera switched off in the Kasa app looks exactly like a camera
# that is unreachable.
SWITCH = "smartlife.cam.ipcamera.switch"


def enabled(dev):
    """True/False: is the camera itself switched on?"""
    res = kc._call("passthrough", {
        "deviceId": dev["deviceId"],
        "requestData": json.dumps({SWITCH: {"get_is_enable": {}}})})
    body = json.loads(res["responseData"]).get(SWITCH, {}).get("get_is_enable", {})
    if body.get("err_code") != 0:
        raise PTZError("get_is_enable failed: %s" % json.dumps(body))
    return body.get("value") == "on"


def set_enabled(dev, on):
    res = kc._call("passthrough", {
        "deviceId": dev["deviceId"],
        "requestData": json.dumps(
            {SWITCH: {"set_is_enable": {"value": "on" if on else "off"}}})})
    body = json.loads(res["responseData"]).get(SWITCH, {}).get("set_is_enable", {})
    _ok(body, "set_is_enable")
    return enabled(dev)


def solve(delta, max_moves=5):
    """Shortest [(speed, sign)] whose step sizes sum to *delta*, or None.

    A single step is at least 50 units, so any smaller correction has to be
    built from larger moves that cancel: +83 +83 -50 -50 -62 nets +4. Without
    this, an off-lattice target is simply unreachable and a naive loop hunts
    around it forever.
    """
    if delta == 0:
        return []
    options = [(sp, sg) for sp in sorted(STEP_UNITS) for sg in (1, -1)]
    for n in range(1, max_moves + 1):
        for combo in product(options, repeat=n):
            if sum(STEP_UNITS[sp] * sg for sp, sg in combo) == delta:
                return list(combo)
    return None


def _reapproach(dev, axis, verbose=True):
    """Re-enter the current position from a fixed direction, to take up backlash.

    The reading does NOT determine where the camera physically points. Landing
    on x=-123 from the left and from the right gave frames 121 px apart, with
    the rightward approach matching the original to 5 px. So after any move,
    step away and come back from a consistent side.

    The positive side is used because it is the only one available on the tilt
    axis: y=363 sits within one minimum step (50) of the 387 limit, so it can
    never be entered from above.
    """
    back, fwd = ("left", "right") if axis == "x" else ("bottom", "top")
    before = position(dev)
    try:
        step(dev, back, 1)
        after = step(dev, fwd, 1)
    except PTZError as exc:
        if verbose:
            print("  re-approach on %s skipped (%s)" % (axis, exc))
        return before
    if after != before and verbose:
        print("  re-approach on %s did not return to %s (now %s)"
              % (axis, before, after))
    return after


def goto(dev, want_x, want_y, verbose=True, settle=True):
    """Drive to exactly (want_x, want_y), or explain why it cannot.

    Solves each axis as a sum of step sizes rather than stepping greedily
    toward the target. A greedy loop oscillates forever on an off-lattice
    target -- the y axis steps 337 <-> 387 and never lands on 363 between them.
    """
    pos = position(dev)
    for axis, index, want in (("x", 0, want_x), ("y", 1, want_y)):
        delta = want - pos[index]
        plan = solve(delta)
        if plan is None:
            raise PTZError(
                "cannot reach %s=%d from %d: no combination of step sizes %s "
                "within 5 moves sums to %+d"
                % (axis, want, pos[index], sorted(STEP_UNITS.values()), delta))
        if verbose and plan:
            print("  %s: %+d in %d move(s)" % (axis, delta, len(plan)))
        for speed, sign in plan:
            direction = (POSITIVE[index] if sign > 0
                         else ("left" if index == 0 else "bottom"))
            pos = step(dev, direction, speed)
            if verbose:
                print("    %-6s speed %-2d -> %s" % (direction, speed, str(pos)))
        if pos[index] != want:
            raise PTZError("%s landed on %d, wanted %d -- an axis limit was hit"
                           % (axis, pos[index], want))
    if settle:
        _reapproach(dev, "x", verbose)
        _reapproach(dev, "y", verbose)
    stop(dev)
    return position(dev)


def _guard(name, force):
    if name.lower() in PROTECTED and not force:
        raise PTZError(
            "%r is protected: it carries the all-sky plate solution, and "
            "moving it invalidates every alt/az mapping derived from it. "
            "Pass --force only if you intend to re-solve afterwards." % name)


def main(argv):
    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]
    if not argv:
        print(__doc__)
        return 1
    action, rest = argv[0], argv[1:]

    if action == "list":
        for d in kc._call("getDeviceList").get("deviceList", []):
            if "KC" not in str(d.get("deviceModel", "")).upper():
                continue
            mark = "  [protected]" if d.get("alias", "").lower() in PROTECTED else ""
            print("  %-18s %-14s online=%s%s"
                  % (d.get("alias"), d.get("deviceModel"),
                     bool(d.get("status")), mark))
        return 0

    if not rest:
        print("that action needs a camera name")
        return 1
    name = rest[0]
    dev = _device(name)

    if action == "pos":
        # Not every KC camera has a motor: the KC420WS sky camera answers
        # -10008 for the whole ptz module. Report what it does have rather
        # than failing the command.
        try:
            where = "position %s  capability %s" % (position(dev), capability(dev))
        except PTZError:
            where = "no pan/tilt (fixed camera)"
        print("%s  %s  camera %s"
              % (name, where, "ON" if enabled(dev) else "OFF"))
        return 0
    if action in ("on", "off"):
        # Deliberately NOT guarded by PROTECTED: this does not move anything,
        # and being able to switch the sky camera back on without the app is
        # worth more than the risk of switching it off.
        was = enabled(dev)
        now = set_enabled(dev, action == "on")
        print("%s camera %s -> %s" % (name, "on" if was else "off",
                                      "on" if now else "off"))
        if not now:
            print("  NOTE: the 19443 stream now answers 503 -- video AND "
                  "microphone are both dead until this is switched back on")
        return 0
    if action == "stop":
        stop(dev)
        print("%s stopped at %s" % (name, position(dev)))
        return 0
    if action == "move":
        _guard(name, force)
        if len(rest) < 2:
            print("move needs a direction: %s" % (DIRECTIONS,))
            return 1
        count = int(rest[2]) if len(rest) > 2 else 1
        speed = int(rest[3]) if len(rest) > 3 else 1
        print("%s from %s" % (name, position(dev)))
        for _ in range(count):
            print("  -> %s" % str(step(dev, rest[1], speed)))
        stop(dev)
        return 0
    if action == "goto":
        _guard(name, force)
        if len(rest) < 3:
            print("goto needs x and y")
            return 1
        want = (int(rest[1]), int(rest[2]))
        print("%s from %s to %s" % (name, position(dev), want))
        final = goto(dev, want[0], want[1])
        print("%s final %s  %s"
              % (name, final, "OK" if final == want else "OFF TARGET"))
        return 0 if final == want else 2

    print("unknown action %r" % action)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (PTZError, kc.CloudError) as exc:
        print("FAILED: %s" % exc)
        sys.exit(1)
