"""conductor_drill.py -- the Phase 2 deliberate-refusal drill, without moving anything.

    python scripts/conductor_drill.py stage1   # advisory mode: dry-run refusal, auto-resolve
    python scripts/conductor_drill.py before   # stage 2: snapshot, then post the hold
    python scripts/conductor_drill.py after    # stage 2: verify nothing moved (--vision to look)
    python scripts/conductor_drill.py hold     # post ESTOP_REQUESTED only
    python scripts/conductor_drill.py resolve  # post OPERATOR_RESOLVE only

WHY A HOLD. open_roof's legacy gates (no imaging run, safe!, mount off, roof lock,
vision parked + shut) all run BEFORE the conductor is asked, and the conductor's
guards read the same sensors, so every sensor-based refusal is caught by legacy
code first. The refusal that belongs to the conductor alone is its STATE: in a
hold (ESTOP here) the Night machine has no ROOF_OPEN_REQUESTED row. Posting
ESTOP_REQUESTED changes only the conductor's state; no legacy code acts on it
(only live_skymap shows "hold"). Nothing in this drill ever unparks the scope or
clears safety to provoke a refusal: under a closed roof that is the collision case.

STAGE 1 (advisory mode, roof_authority False) never fires anything: it asks the
conductor exactly as roof!! open would, with a fresh vision read, and prints the
verdict. Advisory mode lets a refused move proceed by definition, so the relay is
simply not called.

STAGE 2 (authority on; operator present) uses the real `roof!! open`:
  before -> type `roof!! open` in the chat -> after -> resolve
`before` refuses to post the hold unless roof_authority is on, because in
advisory mode that roof!! open WOULD move the roof.

Refuses to start unless the conductor is IDLE_DAY. The drill ends in IDLE_DAY
(OPERATOR_RESOLVE), so it must not run once the day is ARMED for tonight.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SNAPSHOT = os.path.join(ROOT, "local", "drill_before.json")
LOG = os.path.join(ROOT, "iris.log")
SOURCE = "operator"
TAG = {"drill": "refusal", "by": "scripts/conductor_drill.py"}

# Lines that prove the roof was energised or fired (super_user_commands.toggle_roof).
MOTION_MARKERS = ("roof relay fire:", "'Roof motor' switched on")


# ------------------------------------------------------------------ pure verdicts

def stage1_problems(held_state, reply, decision, resolved_state):
    """Problems with a stage-1 run, [] when it passed. Pure.

    held_state      conductor state after ESTOP_REQUESTED
    reply           the conductor's reply to ROOF_OPEN_REQUESTED (or None)
    decision        client.decide(reply, authority) -> (allowed, reason)
    resolved_state  conductor state after OPERATOR_RESOLVE
    """
    p = []
    if held_state != "ESTOP":
        p.append("hold not entered (state %s)" % held_state)
    if reply is None:
        p.append("conductor unreachable for the request")
    else:
        if reply.get("accepted"):
            p.append("conductor ACCEPTED a roof open while in ESTOP")
        if reply.get("state") != "ESTOP":
            p.append("request moved the machine out of ESTOP (to %s)" % reply.get("state"))
        if "no transition" not in str(reply.get("guard") or ""):
            p.append("refusal reason is not the missing row: %r" % reply.get("guard"))
    allowed, reason = decision
    if reply is not None and not reason:
        p.append("no refusal text reached the caller")
    if resolved_state != "IDLE_DAY":
        p.append("resolve did not return to IDLE_DAY (state %s)" % resolved_state)
    return p


def nothing_moved_problems(new_log_text, evidence_before, evidence_after, roof_plug,
                           roof_shut=None):
    """Problems proving the roof WAS touched, [] when nothing moved. Pure."""
    p = []
    for m in MOTION_MARKERS:
        if m in new_log_text:
            p.append("iris.log shows %r since the snapshot" % m)
    if (evidence_before or {}).get("motion_possible") != (evidence_after or {}).get("motion_possible"):
        p.append("roof_evidence motion_possible changed: %s -> %s"
                 % ((evidence_before or {}).get("motion_possible"),
                    (evidence_after or {}).get("motion_possible")))
    if roof_plug != 0:
        p.append("roof motor plug not confirmed OFF (%s)" % roof_plug)
    if roof_shut is False:
        p.append("vision does not read the roof shut")
    return p


# ------------------------------------------------------------------ plumbing

def _client():
    from iris import client
    return client


def _state():
    st = _client().state()
    return (st or {}).get("state")


def _post(event, data=None):
    return _client().post_event(event, SOURCE, dict(TAG, **(data or {})))


def _imaging_state():
    try:
        with open(os.path.join(ROOT, "imaging.txt")) as fh:
            return fh.read().split()[-1]
    except (OSError, IndexError):
        return "unknown"


def _nina_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq NINA.exe"],
                         capture_output=True, text=True).stdout
    return "NINA.exe" in out


def _preconditions():
    problems = []
    state = _state()
    if state is None:
        problems.append("conductor unreachable")
    elif state != "IDLE_DAY":
        problems.append("conductor is %s, drill needs IDLE_DAY (it ends by resolving to IDLE_DAY)" % state)
    if _imaging_state() != "NONE":
        problems.append("imaging state is %s" % _imaging_state())
    if _nina_running():
        problems.append("NINA.exe is running")
    return problems


def _roof_plug():
    import asyncio
    from hardware_control import kasa_utils as ku
    dev_map = asyncio.run(ku.make_discovery_map(expect=("Roof motor",)))
    return ku.legacy_relay(dev_map["Roof motor"]) if "Roof motor" in dev_map else None


def _vision():
    from sentry import vision_safety
    parked, closed, is_open, _ = vision_safety.visual_status()
    return parked, closed, is_open


def _say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ commands

def stage1(_args):
    client = _client()
    pre = _preconditions()
    if pre:
        _say("NOT STARTED: " + "; ".join(pre))
        return 2
    authority = client.roof_authority()
    _say("roof_authority (this process's config): %s" % authority)
    _say("vision read (as roof!! open would)...")
    parked, closed, is_open = _vision()
    evidence = client.evidence_from_vision(parked, closed, is_open)
    _say("  parked=%s closed=%s open=%s -> %s" % (parked, closed, is_open,
                                                  {k: v for k, v in evidence.items() if k != "ts"}))

    held = resolved = None
    reply = None
    decision = (None, None)
    try:
        r = _post("ESTOP_REQUESTED")
        held = _state()
        _say("ESTOP_REQUESTED -> %s (reply %s)" % (held, r and {k: r.get(k) for k in ("accepted", "state")}))
        if held == "ESTOP":
            reply = client.post_event("ROOF_OPEN_REQUESTED", SOURCE, dict(TAG), evidence)
            decision = client.decide(reply, authority)
            _say("ROOF_OPEN_REQUESTED reply: %s" % (reply and {k: reply.get(k) for k in
                                                                ("accepted", "kind", "guard", "state")}))
            _say("caller decision: allowed=%s reason=%r" % decision)
            _say("  (dry run: the relay is NOT called, whatever the decision)")
    finally:
        if _state() == "ESTOP":
            r = _post("OPERATOR_RESOLVE", {"account": "conductor_drill"})
            _say("OPERATOR_RESOLVE -> %s" % (r and r.get("state")))
        resolved = _state()
    problems = stage1_problems(held, reply, decision, resolved)
    _say("STAGE 1 %s" % ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
    return 0 if not problems else 1


def before(_args):
    client = _client()
    pre = _preconditions()
    if pre:
        _say("NOT STARTED: " + "; ".join(pre))
        return 2
    if not client.roof_authority():
        _say("NOT STARTED: roof_authority is off -- `roof!! open` would MOVE the roof in advisory mode")
        return 2
    from sentry import roof_evidence
    snap = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
            "log_offset": os.path.getsize(LOG),
            "roof_evidence": {k: (v.isoformat() if v else None) for k, v in roof_evidence.read().items()},
            "roof_plug": _roof_plug()}
    with open(SNAPSHOT, "w") as fh:
        json.dump(snap, fh, indent=1)
    r = _post("ESTOP_REQUESTED")
    _say("snapshot saved; ESTOP_REQUESTED -> %s" % (r and r.get("state")))
    _say("NOW type `roof!! open` in the web chat, wait for the refusal, then run `after`.")
    return 0


def after(args):
    try:
        with open(SNAPSHOT) as fh:
            snap = json.load(fh)
    except OSError:
        _say("no snapshot; run `before` first")
        return 2
    from sentry import roof_evidence
    with open(LOG, "rb") as fh:
        fh.seek(snap["log_offset"])
        new = fh.read().decode("utf-8", "replace")
    ev_after = {k: (v.isoformat() if v else None) for k, v in roof_evidence.read().items()}
    shut = None
    if args.vision:
        _, closed, _ = _vision()
        shut = bool(closed)
    refusal_seen = "conductor refused" in new
    problems = nothing_moved_problems(new, snap["roof_evidence"], ev_after, _roof_plug(), shut)
    if not refusal_seen:
        problems.append("no 'conductor refused' line in iris.log since the snapshot")
    _say("conductor state now: %s (run `resolve` when done)" % _state())
    _say("STAGE 2 REFUSAL %s" % ("PASSED: refused and nothing moved" if not problems
                                 else "FAILED: " + "; ".join(problems)))
    return 0 if not problems else 1


def hold(_args):
    r = _post("ESTOP_REQUESTED")
    _say("ESTOP_REQUESTED -> %s" % (r and r.get("state")))
    return 0 if r else 1


def resolve(_args):
    r = _post("OPERATOR_RESOLVE", {"account": "conductor_drill"})
    _say("OPERATOR_RESOLVE -> %s" % (r and r.get("state")))
    return 0 if r else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    sub.add_parser("before")
    a = sub.add_parser("after")
    a.add_argument("--vision", action="store_true", help="also read the roof with the camera")
    sub.add_parser("hold")
    sub.add_parser("resolve")
    args = ap.parse_args()
    from utils import utils
    utils.set_logger()
    return {"stage1": stage1, "before": before, "after": after,
            "hold": hold, "resolve": resolve}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
