"""morning_check.py -- is the observatory in its daytime rest state?

    python scripts/morning_check.py            # check, push only on a problem
    python scripts/morning_check.py --dry-run  # check and print, never push
    python scripts/morning_check.py --always-push  # push the result even when clear

Run daily at 09:00 by the scheduled task IrisMorningCheck (morning_check.cmd).
User request 2026-09-16. Every morning all of these must hold:

  roof closed        vision: roof tag decoded at its shut position
  scope parked       vision: scope tag within tolerance of the park pose
  mount off          Kasa "Telescope mount" relay read back as off
  Pegasus ports off  UPBv3 ports 1-3 (camera, gemini, fan) read back at level 0
  Kasa reachable     every plug the system commands resolves by name AND
                     answers a relay read (the inside camera is covered by
                     the vision read, which cannot happen without it)

Anything that cannot be READ counts as a failure, the same way it does at the
roof gates: "could not confirm off" is not "off". Nothing here moves hardware
except what a normal vision read does (camera to its reference pose, inside
light on then restored). The mount is expected to be unpowered, so PWI4 has no
park state to offer and is not asked; the tag is the park sensor.

SCOPE TAG HEALTH. The trigger for this check was the scope tag's corner starting
to peel (2026-09-16). A peeling tag does not fail all at once. It decodes in
fewer frames, and a lifted corner shows up as a growing worst-corner offset
while the other corners stay still. So each morning's decode count and
per-corner offsets go to local/morning_check_log.jsonl, and a scope tag missing
from any frame of a daylight read is pushed as a warning even when the verdict
still says parked.

A hung check is itself a problem worth hearing about. A watchdog pushes and
exits if the whole run passes HARD_TIMEOUT_S.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils import utils

_logger = utils.set_logger()

LOG_PATH = "local/morning_check_log.jsonl"
HARD_TIMEOUT_S = 600

# The plugs the system commands. Same list as the imaging preflight
# (super_user_commands.KASA_REQUIRED), kept literal here so this check does not
# import the web chat stack. Advisory lights are deliberately not checked.
KASA_DEVICES = ("Telescope mount", "Roof motor", "Iris inside light")
MOUNT = "Telescope mount"
SCOPE_TAG_ID = 0


def evaluate(vision, kasa, pegasus):
    """(problems, warnings) from the three readings. Pure, so it can be tested.

    vision  : {"closed", "parked", "camera", "why", "scope_tag_frames", "frames",
               "worst_corner_px"}
    kasa    : {"resolved": {name: relay 0/1/None}, "error": str|None}
               a name absent from resolved did not answer discovery
    pegasus : {port: level} or None when the box could not be read
    """
    problems, warnings = [], []

    if not vision.get("camera"):
        problems.append("vision read failed (%s): roof and park unconfirmed"
                        % (vision.get("why") or "no frame"))
    else:
        if not vision.get("closed"):
            problems.append("roof NOT confirmed closed")
        if not vision.get("parked"):
            problems.append("scope NOT confirmed parked (scope tag %s)"
                            % vision.get("scope_verdict", "unknown"))
        seen, n = vision.get("scope_tag_frames", 0), vision.get("frames", 0)
        if n and seen < n:
            warnings.append("scope tag decoded in only %d/%d frames -- check the tag "
                            "(peeling?)" % (seen, n))

    if kasa.get("error"):
        problems.append("Kasa discovery failed (%s)" % kasa["error"])
    resolved = kasa.get("resolved") or {}
    for name in KASA_DEVICES:
        if name not in resolved:
            problems.append("Kasa '%s' not found on the LAN" % name)
        elif resolved[name] is None:
            problems.append("Kasa '%s' found but did not answer a relay read" % name)
    mount = resolved.get(MOUNT)
    if mount == 1:
        problems.append("telescope mount is powered ON")

    if pegasus is None:
        problems.append("Pegasus could not be read (Unity not running?): ports 1-3 unconfirmed")
    else:
        on = [p for p in (1, 2, 3) if pegasus.get(p) != 0]
        if on:
            problems.append("Pegasus port(s) not off: %s"
                            % ", ".join("%d=%s" % (p, pegasus.get(p)) for p in on))
    return problems, warnings


def read_kasa():
    from hardware_control import kasa_utils as ku
    try:
        dev_map = asyncio.run(ku.make_discovery_map(expect=KASA_DEVICES))
    except Exception as exc:  # noqa: BLE001
        return {"resolved": {}, "error": "%s: %s" % (type(exc).__name__, exc)}
    resolved = {name: ku.legacy_relay(dev_map[name])
                for name in KASA_DEVICES if name in dev_map}
    return {"resolved": resolved, "error": None}


def read_pegasus():
    from hardware_control import pegasus
    try:
        return pegasus.read_power_ports()
    except Exception:  # noqa: BLE001
        _logger.warning("morning check: Pegasus read raised", exc_info=True)
        return None


def read_vision():
    from sentry import vision_safety, kasa_state
    try:
        parked, closed, _open, _when = vision_safety.visual_status()
    except Exception as exc:  # noqa: BLE001
        return {"camera": False, "why": "%s: %s" % (type(exc).__name__, exc)}
    det = dict(kasa_state.last_detail or {})
    if not det.get("camera"):
        return {"camera": False,
                "why": (vision_safety.last_match or {}).get("error") or det.get("why")}
    per_frame = det.get("per_frame") or []
    sdet = det.get("scope_detail") or {}
    per_id = sdet.get("per_id") or {}
    tag = per_id.get(SCOPE_TAG_ID) or per_id.get(str(SCOPE_TAG_ID)) or {}
    return {"camera": True, "parked": bool(parked), "closed": bool(closed),
            "scope_verdict": det.get("scope"), "roof_verdict": det.get("roof"),
            "frames": len(per_frame),
            "scope_tag_frames": sum(1 for f in per_frame if SCOPE_TAG_ID in (f.get("tags") or [])),
            "worst_corner_px": sdet.get("worst_corner_px"),
            "scope_tag_mean_px": tag.get("mean_px"),
            "regimes": [f.get("regime") for f in per_frame]}


def _append_log(entry):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        _logger.warning("morning check: could not append %s", LOG_PATH, exc_info=True)


def _watchdog(dry_run):
    def fire():
        msg = "Iris morning check HUNG (> %d s) -- state not verified" % HARD_TIMEOUT_S
        _logger.error(msg)
        if not dry_run:
            from utils import pushover
            pushover.push_message(msg)
        os._exit(3)
    t = threading.Timer(HARD_TIMEOUT_S, fire)
    t.daemon = True
    t.start()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print; never push")
    ap.add_argument("--always-push", action="store_true",
                    help="push the result even when all clear (tests the Pushover path)")
    args = ap.parse_args()
    _watchdog(args.dry_run)

    # Kasa and Pegasus first: they are quick, and the vision read switches the
    # inside light, which would be pointless to do before knowing the plugs answer.
    kasa = read_kasa()
    pegasus = read_pegasus()
    vision = read_vision()
    problems, warnings = evaluate(vision, kasa, pegasus)

    entry = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
             "ok": not problems, "problems": problems, "warnings": warnings,
             "vision": vision, "kasa": kasa, "pegasus": pegasus}
    _append_log(entry)
    print(json.dumps(entry, indent=2, default=str))

    if not problems and not warnings and not args.always_push:
        _logger.info("morning check: all clear (scope tag %s/%s frames, %.1f px off park)",
                     vision.get("scope_tag_frames"), vision.get("frames"),
                     float(vision.get("worst_corner_px") or 0.0))
        return 0

    if problems:
        lines = ["Iris morning check FAILED:"] + ["- " + p for p in problems]
    elif warnings:
        lines = ["Iris morning check: OK, with a warning:"]
    else:
        lines = ["Iris morning check: all clear (roof closed, scope parked, mount off, "
                 "Pegasus 1-3 off, Kasa reachable; scope tag %s/%s frames, %.1f px off park)"
                 % (vision.get("scope_tag_frames"), vision.get("frames"),
                    float(vision.get("worst_corner_px") or 0.0))]
    lines += ["- warning: " + w for w in warnings]
    msg = "\n".join(lines)
    _logger.warning(msg.replace("\n", " | "))
    if args.dry_run:
        return 1 if problems else 0

    from configs import config
    from utils import pushover
    image = None
    if not vision.get("parked") or not vision.get("closed") or warnings:
        view = config.data().get("camera safety", {}).get("scope_view")
        image = view if view and os.path.exists(view) and vision.get("camera") else None
    if not pushover.push_message(msg, image=image):
        _logger.error("morning check: Pushover did not accept the alert")
        return 2
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
