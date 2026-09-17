"""roof_evidence.py -- when the roof was last SEEN open, and when it last could have moved.

Two timestamps in one small JSON file, shared by every process:

  open_confirmed   a gating vision read confirmed the roof OPEN (kasa_state)
  motion_possible  the roof motor was powered or its relay fired (toggle_roof,
                   utl_shelly.fire_roof_relay) -- recorded BEFORE the fire, so a
                   fire that crashes half way still counts

They exist for one decision: a stop! whose vision read cannot see the roof
because the unparked scope is in front of the tags and the gold star
(2026-09-17 00:33: tube across the frame, shroud over the star, scope tag
edge-on). "Confirmed open, and nothing that could move the roof since" is the
evidence that lets that stop! park the scope instead of leaving it tracking.
See super_user_commands.blind_park_refusal for the whole rule.

Writes never raise: an evidence file that cannot be written only makes the
blind park refuse (no open record, or a stale one), which is the old behaviour.

When the roof limit switches are commissioned (hardware_control/
roof_limit_switches.py), the open-limit reading is the better witness and
should take this file's place in the rule.
"""
import json
import logging
import os
from datetime import datetime

_logger = logging.getLogger(__name__)

PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                    "local", "roof_evidence.json")


def _now():
    return datetime.now().astimezone()


def read() -> dict:
    """{"open_confirmed": datetime|None, "motion_possible": datetime|None}."""
    try:
        with open(PATH) as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        raw = {}
    out = {}
    for key in ("open_confirmed", "motion_possible"):
        try:
            out[key] = datetime.fromisoformat(raw[key]) if raw.get(key) else None
        except (TypeError, ValueError):
            out[key] = None
    return out


def _record(key, detail):
    try:
        try:
            with open(PATH) as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            raw = {}
        raw[key] = _now().isoformat(timespec="seconds")
        raw[key + "_by"] = "%s pid=%d" % (detail, os.getpid())
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(raw, fh, indent=1)
        os.replace(tmp, PATH)
    except Exception:  # noqa: BLE001 -- evidence must never break the caller
        _logger.exception("roof_evidence: could not record %s", key)


def record_open_confirmed(detail="vision"):
    _record("open_confirmed", detail)


def record_motion_possible(detail="roof"):
    _record("motion_possible", detail)
