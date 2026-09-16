"""The conductor's client: how the legacy actuators ask before moving the roof.

Phase 2 of docs/ARCHITECTURE_PLAN.md makes the conductor authoritative for
the roof. This module is the seam: every place that fires the roof relay
(open_roof / close_roof / roof!! toggle in super_user_commands, end.py's
end-of-night close, scripts/cycle_roof.py) first posts its request and the
evidence it sensed to POST /v1/events, and fires only on `accepted`. The
conductor steps the Night machine with that evidence, so Invariant A is
decided in ONE place, journaled with the guard's reason when it refuses.

Two modes, one config flag (`conductor.roof_authority`):

  * OFF (decision-diff): the request is posted and journaled, the conductor
    reports what it WOULD have said, the caller proceeds exactly as before.
    Nothing behavioural changes; the journal gains a verdict per move.
  * ON (authority): the caller moves only on `accepted`. A refusal is
    posted to the chat with the reason. An unreachable conductor refuses
    too -- except in end.py, whose last-resort fallback is the one designed
    exception (see `decide`), because the close must happen even if the
    brain died.

Stdlib only (urllib): end.py runs from NINA's end.bat in its own process and
must not need the web server's stack to reach the conductor.
"""
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

_logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8096"
DEFAULT_TIMEOUT_S = 8.0


def _cfg() -> dict:
    try:
        from configs import config
        return config.data().get("conductor", {}) or {}
    except Exception:
        return {}


def conductor_url() -> str:
    return str(_cfg().get("url") or DEFAULT_URL).rstrip("/")


def roof_authority() -> bool:
    """True when the conductor's verdict is binding for roof motion."""
    return bool(_cfg().get("roof_authority", False))


# ------------------------------------------------------------------ evidence

def evidence_from_vision(parked: bool, closed: bool, is_open: bool) -> dict:
    """The gating vision read, as the three-valued evidence the guards want.

    vision_safety.visual_status collapses "saw the scope off park" and
    "could not see" into parked=False; both are UNKNOWN here (refuse), never
    DENIED -- a denial is a positive statement and this read cannot make it.
    The indoor camera's own raw verdict (kasa_state.last_detail["scope"]:
    'safe' / 'UNSAFE' / 'unknown') is three-valued at the source and supplies
    parked_kasa, whose DENIED is the veto the mount_parked guard honours.
    """
    kasa = "UNKNOWN"
    try:
        from sentry import kasa_state
        det = kasa_state.last_detail or {}
        kasa = {"safe": "CONFIRMED", "UNSAFE": "DENIED"}.get(det.get("scope"), "UNKNOWN")
    except Exception:
        pass
    return {
        "parked_vision": "CONFIRMED" if parked else "UNKNOWN",
        "parked_kasa": kasa,
        "roof": "CONFIRMED" if is_open else ("DENIED" if closed else "UNKNOWN"),
        "ts": time.time(),
    }


def asserted_evidence(direction: str) -> dict:
    """What an operator's `force` asserts: scope parked, roof at the position
    the move starts from. Journaled as an assertion, never as a sensor read."""
    return {
        "parked_vision": "CONFIRMED", "parked_kasa": "UNKNOWN",
        "roof": "DENIED" if direction == "open" else "CONFIRMED",
        "ts": time.time(), "asserted": True,
    }


# ------------------------------------------------------------------ transport

def post_event(event: str, source: str, data: Optional[dict] = None,
               evidence: Optional[dict] = None, kind: str = "event",
               timeout: float = DEFAULT_TIMEOUT_S) -> Optional[dict]:
    """POST one event. Returns the conductor's reply, or None if unreachable.
    Never raises: the caller decides what an unreachable conductor means."""
    body = {"event": event, "source": source, "kind": kind, "data": data or {}}
    if evidence is not None:
        body["evidence"] = evidence
    req = urllib.request.Request(
        conductor_url() + "/v1/events",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _logger.warning("conductor unreachable for %s: %s", event, exc)
        return None


def report(event: str, source: str, data: Optional[dict] = None,
           evidence: Optional[dict] = None) -> Optional[dict]:
    """Tell the conductor what happened (a confirmation, a timeout). Best
    effort: the move is already over, so nothing waits on the answer."""
    return post_event(event, source, data, evidence)


# ------------------------------------------------------------------ decision

def decide(reply: Optional[dict], authority: bool) -> tuple:
    """(allowed, reason) from the conductor's reply. Pure; CI-tested.

    reply None means unreachable. With authority the move is refused (the
    brain is the safety system now); without it the caller proceeds and the
    reason is advisory. A reply that was not accepted refuses with the
    conductor's reason under authority, and is advisory otherwise: the
    conductor journaled the would-be refusal, the legacy gates still stand.
    """
    if reply is None:
        reason = "conductor unreachable — no verdict on this roof move"
        return (not authority), reason
    if reply.get("accepted"):
        if reply.get("would_refuse"):
            # Stepped permissively (authority off at the conductor): the move
            # is allowed, the verdict is the decision-diff for the morning.
            return True, "conductor would have refused: " + str(reply["would_refuse"])
        return True, None
    reason = reply.get("guard") or reply.get("reason") or "conductor refused"
    if authority:
        return False, reason
    return True, "conductor would have refused: " + reason


def request_roof_move(event: str, source: str, evidence: dict,
                      data: Optional[dict] = None) -> tuple:
    """Ask for the roof. Returns (allowed, reason, reply).

    `reason` is None when allowed outright, the refusal when not, and an
    advisory ("conductor would have refused: ...") when allowed only because
    authority is off. Callers post the advisory to the chat: that is the
    decision-diff a night's operator reads the next morning."""
    reply = post_event(event, source, data, evidence)
    allowed, reason = decide(reply, roof_authority())
    _logger.info("conductor %s for %s: allowed=%s%s", "reply" if reply else "unreachable",
                 event, allowed, (" (%s)" % reason) if reason else "")
    return allowed, reason, reply


def state(timeout: float = DEFAULT_TIMEOUT_S) -> Optional[dict]:
    """GET /v1/state, or None if unreachable."""
    try:
        with urllib.request.urlopen(conductor_url() + "/v1/state", timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None
