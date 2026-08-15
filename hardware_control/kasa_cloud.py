#!/usr/bin/env python3
"""Read Kasa device state through TP-Link's cloud, because the LAN cannot.

The plugs are the problem the LAN path cannot solve. HS103/HS300 units speak
KLAP, and a KLAP discovery reply carries only metadata — device_id, model, mac,
owner — with **no alias**. `kasa_utils.make_discovery_map()` builds its
{alias: ip} table from exactly that reply, so every plug collapses into a single
`None` key and a lookup for "Telescope mount" finds nothing. Worse, they also
refuse TCP 9999 (the old unauthenticated port) and reject the account
credentials locally, so there is no LAN route to their state at all.

The cloud has all of it. The same account returns 29 devices with real aliases
and relay states, including `Telescope mount`, which is the exact string the roof
gate looks for. This is the route the phone app uses, and it works from this
machine today.

This matters because a failed *read* was being reported as a measured *state*:
the roof gate said "the telescope mount is powered on" when it had actually
found nothing, which sent a night's debugging at the mount instead of the plug
lookup. Callers here get None for "cannot tell" so they can say so.

Only reads live here. Nothing in this module switches a plug: the roof gate needs
to know the mount's state, not change it, and a module that cannot write cannot
be the thing that powers a mount at the wrong moment.

Credentials come from `kasa auth` in the private config, falling back to
`sky camera auth` (same TP-Link account) and then KASA_USERNAME/KASA_PASSWORD.
They are never logged.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

ENDPOINT = "https://wap.tplinkcloud.com/"
TOKEN_CACHE = Path("local/kasa_cloud_token.json")
TOKEN_TTL_S = 23 * 3600

# Every call is bounded. This module is reached from the roof gate, and an
# unbounded network call there would hang a hardware decision — the same failure
# that took the live sky feed down when scp had no timeout.
LOGIN_TIMEOUT_S = 20
CALL_TIMEOUT_S = 15

_token_mem = None


class CloudError(RuntimeError):
    """Cloud state could not be read. Never means 'the device is off'."""


def _credentials():
    cfg = config.data()
    for section in ("kasa auth", "sky camera auth"):
        a = cfg.get(section) or {}
        u, p = a.get("username"), a.get("password")
        if u and p:
            return u, p
    u = os.environ.get("KASA_USERNAME")
    p = os.environ.get("KASA_PASSWORD")
    if u and p:
        return u, p
    raise CloudError("no Kasa credentials in 'kasa auth', 'sky camera auth', "
                     "or KASA_USERNAME/KASA_PASSWORD")


def _login():
    u, p = _credentials()
    try:
        r = requests.post(ENDPOINT, json={"method": "login", "params": {
            "appType": "Kasa_Android", "cloudUserName": u, "cloudPassword": p,
            "terminalUUID": str(uuid.uuid4())}}, timeout=LOGIN_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as exc:
        # Deliberately not including the response body: a failed login can echo
        # the submitted account name back.
        raise CloudError("cloud login failed: %s" % type(exc).__name__) from exc
    if body.get("error_code"):
        raise CloudError("cloud login rejected (error_code %s)" % body["error_code"])
    return body["result"]["token"]


def _token(force_new=False):
    """Cached token. Logging in on every call would rate-limit the account."""
    global _token_mem
    if not force_new:
        if _token_mem and _token_mem[1] > time.time():
            return _token_mem[0]
        try:
            c = json.loads(TOKEN_CACHE.read_text())
            if c.get("expires", 0) > time.time():
                _token_mem = (c["token"], c["expires"])
                return c["token"]
        except (OSError, ValueError, KeyError):
            pass
    tok = _login()
    exp = time.time() + TOKEN_TTL_S
    _token_mem = (tok, exp)
    try:
        TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_CACHE.write_text(json.dumps({"token": tok, "expires": exp}))
    except OSError:
        pass                      # cache is an optimisation, not a requirement
    return tok


def _call(method, params=None, retry_auth=True):
    tok = _token()
    payload = {"method": method}
    if params:
        payload["params"] = params
    try:
        r = requests.post(ENDPOINT, params={"token": tok}, json=payload,
                          timeout=CALL_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as exc:
        raise CloudError("cloud call %s failed: %s" % (method, type(exc).__name__)) from exc
    if body.get("error_code"):
        # -20651 is an expired/invalid token; one silent re-login, then give up.
        if retry_auth and body.get("error_code") in (-20651, -20675):
            _token(force_new=True)
            return _call(method, params, retry_auth=False)
        raise CloudError("cloud call %s rejected (error_code %s)"
                         % (method, body["error_code"]))
    return body["result"]


def device_states():
    """{alias: True/False} for every switchable device on the account.

    Strip outlets are returned under their own child alias, since that is the
    name a human gave the socket and the name callers ask for. Devices that are
    offline, or that have no relay at all (cameras, light strips), are omitted
    rather than reported as off — absent must not read as 'powered down'.
    """
    out = {}
    for d in _call("getDeviceList").get("deviceList", []):
        if not d.get("status"):
            continue                             # offline: unknown, not off
        try:
            res = _call("passthrough", {
                "deviceId": d["deviceId"],
                "requestData": json.dumps({"system": {"get_sysinfo": None}})})
            info = json.loads(res["responseData"])["system"]["get_sysinfo"]
        except (CloudError, KeyError, ValueError):
            continue
        if info.get("children"):
            for c in info["children"]:
                if c.get("alias") is not None and "state" in c:
                    out[c["alias"]] = bool(c["state"])
        elif "relay_state" in info:
            out[d.get("alias") or info.get("alias")] = bool(info["relay_state"])
    return out


def is_on(name):
    """True/False for *name*, or None when it cannot be determined.

    None is the important return: it is what lets a caller distinguish "the
    device is on" from "we could not read the device", which the roof gate was
    conflating.

    Resolves a top-level alias with ONE passthrough rather than enumerating the
    account. The roof gate calls this while holding the roof lock, and walking
    all 29 devices took 12s a call; a named plug takes about 2. Strip outlets
    are not top-level, so those still fall back to the full sweep.
    """
    try:
        for d in _call("getDeviceList").get("deviceList", []):
            if d.get("alias") != name:
                continue
            if not d.get("status"):
                return None                      # offline: unknown, not off
            res = _call("passthrough", {
                "deviceId": d["deviceId"],
                "requestData": json.dumps({"system": {"get_sysinfo": None}})})
            info = json.loads(res["responseData"])["system"]["get_sysinfo"]
            if "relay_state" in info:
                return bool(info["relay_state"])
            break
        return device_states().get(name)         # strip child, or odd device
    except (CloudError, KeyError, ValueError):
        return None


if __name__ == "__main__":
    want = sys.argv[1:] or None
    try:
        states = device_states()
    except CloudError as exc:
        print("FAILED:", exc)
        raise SystemExit(1)
    for k in sorted(states, key=str.lower):
        if want and k not in want:
            continue
        print("%-28s %s" % (k, "ON" if states[k] else "off"))
