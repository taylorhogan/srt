"""Self-supervising roof limit switches on a Shelly Plus i4 DC.

Each SPDT limit switch wires BOTH contacts to their own Shelly input
(sheet RLS-1): the NO contact asserts "at limit", the NC contact asserts
"away from limit", and a healthy switch always asserts exactly one of the
pair. That turns the classic silent failure -- a broken switch that reads as
a position -- into a distinct, detectable signature:

    NO  NC   meaning
    on  off  at this limit
    off on   away from this limit
    off off  cut wire / dead switch        -> SENSOR FAULT
    on  on   short / pinched cable / miswire -> SENSOR FAULT

plus one cross-switch rule: both switches at-limit at once is physically
impossible and is also a fault, never a tiebreak.

DELIBERATELY NOT WIRED INTO THE SAFETY GATING. The switches become a voter
in roof-state decisions only after the commissioning drills
(scripts/roof_limits_check.py) pass on the installed hardware -- inducing
every fault row above and watching it read as a fault. Until
cfg["hardware"]["roof_limit_shelly_ip"] is set, read() reports NOT_CONFIGURED
and everything behaves as before this module existed.

decode() is pure so the truth table is pinned by CI without a Shelly.
"""
import json
import urllib.request
from dataclasses import dataclass

# Which Shelly input id (0-3, printed I1-I4 on the case) carries which
# contact. Overridable via cfg["hardware"]["roof_limit_inputs"] if the
# physical wiring lands differently.
DEFAULT_INPUTS = {"closed_no": 0, "closed_nc": 1, "open_no": 2, "open_nc": 3}

# Verdicts. OPEN/CLOSED are trustworthy position claims (both pairs healthy
# and consistent); IN_TRANSIT means both switches agree the roof is between
# limits; every other reading is FAULT and must be treated as roof UNKNOWN.
OPEN = "open"
CLOSED = "closed"
IN_TRANSIT = "in_transit"
FAULT = "fault"
NOT_CONFIGURED = "not_configured"
UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class LimitReading:
    state: str                 # OPEN/CLOSED/IN_TRANSIT/FAULT/NOT_CONFIGURED/UNREACHABLE
    detail: str = ""           # human-readable reason, "" when healthy
    raw: tuple = ()            # (closed_no, closed_nc, open_no, open_nc) as read


def _pair(no: bool, nc: bool, name: str):
    """(at_limit, fault_reason) for one switch's complementary pair."""
    if no and not nc:
        return True, None
    if nc and not no:
        return False, None
    if no and nc:
        return None, f"{name} switch: NO and NC both asserted (short or miswire)"
    return None, f"{name} switch: neither contact asserted (cut wire or dead switch)"


def decode(closed_no: bool, closed_nc: bool, open_no: bool, open_nc: bool) -> LimitReading:
    raw = (closed_no, closed_nc, open_no, open_nc)
    at_closed, err_c = _pair(closed_no, closed_nc, "CLOSED-limit")
    at_open, err_o = _pair(open_no, open_nc, "OPEN-limit")
    errs = [e for e in (err_c, err_o) if e]
    if errs:
        return LimitReading(FAULT, "; ".join(errs), raw)
    if at_closed and at_open:
        return LimitReading(FAULT, "both limits asserted at once (physically impossible)", raw)
    if at_closed:
        return LimitReading(CLOSED, raw=raw)
    if at_open:
        return LimitReading(OPEN, raw=raw)
    return LimitReading(IN_TRANSIT, raw=raw)


def read(cfg=None, timeout_s: float = 3.0) -> LimitReading:
    """Poll the Shelly and decode. Never raises.

    An unreachable or unconfigured sensor is reported as such -- the caller
    maps both to roof UNKNOWN, exactly like a switch fault: absence of a
    measurement must never read as a position.
    """
    if cfg is None:
        from configs import config
        cfg = config.data()
    hw = cfg.get("hardware", {})
    ip = hw.get("roof_limit_shelly_ip")
    if not ip:
        return LimitReading(NOT_CONFIGURED, "cfg['hardware']['roof_limit_shelly_ip'] not set")
    inputs = {**DEFAULT_INPUTS, **hw.get("roof_limit_inputs", {})}
    try:
        with urllib.request.urlopen(
                f"http://{ip}/rpc/Shelly.GetStatus", timeout=timeout_s) as r:
            status = json.load(r)
        vals = {k: bool(status[f"input:{v}"]["state"]) for k, v in inputs.items()}
    except Exception as exc:
        return LimitReading(UNREACHABLE, f"{type(exc).__name__}: {exc}")
    return decode(vals["closed_no"], vals["closed_nc"],
                  vals["open_no"], vals["open_nc"])
