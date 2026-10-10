"""The only code that can make the roof move.

Two things move the roof, and both are here:

  * the motor plug ("Roof motor", a Kasa plug) -- the motor and the relay
    Shelly are powered through it;
  * the relay fire (the Shelly at hardware.roof_relay_url) -- a momentary
    pulse; the roof goes whichever way its position dictates.

Nothing here decides WHETHER to move. The callers ask first -- the
conductor (`iris.client.request_roof_move`, binding under
conductor.roof_authority) and the legacy gates (vision: scope parked) --
and then call these. What this module guarantees is the other half of
Phase 2b: there is exactly one path to the hardware, so a new caller cannot
quietly bypass the asking. tests/test_roof_single_path.py fails CI if any
other file fires the relay or switches the motor plug ON.

Moved here 2026-10-10 from cmd_processing.super_user_commands.toggle_roof
and end_points/end.py's shutdown step, with behaviour unchanged: the
sequencing around a move (relay boot wait, current and audio capture, the
travel watchdog, the log anchor the shadow conductor matches) stays in
toggle_roof, which now calls these two functions.
"""
import asyncio
import logging
from typing import Optional

_logger = logging.getLogger(__name__)

MOTOR_PLUG = "Roof motor"


def switch_motor(dev_map: dict, on: bool, also: Optional[dict] = None) -> dict:
    """Switch the roof motor plug, verified; returns kasa_do's {name: ok}.

    *also* switches other plugs in the same call (end.py powers the motor,
    cuts the mount and lights the room in one step, and that timing is kept).
    It may not name the motor plug: the motor's state is *on*.
    """
    from hardware_control import kasa_utils as ku
    inst = dict(also or {})
    if MOTOR_PLUG in inst:
        raise ValueError("switch_motor: pass the motor state as `on`, not in `also`")
    inst[MOTOR_PLUG] = "on" if on else "off"
    return asyncio.run(ku.kasa_do(dev_map, inst))


def fire_relay(timeout: float = 10):
    """Pulse the roof relay. Returns the Shelly's response, or None when the
    command did not go out (the roof did not move). Records "motion possible"
    for stop!'s blind-park evidence before the fire (utl_shelly does that)."""
    from hardware_control import utl_shelly
    return utl_shelly.fire_roof_relay(timeout=timeout)
