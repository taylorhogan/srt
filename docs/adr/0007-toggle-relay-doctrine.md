# ADR 0007: The roof relay is a toggle: sense, guard, act, confirm

Status: accepted 2026-08-28

The relay cannot be commanded open or closed -- it toggles. Therefore
roof state must always be SENSED, never assumed; every roof action is
sense -> guard -> toggle -> confirm -> journal; and FAULT_ROOF_UNKNOWN is
a first-class persisted state entered whenever sense and expectation
disagree (stall, timeout, contradiction), exited only by an operator.
Related placements: Invariant A's guard sits on the PARKING->CLOSING_ROOF
edge -- between the park CONFIRMATION and the relay fire -- because
reality parks the mount as part of closing; the close DECISION itself is
never guarded (deciding to go home must always be possible).
