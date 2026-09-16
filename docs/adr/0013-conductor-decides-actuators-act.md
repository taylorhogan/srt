# ADR 0013: Phase 2 lands as "the conductor decides, the actuators act"

Status: accepted 2026-09-16

Phase 2 makes the conductor authoritative for the roof. It lands in two
steps. In 2a (this ADR) the relay still fires from the legacy code, but no
site fires without first posting its request and the evidence it just
sensed to `POST /v1/events` and receiving `accepted`; the Night machine
steps with that evidence, so Invariant A is decided in one place and every
refusal is journaled with the guard's reason. 2b moves the relay fire
itself into `iris/hardware/roof.py` (CI-grepped as the only file touching
it).

Why not move the actuator first: the roof sequence is not one call. It is
motor power, a Sonos warning, the relay, a 45 s travel with a stall
watchdog, a current-signature capture, an audio capture on the one camera
stream, a 30 s wait and up to five vision confirmations. That code has a
year of nights behind it and every one of its incidents is written into
it. Deciding is the dangerous part -- the toggle relay cannot be told a
direction, so firing it on a wrong belief is the collision case -- and the
decision is what the machine and its guards exist for. So authority moves
first and the mechanism stays where it is proven.

Consequences:

* One config flag, `conductor.roof_authority`, and it ships OFF: the
  requests are posted and journaled, the verdict comes back as
  `would_refuse`, nothing behavioural changes. ON, a refusal stops the move.
  Flip only after a supervised daytime `scripts/cycle_roof.py` and a
  deliberate refusal have been watched.
* Evidence is what the actuator sensed at the moment of asking, three
  valued (`iris/client.py`), and is recorded on the journal entry the
  decision was made on. The conductor still never opens a camera.
* An unreachable conductor refuses under authority -- except `end.py`,
  whose last-resort close proceeds on the vision check alone and leaves
  `local/roof_fallback_marker.json` for the conductor to ingest at its next
  start (a confirmed close is a loud note; an unconfirmed one is
  `VISION_CONTRADICTION` -> `FAULT_ROOF_UNKNOWN`).
* The roof moved by hand outside a night has its own states
  (`MANUAL_OPENING` / `MANUAL_OPEN` / `MANUAL_CLOSING`) with the same guards
  as the night's roof moves; an unplanned `image!!` opens from `IDLE_DAY`
  with the same safety case and no weather guard (the operator is the
  weather verdict).
* Every roof-moving state times out into `FAULT_ROOF_UNKNOWN` when it
  outlives the confirm loop; a restart that lands mid-move faults the same
  way; `resolve!` is the only exit, and it means "I have looked".
* `force` is an operator assertion (scope parked, roof at the start
  position), journaled as such, checked once after the move; a forced
  toggle with no direction asserts nothing about the roof and is refused
  under authority.
