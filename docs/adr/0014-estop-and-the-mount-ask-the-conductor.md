# ADR 0014 — stop! tells the conductor, and the mount asks it

Date: 2026-09-25. Status: accepted (Phase 2b of docs/ARCHITECTURE_PLAN.md).
Builds on ADR 0013 (the conductor decides, actuators act).

## Context

ADR 0013 made the conductor authoritative for the roof: every relay fire asks
`POST /v1/events` with its evidence and fires only on `accepted`. Two things
still routed around it, and the 2026-09-17 stop! showed both at once:

* **The emergency stop never told the conductor.** stop! wrote `safety.txt`
  (which the shadow read as SAFETY_CLEARED, so the machine sat in SAFE_HOLD)
  but posted nothing. The hold was a side effect of a file, not a decision.
* **Nothing that moved or powered the mount asked.** While the conductor was
  in SAFE_HOLD, legacy code powered the mount and launched flats under an
  open roof, because the imaging run had missed the abort and the flats gate
  was a warning. Invariant B ("the mount moves only under a confirmed-open
  roof") existed as a guard function, `roof_open`, that no row used for a
  mount action -- there was no mount action to guard.

## Decision

1. **`stop!` posts `ESTOP_REQUESTED` before it touches anything.** The machine
   reaches ESTOP from every non-hold state and leaves it only on
   `OPERATOR_RESOLVE` (`resolve!`). Inside the hold the roof may still be
   closed (`_close_in_hold`: Invariant A's guards on the evidence, hold kept)
   and the scope may still be parked (`_park_in_hold`, below); nothing may
   open the roof, move the scope elsewhere, or power the mount until an
   operator has looked. The post is best effort: the physical stop never
   waits on the brain, and an unreachable conductor is logged, not obeyed.
   stop! ends by posting a journal note, `ESTOP_DONE` or `ESTOP_FAILED`.

2. **The mount asks.** Two new events, answered by *permission rows* in the
   Night machine -- rows whose destination is their source, so the outcome
   is `allowed` and the night does not move:
   * `MOUNT_MOVE_REQUESTED` -- a slew, home, park, unpark or tracking-on.
     Allowed in PRELUDE, SLOT_SETUP, SLOT_IMAGING, PARKING and MANUAL_OPEN,
     every row guarded by `roof_open`. That is Invariant B, in the table,
     asserted by `tests/test_machine.py` over every state and every snapshot.
   * `MOUNT_POWER_REQUESTED` -- switching the mount on. Allowed in the same
     states under `roof_open`, and additionally in IDLE_DAY, ARMED,
     NIGHT_DONE and FLATS under `mount_parked` + `roof_closed`: powering does
     not move the mount, and parked-under-a-shut-roof is the safe geometry
     flats and a daytime `doflats` need.
   * **No row in any hold**, and none in the roof-moving states. A request
     there is refused. That is the 09-17 case, closed.
   * **A park inside a hold** is the one exception, decided outside the table
     like the close: `roof_open` on the evidence, or -- with
     `blind_park=True` -- the blind-park rule stop! already applied
     (CLAUDE.md, operator decision 2026-09-17), journaled as an assertion.

   Sites rewired: stop!'s park and blind park, the NINA prelude launch and
   the main-sequence launch (`doit_cmd`), flats' mount power (`do_flats`),
   the prelude's mount power (`end_points/start.py`). The imaging-run open
   and close confirmations now post the confirmed roof position as evidence,
   so the conductor's roof reading is fresh when the prelude asks.

3. **Dark first.** `conductor.mount_authority` (default False) is the mount's
   `roof_authority`: off, every request is journaled with the verdict it
   would have received and posted to the chat as an advisory, and the action
   proceeds; on, a refusal stops it. The ESTOP post is live regardless -- it
   changes only the conductor's state, and under roof authority that is what
   makes "nothing opens until resolve!" true after a stop!.

## Consequences

* After a stop! the operator must `resolve!` before the next roof open or
  imaging run. On 09-17 they ran `resolve!` and were told there was nothing
  to resolve; now there is.
* `_stop_mount_motion` (stop, tracking off) does not ask: it ends a move.
* `scripts/dither_now.py` and the lab scripts that call `mount_offset` do
  not ask yet; the plan's "narrow dither guard" is still open.
* Still 2b: the relay fire into `iris/hardware/roof.py` with a CI grep, and
  `roof_evidence` / the north camera's verdict as the second roof sense (the
  owner dropped the limit switches on 2026-09-25).

## Verification

* `tests/test_machine.py`: Invariant B static (every move row carries
  `roof_open`, none in a hold) and dynamic (no snapshot with the roof not
  CONFIRMED is allowed a move from any state; power never under an unknown
  roof).
* `tests/test_authority.py`: the mount's requests under authority and
  advisory; park in ESTOP on the evidence or the blind assertion; stop!'s
  ESTOP holds the roof until resolve! while the close is still answered.
* `tests/test_emergency_stop.py`: stop! posts `ESTOP_REQUESTED` before
  anything else.
* Live: a night of clean `MOUNT_*_ALLOWED` notes with `guard_would` null,
  then `stop!` during a daytime rehearsal (`image!! 3`) -- expect ESTOP,
  the park and close answered in the hold, and `resolve!` needed before the
  next open -- then flip `mount_authority`.
