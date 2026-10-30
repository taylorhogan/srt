# ADR 0016 — a target retires itself; a stopped observatory rings the phone

Date: 2026-10-29. Status: **accepted** 2026-10-29 (phases A and B of
docs/AUTOMATION_PLAN.md; auto-stop ships in shadow, the bell live).
Builds on ADR 0011 (months unattended) and ADR 0015 (the conductor plans).

## Context

Two things still needed a person every day. Nothing retired a finished
target: `convergence.is_dso_done` and the filter advisor's "every goal met"
both existed, but one only coloured the `tonight` grid and the other a
journal note, and a converged target was ranked and picked again until the
owner typed `dbc`. And a stopped observatory was silent: entering a hold
(`SAFE_HOLD`, `FAULT_ROOF_UNKNOWN`, `ESTOP`) was journaled and nothing more;
every following noon was recorded `SKIPPED`, the scheduler posted one chat
line that `noon_watch` counted as a finished noon, and a dead scheduler or
conductor was never relaunched because `start_srt` joined the chat server
alone. The owner asked for both to be closed (2026-10-29): "alert someone
that there is a problem with the observatory and it cannot continue until
addressed", and "when to stop imaging". Disk space on both machines was
added as a stopping item the same day.

## Decision

1. **"Done" is either verdict**, kept apart: convergence (every filter at
   its flat tail) OR the advisor's frame goals all met. A light-curve
   monitor is never done. The noon plan evaluates it (`control/auto_stop`)
   and the list rides the `NIGHT_PLAN` record as `auto_stop`.
2. **A new queue status, `converged`**, is the machine's word; `completed`
   stays the owner's (`dbc`). The ranking takes only `waiting`, so a
   converged row leaves the night with no change to the ranking.
   `reopen <dso> [for N]` puts it back and holds auto-stop off it for N
   nights (default 7; the row's `auto_stop_hold_until`), because its
   convergence record still says done and the next live noon would retire
   it again. In **shadow** the list is a chat line; in
   **live** (`planner.auto_stop`) the status is written at the noon refresh,
   before the ranking reads the queue.
3. **One bell** (`iris/conductor/alerts`): a transition into a hold from
   outside one, or a blocking event (`CHECKS_FAILED`, `PLAN_FAILED`,
   `CAPTURE_LOST`, a `PLAN_BAD` for the disk) is one Pushover emergency
   (priority 2, repeats until acknowledged) with the north camera's
   deciding frame; a machine still held at noon gets one priority-1
   reminder a day; `noon_watch` treats a hold skip as a failure from
   outside the process. The key (state, day) is kept on disk so a restart
   inside a hold does not ring twice.
4. **The disk is a stopping item on both machines.** The PC's noon plan
   calls the night off in the refuse band (the reason in the plan); the
   Spark's morning trigger refuses to sync or render below its own floor
   and rings once a day; raw lights are never pruned, only cached stacks.
5. **`start_srt` supervises all three children**; a dead scheduler or
   conductor relaunches alone with a push, with its own crash-loop guard.

## Consequences

* The auto-stop runs in shadow for a week of noon lists the owner agrees
  with; the flag flip is one config line, and the status write is the only
  behaviour it adds.
* The actuators' own pushes are unchanged; the conductor's alerts add to
  them. "ROOF NOT CONFIRMED CLOSED" and a dawn retry that gives up are now
  emergencies too.
* Two more watchdogs: `scripts/convergence_refresh.py` (PC, 09:15) keeps
  `convergence.json` current when the end of night did not; and
  `scripts/morning_watch.py` (Spark, 10:00) says when the morning run did
  not happen or did not render.
