# ADR 0015 — the conductor plans the night; the scheduler carries it out

Date: 2026-10-16. Status: **accepted** 2026-10-16 (Phase 3 step 2 of docs/ARCHITECTURE_PLAN.md).
Builds on ADR 0013 (the conductor decides, actuators act) and on Phase 3
step 1 (shadow planning, d02a8e0).

## Context

Step 1 had the conductor re-plan every noon and sunset check beside the
scheduler, using the same code (`control/night_plan`) in its own process.
From 2026-10-10 to 2026-10-15 the journal holds 12 `PLAN_MATCH`, and no
`PLAN_DIFF`, `SHADOW_PLAN_FAILED` or `SHADOW_PLAN_SKIPPED`. The step's
criterion was two clean nights; there are six.

Today the scheduler (`end_points/scheduler_server.py`) still owns both the
clock and the decision, in-process: it sleeps to noon, ranks the queue,
decides, writes the sequence, sleeps to sunset − 10 min, re-ranks, maybe
regenerates, and launches `image!!`. Three costs show up in the record:

* **A hang is silent.** On 2026-10-15 the scheduler entered NOON_CHECK at
  12:00:07 and logged nothing more, likely stuck on a SIMBAD socket inside
  the ranking. No timeout, no journal entry, no push; only the new noon
  watch (scripts/noon_watch.py, outside both processes) would now notice.
* **A restart re-decides.** `_initial_state()` re-runs the noon check after
  any afternoon restart, against a newer forecast. 2026-09-13 a restart
  re-planned and a name-only regeneration lost m33 (d3921f0).
* **The conductor only learns of the plan from a file.** `ARMED` in the
  machine is inferred from `scheduler_state.json`, so it survives a
  conductor restart only as far as the file agrees.

## Decision

**The conductor owns the clock and the decision; the scheduler becomes the
executor of the decision.** Behind a new flag `conductor.drives_checks`
(False = today's behavior, shadow planning continues unchanged).

### 1. The conductor's clock

A clock thread in the conductor offers two events itself:

* `NOON_TICK` at 12:00 local. IDLE_DAY → PLANNING (the row exists).
* `PRE_SUNSET_TICK` at sunset − 10 min. ARMED → PRE_FLIGHT (the row exists).
  Sunset computed in the conductor (astral, already a dependency), not read
  from the scheduler.

Ticks are idempotent per local date: the journal records which ticks fired
for which date, so a restart never fires the same check twice and never
skips one that is still due (see 4).

### 2. The decision, in its own process, with a deadline

On entering PLANNING or PRE_FLIGHT the conductor runs
`python -m control.night_plan --refresh` exactly as the shadow planner does
now (worker thread, subprocess, `PLAN_TIMEOUT_S` 600 s, never under the
machine lock). `--refresh` first runs `instructions.calc_and_store_hours_above_horizon()`,
the queue-display refresh the noon check does today, so the decision reads
the same inputs it always did.

| Outcome | Noon (PLANNING) | Sunset (PRE_FLIGHT) |
|---|---|---|
| will_image | `PLAN_GOOD` → ARMED | no event: stays PRE_FLIGHT; the `good` record authorises the launch |
| not enough hours | `PLAN_BAD` → IDLE_DAY | `WEATHER_BAD` → IDLE_DAY (row exists) |
| crash / timeout | `PLAN_FAILED` (new) → IDLE_DAY, one retry after 5 min, then a Pushover push from the conductor itself | same, then IDLE_DAY: no night |

The full plan (best, good hours, best_start, slots with start/end/hours) is
the payload of the journal entry. That entry is the night plan; nothing else
holds it.

**The 10-15 hang becomes a `PLAN_FAILED` at 12:10 with a push**, from the
process that was never at risk of hanging.

### 3. The scheduler carries it out

With the flag on, the scheduler stops ranking. Its loop becomes:

1. Wait for a plan dated today: poll `GET /v1/plan` (new endpoint: the
   latest `NIGHT_PLAN` record per check
   for today, with its journal seq) every 30 s.
2. On `PLAN_GOOD`: everything the noon check does after deciding, unchanged
   (`tonight` post, calendar stat, sequence generation from the plan's
   slots, `_push_imaging_plan`, site picture). Slots are rebuilt from the
   plan JSON into `control.slot_plan` slots, never re-planned.
3. On `PLAN_BAD` / `PLAN_FAILED`: the skip posts it makes today.
4. At sunset, on a `good` sunset record: the existing signature comparison
   against the sequence on disk (regenerate on any difference), then
   `image_cmd(["", "image!!", "1"])` as today. On `WEATHER_BAD`: the
   sunset skip posts.

The executor never decides whether. If the conductor is unreachable it
**does not fall back to planning itself**: no plan means no night, a push,
and an entry in the log (ADR 0013's "unreachable refuses"; skipping a night
is the safe direction). The chat and Pushover side effects stay in the
scheduler process, where they already are, so the conductor gains no chat
dependency (ADR 0008).

### 4. Restarts

* **Conductor restart.** `_recover()` replays the journal: the day's ticks
  and plan entries come back with it. ARMED with today's plan is resumed
  from the PLAN_GOOD payload, not from `scheduler_state.json`. A tick that
  is due and has not fired for today fires once (12:00–sunset with no noon
  plan → plan now, the job `_initial_state()` does today). Mid-night states
  recover exactly as they do now; the clock offers nothing mid-night.
* **Scheduler restart.** It reads today's plan from `GET /v1/plan` and
  resumes the step it was at (sequence already generated when the plan's
  signature matches the file on disk). It never re-plans, so an afternoon
  restart cannot change the night (the 09-13 m33 loss).

### 5. What is not in this step

* `scheduler_state.json`, `imaging.txt`, `mode.txt`, `safety.txt` and MQTT
  stay; the scheduler keeps writing `scheduler_state.json` so its readers
  (social server `status`, morning_check, dusk_check, the shadow) see no
  change. Retiring them is step 3.
* No mid-night re-plan (`REPLAN_REQUESTED` stays unused; Phase 7a).
* Manual mode is unchanged: the plan is made and announced, and the
  executor does not launch (`get_mode() != "auto"`), as today.

## Rollout

1. **Build, flag off.** Clock thread, `--refresh`, the new outcomes, `GET
   /v1/plan`, executor path in the scheduler, and an operator `replan`
   command (re-runs the noon decision after a priority change: the "after-
   noon priority change needs a manual regen" chore, done properly). Tests:
   machine rows for PLAN_FAILED and the daytime REPLAN_REQUESTED; the clock's idempotence
   across restarts at 11:59, 12:01, 15:00, sunset+1 and 23:00; the executor
   rebuilding slots and producing the same sequence the in-process path
   does for the same plan; unreachable conductor → no night.
2. **Shadow the other way, flag on, one day.** The conductor decides; the
   scheduler ALSO ranks in-process, does not act on it, and journals
   `EXECUTOR_PLAN_DIFF` when its answer differs. Same evidence as step 1,
   pointing the other way.
3. **Supervised live night** with the flag on: no operator touch from noon
   to close (the Phase 3 criterion).
4. **Mid-night conductor restart drill**, during SLOT_IMAGING, away from
   any roof or mount ask: the conductor comes back to SLOT_IMAGING, the run
   carries on, the close at the end asks and is answered.
5. Remove the in-process decision path from the scheduler.

## Consequences

* A hang or a crash in the ranking becomes a journaled `PLAN_FAILED` and a
  push, within the timeout, instead of silence.
* One process decides each night, once per check; restarts resume the
  decision instead of making a new one.
* The conductor gains a clock and a planning subprocess; it still actuates
  nothing, imports no astronomy stack in-process, and has no chat
  dependency.
* The scheduler shrinks to waiting, generating and launching: the shape the
  step-3 retirement of its state files needs.

## As built (2026-10-16, rollout step 1)

* **The record is one note, `NIGHT_PLAN`**: {date, check, outcome good |
  bad | failed | skipped | missed, plan, why, trigger clock | replan |
  recovery, attempt}. The latest per check per date is the plan, served by
  `GET /v1/plan?date=`. The sunset confirmation is that record; it needed
  no machine event (`PLAN_CONFIRMED` was dropped from the design). Machine
  events follow the record where the table has a row: `PLAN_FAILED` (new;
  PLANNING and PRE_FLIGHT → IDLE_DAY) and `REPLAN_REQUESTED` from IDLE_DAY
  and ARMED → PLANNING.
* **`iris/conductor/clock.py`**: `due()` (pure: which check is owed, from
  the journal) and `NightDriver` (15 s loop, one planning subprocess at a
  time, retry once after 5 min, push on the second failure). A hold at noon
  records `skipped`; a daytime run still on the hardware at 12:00 is left
  alone and planned anyway; a leftover mid-night state is re-seated, as the
  shadow did at the scheduler's noon.
* **The scheduler** (`driven_cycle` / `_driven_main`): fetches the record,
  carries it out, calls the Prefect tasks' plain functions (no flow, so no
  temporary Prefect server), and with `conductor.executor_cross_check`
  also ranks in-process in a daemon thread, journaling
  `EXECUTOR_PLAN_MATCH` / `EXECUTOR_PLAN_DIFF` (and posting the sunset
  grid). The shadow stops translating the scheduler's checks into machine
  events while the flag is on.
* **`replan`** in the chat: refused before noon, after the sunset check,
  or with the flag off.
* Tests: tests/test_night_driver.py (a day on a fake clock, restarts,
  failures, holds, replans), tests/test_plan_executor.py (the scheduler
  side, full stack, skipped on CI), tests/test_plan_api.py, a shadow test.

## Owner's answers (2026-10-16)

1. **A failed plan** is retried once, 5 minutes after the failure; a second
   failure is `PLAN_FAILED`, no night, and a push.
2. **An unreachable conductor** means no night and a push. The scheduler
   never plans for itself.
3. **`replan`** is in this step.
