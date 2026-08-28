# Iris Architecture

*How the observatory is structured, maintained alongside the code. The
migration plan that produced this shape is `ARCHITECTURE_PLAN.md`; the
decisions behind it are in `adr/`. If this document disagrees with the code,
the code is wrong or this is — fix one.*

## The four services

| Service | Runs as | Owns |
|---|---|---|
| **iris-conductor** | daemon, supervised by `end_points/start_srt.py` | The Night machine, the Target registry, the journal, all safety guards; (from Phase 2) roof, mount, power, dehumidifier, capture. HTTP **:8096** |
| **iris-chat** | daemon (FastAPI, **:8095**) | The operator console: commands, job cards, Pushover, rendering journal events. Owns no observatory state |
| **iris-publish** | Windows Scheduled Tasks (5-min publishers, morning jobs) | `/srv/iris-live` content via the one scp transport; site staging; (from Phase 4) automatic gallery publication |
| **iris-n2n** | cron on the Spark (05:00/06:00 rsync, 07:00 render) | Training/inference, morning stack+denoise, render reporting |

**Migration status:** the conductor currently runs in **shadow mode** — it
watches the legacy state files and iris.log, synthesizes events, and builds
the journal; it commands nothing. Authority transfers per the plan's phases,
each gated by clean shadow nights.

## The machines

Both machines are **data**: transition tables in `iris/core/machine.py` and
`iris/core/target_machine.py`, stepped by a small pure interpreter. The
diagrams are generated from the tables (`scripts/gen_state_diagrams.py`);
CI fails if they are stale.

- **Night machine** — [night_machine.mmd](night_machine.mmd). One night of
  operation: planning → arming → roof open → prelude → slots (multi-target) →
  flats → parking → roof close → shutdown, plus the holds (`SAFE_HOLD`,
  `FAULT_ROOF_UNKNOWN`, `ESTOP`) that only operator events exit.
- **Target machine** — [target_machine.mmd](target_machine.mmd). One DSO's
  life: `WISHED → QUEUED → ACQUIRING → CONVERGED → RENDERING → PUBLISHED →
  RETIRED`. `CONVERGED` is the auto-stop (the planner never schedules past
  `ACQUIRING`); the path to `PUBLISHED` contains no operator event — that is
  the auto-publish, and a test asserts it.

**The safety invariants live in `iris/core/guards.py` and nowhere else:**

- *Invariant A* — the roof moves only when the scope is confirmed parked by
  **both** sensors (vision AND PWI4) and the roof position is known. The guard
  sits on the `PARKING → CLOSING_ROOF` and `PRE_FLIGHT → OPENING_ROOF` edges.
- *Invariant B* — the mount moves only under a confirmed-open roof.

Guards are pure functions of a `SensorSnapshot` with three-valued readings
(`CONFIRMED/DENIED/UNKNOWN` — a sensor declining to answer refuses motion).
The tests sweep the entire snapshot space (1,944 worlds) and assert the
invariants over all of them; five months of recorded real nights replay
through the table in CI (`tests/replay/`).

## The journal

Append-only JSONL, one file per day, `local/journal/`; monotonic `seq` across
days; fsync per line; replay tolerates a crash-truncated tail. It is the
system of record and the primary backup target.

```json
{ "seq": 4124, "ts": "…", "kind": "transition|target|rejected|note",
  "event": "NINA_SLOT_DONE", "source": "nina|chat|vision|watchdog|conductor|operator|spark|publish|shadow",
  "from": "SLOT_IMAGING", "to": "FLATS", "guard": null, "data": {} }
```

`rejected` entries carry the guard's refusal — the audit trail of every time
a guard said no. `note` entries are annotations (the eventual home of every
`post_social_message`).

## The API (conductor, :8096)

```
GET  /v1/state              current state + context
POST /v1/events             offer an event; 200 accepted / 409 with guard reason
GET  /v1/journal?since=SEQ  page of entries
GET  /v1/journal/stream     SSE of new entries
GET  /v1/targets[/dso]      the Target registry
```

No "request transition" endpoint exists: commands, sensor reports and capture
signals are all just events; the machine decides.

## Deploy

Push to `main` → CI (tests + diagram staleness) → on green, CI fast-forwards
**`release`** → the observatory's boot pull and the `update` command do
`git fetch origin release && git merge --ff-only origin/release`. A red main
means the observatory keeps running yesterday's green. Emergency bypass:
`git push origin <sha>:release`.

## Verification culture

- Every deployed `.py` file must parse (`tests/test_compile.py`) — the 4 a.m. test.
- Machine changes are table diffs, exhaustively stepped and property-checked.
- Recorded history replays in CI; live nights are judged each morning by
  `apps/shadow_report.py` (verdict `CLEAN`/`DIVERGED` posted to the webchat).
- Nothing gains hardware authority without clean shadow nights and a
  supervised daytime drill (ADR 0005).
