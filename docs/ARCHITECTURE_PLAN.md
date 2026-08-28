# Iris: One Repo, Four Services, Two Machines

*A restructuring plan for review — v2, 2026-08-28. Adds: target lifecycle with auto-stop at
convergence and auto-publish; multi-target nights; global state on the website; website
ascendant over webchat.*

## Context

Iris works, but its correctness lives in fragile places: the program counter of a Prefect
flow (persistence disabled), a 5,517-line file that is simultaneously the safety system and
the chat bot, and ~15 state files with no reader/writer discipline. Measured facts:

- The scheduler's `State` enum is not even a variable — the real state is a string in a dict
  plus a program counter. `scheduler_state.json` is **write-only**; nothing reads it back.
- A restart at 20:00 with a planned night **loses the night**. The restart also rewrites
  `safety.txt` to `USER SAFE` (re-arming a deliberately cleared flag) and `mode.txt` to
  `manual` (silently stopping auto imaging). Live bugs today.
- Invariant A ("roof moves only when parked") is enforced in **six places**; `end.py`'s
  close path bypasses every gate and calls `toggle_roof()` directly. Invariant B ("mount
  moves only when roof open") is enforced **once**.
- "Is imaging happening" has six representations; "tonight's DSO" five; "is a target done"
  lives in two places (`my_instructions.json` status vs `convergence.json`) with a written
  rule that one must never overwrite the other.
- **There are no tests.**

The goal: Iris as a formal state machine first — stellar open-source code that keeps
choosing targets, imaging them to a measured goodness, publishing the finished pictures,
and showing the world what it is doing, for months unattended, operable by an heir from its
own documents.

## Decisions already made (owner's answers, locked)

| Decision | Choice |
|---|---|
| Topology | One repo; four systems as **separate processes/services** with a defined contract |
| State machine | **Authoritative machine + guards**; roof/mount are owned resources |
| Formality | Machine **as data**; diagram *generated*; every transition + guard exhaustively tested; no FSM framework, no TLA+ |
| Interconnect | Append-only **journal** + **small HTTP API**; files stop being interfaces; MQTT retired |
| Capture | **Capture behind an interface**; NINA first implementation; autofocus stays NINA's longest |
| Migration | **Strangler fig** — never a lost night; each cutover shadowed one clean night first |
| Autonomy | **Months unattended, heir-operable**; self-healing bounded to what half-exists; the rest is runbooks |
| Legacy bar | Tests + CI deploy gate · architecture docs · succession audit · public-repo hygiene |
| **New (v2)** | **Auto-stop at goodness, auto-publish** the denoised picture · **multiple targets per night** (small horizon: re-plan as targets rise/set) · **global state on the website** · over time **website grows, webchat shrinks** |

---

## 1. Target architecture

### The four services

| Service | Process model | Owns exclusively |
|---|---|---|
| **iris-conductor** | Long-running daemon, supervised by `start_srt.py`; **absorbs the scheduler** | The Night machine · the **Target registry** (per-DSO lifecycle) · the journal (`local/journal/`) · roof, mount, power, dehumidifier · the capture interface · vision sensors as inputs · **the night planner (multi-slot)**. HTTP **:8096** |
| **iris-chat** | Long-running daemon (existing FastAPI, :8095) | The **operator console**: command parsing, job cards, Pushover, rendering journal events. Owns no observatory state. Deliberately *not* grown further — see "website ascendant" below |
| **iris-publish** | Stays **cron/Scheduled Tasks** | `/srv/iris-live` content · the one scp transport (`live_push`, timeouts) · **global-state + target-state publication** · **automatic gallery publication** to the site repo |
| **iris-n2n** | Stays **cron on the Spark** | Training/inference, morning stack+denoise, **render-done reporting back to the conductor** |

**Website ascendant (design principle, new):** the webchat remains the *operator's* console
— safety commands, diagnostics, cancels. Everything a *viewer* wants migrates to
irisscience.org: current global state, per-target progress, finished pictures. New features
default to the website; chat gets them only when an operator needs them. Concretely, the
site's Live tab gains the conductor's state and a target board (below), both fed from the
same journal the chat renders — one source, two consoles, with the site as the one that
grows.

### The conductor API — five endpoints

```
GET  /v1/state              → { state, since, seq, night_plan: [slots...],
                                context: { slot, dso, roof, mount, safety, mode } }
POST /v1/events             → 200 { accepted, seq, state } | 409 { accepted:false, guard:"reason" }
GET  /v1/journal?since=SEQ  → { entries: [...], head }
GET  /v1/journal/stream     → SSE (replaces MQTT and cross-process message_bus)
GET  /v1/targets[/<dso>]    → the Target registry: every DSO with lifecycle state,
                                frames/filters accumulated, convergence numbers, render/publish record
```

No "request transition" endpoint: operator commands, sensor reports, NINA signals, and
Spark reports are all just events. The machine transitions or refuses; **refusals are
journaled** with the guard's reason — the audit trail of every time a guard saved the roof
comes free.

### The journal (append-only JSONL, one file per day, `local/journal/`)

```json
{ "seq": 4124, "ts": "2026-08-28T02:14:07-07:00",
  "kind": "transition | target | rejected | note",
  "event": "NINA_SLOT_DONE", "source": "nina|chat|vision|watchdog|conductor|operator|spark|publish",
  "from": "SLOT_IMAGING", "to": "SLOT_SETUP", "guard": null,
  "data": { "slot": 2, "dso": "sh2-129" } }
```

`kind:"target"` records Target-machine transitions (same file — one timeline for the whole
system). `kind:"note"` is how `post_social_message` survives. `seq` monotonic across days;
replay = read yesterday + today. `local/journal/` becomes the primary OneDrive backup target.

---

## 2. The two machines

### 2a. The Night machine (~15 states)

Merges today's scheduler skeleton with `ImagingState`, generalized from "one target per
night" to **a plan of slots**. `PLANNING` now emits a *night plan*: an ordered list of
slots `[(dso, start, end, filters), ...]` fitted to the tree-line horizon (`my.hrz`) —
with a small horizon, a target that sets at 01:00 hands the sky to one that rises at 01:30.

```
IDLE_DAY        nothing planned; roof closed, mount parked
PLANNING        noon check → night plan of 1..N slots (weather, horizon windows,
                Target-registry need — CONVERGED targets are excluded HERE: auto-stop)
ARMED           plan exists — SURVIVES RESTART (fixes the 20:00-restart bug)
PRE_FLIGHT      pre-sunset re-check: weather, safety armed, park confirmed
OPENING_ROOF    relay toggled; awaiting vision confirmation (stall-watchdog bounded)
PRELUDE         once per night: cooling, initial focus (NINA)
SLOT_SETUP      slew/sequence launch for the current slot
SLOT_IMAGING    main sequence for this slot; on NINA_SLOT_DONE or SLOT_WINDOW_END →
                next slot (back to SLOT_SETUP) or FLATS if plan exhausted
FLATS           once per night
CLOSING_ROOF    park confirmed → relay → awaiting closed confirmation
SHUTDOWN        dehumidifier, summary, Spark handoff marker
NIGHT_DONE      terminal → IDLE_DAY at next noon
SAFE_HOLD       operator cleared safety; refuses all motion; ONLY an operator event exits
FAULT_ROOF_UNKNOWN  post-stall/vision contradiction; persisted; exits via OPERATOR_RESOLVE
ESTOP           emergency stop ran; mount also assumed unknown
```

- Mid-night **re-planning is an event**, not a state: `WEATHER_BAD`, `SLOT_UNPRODUCTIVE`
  (cloud eating a slot), or a mid-night `TARGET_CONVERGED` posts `REPLAN_REQUESTED`; the
  planner re-fits the remaining night and the machine continues in `SLOT_SETUP`.
- Degraded modes stay context flags (`context.degraded=["no_camera"]`), not states.
- `ImagingState`'s `DONE_*` values become events; `imaging.txt` and the duplicated enum in
  `set_imaging_state.bat` die.

**The honest cost of multi-slot (called out, not hidden):** today NINA runs ONE generated
sequence per night (prelude+target+flats+end). Multi-slot requires restructuring sequence
generation into prelude-once / per-slot target sequence / flats-once, with the conductor
launching each slot through the capture interface. This is the largest *functional* (not
structural) build in the plan and is sequenced late (Phase 7), after the capture seam
exists — single-slot nights run through the identical machinery first (a one-slot plan is
just today's behavior).

### 2b. The Target machine (new, ~7 states) — auto-stop and auto-publish

Per-DSO lifecycle, persisted in the **Target registry** (`local/targets/<dso>.json` +
journal events). This *unifies* today's split brain: `my_instructions.json` status
(waiting/completed) and `convergence.json` (is it actually done?) become one record with
one owner.

```
WISHED       requested (by operator, or someday by a selection policy)
QUEUED       eligible: coordinates resolved, filter plan set
ACQUIRING    accumulating frames across nights (slot allocation favors need)
CONVERGED    goodness reached — per-filter convergence (existing machinery:
             tail_slope < 0.4, RMSE < 25%, min 16 frames) → planner stops
             scheduling it. THE AUTO-STOP.
RENDERING    Spark render requested/underway (stack + denoise + compose)
PUBLISHED    the denoised picture is on irisscience.org — automatically
RETIRED      archived; can be re-opened (new filters, better sky) by operator event
```

Events: `FRAMES_ADDED` (nightly, from frame accounting), `CONVERGENCE_EVAL` (the existing
`snr`/convergence math, run post-night), `RENDER_DONE` (posted by the Spark 07:00 job —
today it only Pushovers; it gains one `POST /v1/events`), `PUBLISH_DONE` (posted by
iris-publish), `REOPENED` (operator).

**The auto-publish pipeline** (the "publish after I die" spine):
`CONVERGED → conductor marks render-wanted → Spark 07:00 sees it via GET /v1/targets,
renders, posts RENDER_DONE with product paths → iris-publish generates the gallery entry
(site_gen.py: gallery card + processed image, using the existing note-fragment/gallery
conventions) → commits and pushes to the taylorhogan.github.io repo → posts PUBLISH_DONE.`
Publication is journaled like everything else; a chat note announces it, but no human is in
the loop. (The existing manual science-note flow stays manual — prose is authored; gallery
entries are generated.)

### The machines as data

```python
# iris/core/machine.py — the whole machine IS this table; ditto target_machine.py
TRANSITIONS = [
    T("ARMED",        "PRE_SUNSET_TICK",   "PRE_FLIGHT"),
    T("PRE_FLIGHT",   "CHECKS_PASSED",     "OPENING_ROOF", guards=[safety_armed, mount_parked, roof_state_known]),
    T("SLOT_IMAGING", "NINA_SLOT_DONE",    "SLOT_SETUP",   guards=[slots_remaining]),
    T("SLOT_IMAGING", "NINA_SLOT_DONE",    "FLATS",        guards=[plan_exhausted]),
    T("*",            "ESTOP_REQUESTED",   "ESTOP"),
    ...
]
```

Guards are **pure functions of a `SensorSnapshot`** — trivially testable. Both machines'
Mermaid diagrams in `docs/` are generated from their tables; CI fails if stale.

### Subsuming the six enforcement sites

Invariant A gets exactly **one** implementation — guards built around today's
`confirm_roof_state()` (already the one successfully shared piece). Then:
- `open_roof`/`close_roof`/toggle → thin `POST /v1/events`; `_roof_move_blocked_reason`
  becomes the guard reason string; `_roof_lock` disappears (the conductor's single-threaded
  event loop serializes roof motion by construction).
- `end.py::do_main` → posts `NIGHT_END_REQUESTED`. **Critical property preserved:** `end.py`
  is launched by NINA's `end.bat`, so the close happens even if the brain died — it keeps a
  *last-resort fallback* (conductor unreachable → minimal park-confirm-then-toggle via the
  shared vision check + a FAULT marker ingested at next start). CI-tested forever.
- `cycle_roof.py` → 10-line client. `_emergency_stop_sequence` → the `ESTOP` action.
- Invariant B becomes a real guard on every mount action; `dither_now.py`'s exception
  becomes an explicit narrow guard.
- The stall watchdog cuts power then posts `ROOF_STALL` → `FAULT_ROOF_UNKNOWN`, journaled
  and **surviving restart** (today it evaporates).

The roof relay is a *toggle* (no direction): every roof action stays
sense → guard → toggle → confirm → journal, and `FAULT_ROOF_UNKNOWN` is the honest state
whenever sense and expectation disagree.

### Crash recovery: replay + reconciliation

On start: replay the journal to the last state, then reconcile against a fresh
`SensorSnapshot` (NINA presence, PWI4, vision). Agree → resume in place (a 20:00 restart
during `SLOT_IMAGING` resumes the slot; `ARMED` keeps the night). Disagree → `FAULT` with
the discrepancy journaled. `SAFE_HOLD` can only be exited by an operator event — restarting
can never re-arm safety, by construction. The Target registry replays the same way.

---

## 3. Package restructure (one repo)

```
srt/
├── iris/
│   ├── core/        machine.py · target_machine.py · guards.py · journal.py · api.py · events.py · snapshot.py
│   ├── conductor/   main.py · plan.py (noon-check math verbatim + NEW slot fitting) · targets.py (registry) · recovery.py · actions.py
│   ├── hardware/    roof.py (ONLY file touching the relay — CI-grepped) · mount.py · power.py · dehumidifier.py
│   ├── capture/     base.py (CaptureSession protocol) · nina.py (+ sequence gen, slot-ified in Phase 7) · later qhy.py
│   ├── vision/      safety.py (confirm_roof_state home) · kasa_state.py (shadow) · allsky_match.py
│   ├── chat/        server.py · commands/ (the ~39 *_cmd split by domain) · jobs.py
│   ├── publish/     live_push.py (the ONE scp transport, with timeouts) · skymap.py · monitors · site_gen.py (gallery auto-publish) · state_feed.py (global state → site)
│   ├── n2n/         today's nn/ (library untouched) · spark/ entrypoints (+ RENDER_DONE reporting)
│   ├── science/     stacking/ · photometry/ · fits_processing/ · transit/transient — MOVED, NOT REWRITTEN · fwhm.py (public home of _load_precomputed_fwhm_stars)
│   ├── astro/       iris_astronomy/
│   ├── common/      config · utils · notify.py
│   └── vendor/      kasa_local/ (excluded from lint/coverage)
├── apps/            thin entrypoints (~14) · lab/ (~50 experiments, never imported by iris/)
├── tests/           unit/ · fakes/ (kasa, shelly, pwi4, nina) · replay/ · acceptance/ (hand-run injection)
├── docs/            ARCHITECTURE.md · generated diagrams (both machines) · adr/ · HANDBOOK.md · SUCCESSION.md
├── local/           ALL runtime state, gitignored (journal/, targets/, logs)
└── site/            publishing staging (deploy target remains taylorhogan.github.io)
```

**God file dissolves:** safety cluster → `core/guards` + `hardware/roof` +
`conductor/actions`; `ImagingState` → events; ~39 `*_cmd` → `chat/commands/`;
`_load_precomputed_fwhm_stars` → `science/fwhm.py` public (kills both private-import
violations — mechanical, Phase 0).

**Cycles break by direction: doers emit; consoles render.** `post_social_message`
(320 sites) becomes `iris.common.notify.post(...)` → a `kind:"note"` journal event; chat
subscribes to `/v1/journal/stream` and renders; the website's live feed renders the same
stream. Dependency arrows all point inward; `core` never imports `chat` or `publish`.

### The website's new surface (fed by iris-publish, no new daemons)

- `/live/status.json` grows the conductor's `state` + `night_plan` (global state visible,
  as requested) — the existing 5-min publisher just adds fields from `GET /v1/state`.
- New `/live/targets.json`: the Target registry snapshot → a "target board" on the site
  (each DSO: lifecycle state, frames per filter, convergence progress bar, link to its
  published picture). This is the page that grows as the webchat shrinks.
- Auto-published gallery entries land in the site repo via `site_gen.py` (above).

---

## 4. Testing and the CI deploy gate

- **Exhaustive table-driven tests for BOTH machines**: every (state × event) pair asserted
  (transition, rejection, or ignore). Night ≈ 15×22, Target ≈ 7×8 — generated, cheap.
- **Guards property-style**: enumerate the `SensorSnapshot` space; assert *no snapshot with
  `parked != CONFIRMED` ever permits roof motion* — Invariant A as a test.
- **Fakes**: `FakeKasa`/`FakeShelly` (replay the real captured stall waveform), `FakePWI4`,
  `FakeNINA` (scripted event emission, including "dies mid-slot"). CI runs full simulated
  nights: plan → open → slot 1 → slot 2 → NINA dies → recover → close; plus a simulated
  *season*: frames accumulate → converge → render → publish (Target machine end-to-end
  against fakes).
- **Replay tests**: recorded real-night journals replay to the same terminal state;
  truncated-tail (crash mid-write) tolerated.
- **Stays hand-run**: the injection suites (science-against-sky) → `tests/acceptance/`.
- **Shadow-night harness**: `apps/shadow_report.py` diffs the journal against ground truth
  each morning of a cutover phase; posts itself to chat. "Clean night" = zero divergence.

**Deploy gate:** GitHub Actions on `main` (ruff + pytest; `lab/`, `vendor/`, `science/`
excluded from strict gates initially); a green-only job fast-forwards a **`release`**
branch; the observatory's daily pull changes one word to pull `release`. Red `main` =
observatory keeps running yesterday's green. Zero new infrastructure.

---

## 5. Strangler-fig phases

*Observe before you own; own the most dangerous thing (roof) with the smallest change;
absorb the brain; then grow the new capabilities on the new spine. The observatory images
throughout.*

| # | Phase | Size | Clean-night criterion |
|---|---|---|---|
| **0** | **Stop the bleeding.** Fix the restart path (never rewrite `safety.txt`/`mode.txt` on boot) — ship this week regardless. Runtime state → `local/`; delete `not_used_archive/`; `_load_precomputed_fwhm_stars` public; `live_push` timeouts into `live_skymap`. CI skeleton + `release` branch; flip the daily pull. | S | Any night — nothing behavioral changed except the bugs |
| **1** | **Journal + conductor in read-only shadow.** `iris/core` (both machine tables, guards, journal, API); conductor as third supervised process that **commands nothing** — tails existing files/NINA/vision, synthesizes events, builds the journal *and a shadow Target registry* derived from `my_instructions.json` + `convergence.json`. Machine tests + diagram generation land. | M | Journal timeline matches the real night exactly (shadow_report); registry matches the queue+convergence answers for every DSO |
| **2** | **Guards centralized; conductor owns the roof.** All six enforcement sites rewired through `POST /v1/events`; conductor authoritative *for the roof only*. `end.py` fallback added. `FAULT_ROOF_UNKNOWN` real and persistent. | M | Full night incl. open+close via the new path; supervised daytime `cycle_roof`; deliberate guard-rejection |
| **3** | **Scheduler absorbed; files retire; MQTT dies.** Prefect flow → transitions (noon-check math verbatim; **single-slot plans** — behavior identical to today). Replay = recovery; `ARMED` survives restart. NINA `.bat`s → `emit_event.bat`. Short-lived mirror release for legacy readers, then files + MQTT deleted. | **L** | **Two** decision-diffed shadow nights, then a supervised live night with zero touches, **plus a deliberate mid-night conductor restart that resumes correctly** |
| **4** | **Target machine live: auto-stop + auto-publish.** Registry becomes authoritative (queue + convergence unify); planner excludes CONVERGED (auto-stop — formalizing today's overlay hack); Spark posts `RENDER_DONE`; `site_gen.py` auto-generates + commits gallery entries (auto-publish); site gains `/live/targets.json` + global state. | M | One target driven WISHED→PUBLISHED with no human touch (a converged target re-rendered end-to-end); target board live on the site |
| **5** | **Chat decoupled; god file dismantled.** `notify.post` shim → 320-site migration; `*_cmd` → `chat/commands/`; cycles gone; `super_user_commands.py` deleted. | M | Normal night + full pass through the command vocabulary |
| **6** | **Capture interface.** `CaptureSession` protocol; NINA impl wraps sequence gen/launch/events/teardown. Autofocus stays NINA's. The seam multi-slot needs. | M | Normal night through the interface; FakeNINA nights in CI |
| **7** | **Multi-slot nights.** Slot-fitting planner (horizon windows × registry need × priorities); sequences restructured prelude-once/slot/flats-once; `REPLAN_REQUESTED` on weather or mid-night convergence. The largest functional build — lands last, on a proven spine. | **L** | A real two-target night: target A sets behind trees, machine re-slews, target B imaged; morning report shows both slots' frames correctly attributed |
| **8** | **Docs, succession, hygiene** — woven throughout, finalized here. ADRs written as decisions ship. | S | Done-definition audit against §6 |

---

## 6. The legacy artifacts

- **`docs/ARCHITECTURE.md`** — four services, both generated state diagrams (CI-enforced
  fresh), journal schema, API surface.
- **`docs/adr/`** — seeded with the decisions already made (one repo/four processes; two
  machines as data; journal + tiny API; capture interface; strangler fig; release-branch
  gate; toggle-relay sense-then-act doctrine; notes-through-journal; cron clients not
  daemons; **website-ascendant principle**; **auto-stop/auto-publish pipeline**).
- **`docs/HANDBOOK.md`** — per-state operations: resolving `FAULT_ROOF_UNKNOWN`, reading
  the journal, the command vocabulary, cold-start, editing NINA sequences,
  pinning/rolling back `release`, re-opening a RETIRED target.
- **`docs/SUCCESSION.md`** — three ledgers: *accounts & money* (domain, web host + Caddy,
  Tailscale — and what dies when each lapses, Pushover, GitHub, API keys); *physical acts*
  (collimation, filters, desiccant, roof lubrication, disks, camera reseating — frequency
  and symptoms of neglect); *autonomy gaps as runbooks* (successor alerting, per-hardware
  degraded behavior, who merges after the owner). Windows Scheduled Task definitions
  exported (`schtasks /query /xml`) into `docs/`.
- **Public hygiene:** `local/` fully gitignored; root PNGs/logs out; secrets audit; a README
  for the stranger who inherits it; the site's raw.githubusercontent hotlinks noted as a
  succession risk.

---

## 7. Risks — and what NOT to do

- **Do not rewrite science code** (`stacking/`, `photometry/`, transit/transient, `nn/`) —
  validated by years of sky; move, fix imports, stop. The injection suites are the net.
- **Do not split repos. Do not add a broker.** SSE off the journal (or honest polling) is
  enough for a handful of clients on one tailnet.
- **Do not touch NINA autofocus.** The capture interface exists so you never have to until
  you choose to.
- **Auto-publish must be un-embarrassable:** `site_gen.py` publishes only renders that pass
  the existing quality gates, and every publication is journaled + announced — an heir (or
  you) can retract with one revert. Start with gallery entries only; prose notes stay human.
- **Multi-slot is the riskiest functional change** — hence last, behind the capture seam,
  with single-slot nights proving the identical machinery first. A two-target night is
  *attempted* only after a deliberate one-slot night through the slot machinery.
- **Toggle relay during migration:** shadow conductor never actuates; cutover is one config
  flag; `iris/hardware/roof.py` is the only file touching the relay, enforced by a CI grep.
- **`end.py`'s fallback must never rot:** CI-tested against fakes forever.

## Verification (overall)

1. Per-phase clean-night criteria, each reported by `shadow_report` into webchat.
2. CI green = both machines' exhaustive cases + guard property tests + simulated nights
   (NINA-dies, conductor-restart, two-slot) + a simulated season (WISHED→PUBLISHED) +
   journal replay of recorded real nights.
3. Live drills: (a) kill the conductor mid-slot — restart resumes the slot; (b) `unsafe!`
   then reboot — `SAFE_HOLD` holds; (c) simulated stall — `FAULT_ROOF_UNKNOWN` persists and
   refuses motion until `OPERATOR_RESOLVE`; (d) a converged target reaches the site gallery
   with zero human touches.
4. Legacy bar: a stranger, given only `docs/`, can explain both state diagrams, operate the
   console, recover a fault, and list what breaks when each account lapses.

## Open items for the owner (not blockers)

- Naming: "conductor" is a placeholder.
- Phase 0's safety-bug patch is worth shipping this week regardless.
- The 6,685-line `index.html` will strain under the target board + gallery automation;
  restructuring it deserves its own plan when Phase 4 approaches (the "website ascendant"
  direction makes that worth doing properly).
- "Some goodness" is currently defined by the convergence thresholds (slope 0.4 / RMSE 25% /
  16 frames). If you want a different or richer definition of *done* (e.g., per-target
  goals, minimum integration per filter), that slots into `CONVERGENCE_EVAL` without
  touching the machine.
