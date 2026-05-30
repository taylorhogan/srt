# Job-Based UI for the SRT Telescope Chatbot — Design & Integration Spec

This document specifies a UI interaction model for the SRT observatory chatbot, intended as a
reference for integration into the real application. A working prototype exists in
`TelescopeControl.jsx`; that file demonstrates the model with simulated jobs. **This document is
the source of truth for the intent.** Where the prototype and this document disagree, follow this
document — the prototype made some choices for demo convenience (e.g. `setTimeout` job simulation)
that must not survive into production.

---

## 1. The Problem

The chatbot controls a robotic observatory (CDK17 + QHY600M + iOptron L500, driven by the `srt`
pipeline). The user issues natural-language commands; the system responds. Two properties of the
domain break the standard linear-chat UI:

1. **Latency varies by orders of magnitude.** A "sky status" query returns in ~2 seconds. A slew
   takes ~15 seconds. An imaging run or a variable-star search across the FITS archive can take 20+
   minutes.

2. **Commands run concurrently.** The user can issue several long-running commands. Their status
   updates and log messages then arrive interleaved. In a linear chat transcript, the user cannot
   tell which log line belongs to which command. This is the core failure of the current design.

The fix is a conceptual shift: **commands are jobs, not chat messages.** A job has its own
lifecycle, its own log stream, and its own result. The UI is organized around jobs, not around a
single linear conversation.

---

## 2. Mental Model

- **Command** — the natural-language string the user submits ("image NGC 891 in luminance for an
  hour").
- **Job** — the unit of work the backend creates in response to a command. Has a stable `job_id`,
  a lifecycle (state machine below), an append-only log, optional progress, and an optional result
  payload on completion.
- **The chat is not the primary surface.** The primary surface is a panel of job cards. The input
  bar that accepts commands is persistent and always at the bottom (thumb-reachable on mobile).
- A job card is a **live, self-updating summary**. Tapping it opens a **detail sheet** with that
  job's full, isolated log and result.

The single most important principle: **a user must be able to glance at the screen and know the
status of every command, and must never see two jobs' logs intermixed in the same scroll.**

---

## 3. Job State Machine

```
                 ┌──────────┐
   submit ─────► │  QUEUED  │
                 └────┬─────┘
                      │ backend starts work
                      ▼
                 ┌──────────┐   user cancels / backend fails
                 │ RUNNING  │ ──────────────┐
                 └────┬─────┘                │
        backend done  │                      ▼
                      ▼                 ┌──────────┐
                 ┌──────────┐           │  ERROR   │
                 │   DONE    │          │(or CANCELLED)
                 └──────────┘           └──────────┘
```

States:

| State       | Meaning                                                          | Terminal? |
|-------------|------------------------------------------------------------------|-----------|
| `QUEUED`    | Command accepted, work not yet started (e.g. waiting on mount).   | no        |
| `RUNNING`   | Work actively executing. Progress and log lines stream in.        | no        |
| `DONE`      | Completed successfully. Result payload available.                 | yes       |
| `ERROR`     | Failed. Log should contain the failure reason.                    | yes       |
| `CANCELLED` | User cancelled. Treat as a terminal sub-state of ERROR visually, but keep distinct semantically so history/analytics can tell them apart. | yes |

Notes:

- The prototype collapses `CANCELLED` into `ERROR`. In production, keep them distinct in the data
  model; they may render with the same color but should log and report differently.
- A `QUEUED → CANCELLED` transition is valid (cancel before the mount frees up).
- Terminal jobs never transition again. The card persists (see §6, persistence).

---

## 4. Concurrency & Resource Constraints

Not all jobs are freely concurrent. The UI must reflect physical reality:

- **The mount is a single exclusive resource.** Only one slew can execute at a time. A second slew
  command issued while a slew is `RUNNING` must enter `QUEUED` and visibly wait. The backend owns
  this scheduling; the UI must *show* it (a `QUEUED` slew card with a "waiting for mount" hint).
- **Imaging implies the mount is committed.** You cannot slew elsewhere mid-exposure without
  aborting the imaging job. The backend should reject or queue conflicting commands; the UI should
  surface the conflict rather than silently interleaving.
- **Archive/data jobs are freely concurrent.** Variable-star searches, photometry, and other
  read-only operations over the FITS archive (likely running on the DGX Spark) do not touch the
  mount and can run in parallel with each other and with a slew/imaging job.
- **Quick queries (sky status, weather, focus position read-back)** are effectively instantaneous
  and concurrent with everything.

A useful classification for the backend to send per job (so the UI can reason about it):
`resource_class ∈ { mount_exclusive, archive, quick }`. The UI can use this to render a
"waiting for mount" reason on queued jobs, and potentially to offer a queue/priority view (§9).

---

## 5. Backend Contract (to be finalized)

The prototype fakes everything with `setTimeout`. **Delete all of that on integration.** Real jobs
are created when the backend accepts a command and are updated asynchronously over a push transport.

### 5.1 Transport

Recommended: a single **websocket** (or SSE stream) carrying job events keyed by `job_id`. SSE is
simpler if the command submission can stay a normal POST and only *updates* need to stream; a
websocket is better if you want bidirectional control (e.g. cancel over the same channel).
Given Tailscale-connected machines and a long-lived session, a websocket from the phone client to
the `srt` service is the natural fit. Confirm against the actual `srt` implementation.

### 5.2 Command submission

```
POST /commands
  { "text": "image NGC 891 in luminance for 1 hour" }
→ 202 Accepted
  { "job_id": "job_8f3a", "status": "QUEUED", "title": "Imaging NGC 891",
    "resource_class": "mount_exclusive", "created_at": "<iso8601>" }
```

The backend is responsible for parsing the natural-language command into a job, including deriving a
human-readable `title`. (If parsing is itself LLM-mediated and slow, the POST can return a
provisional title and refine it via a later event.)

### 5.3 Job events (over websocket/SSE)

Every event carries `job_id`. Proposed event shapes — **adapt names to match the real `srt` /
Prefect status vocabulary**:

```jsonc
// state transition
{ "type": "status",   "job_id": "job_8f3a", "status": "RUNNING", "at": "<iso8601>" }

// log line (append-only)
{ "type": "log",      "job_id": "job_8f3a", "at": "<iso8601>",
  "level": "info", "text": "Filter wheel → Luminance" }

// progress (optional; 0–100 or a fraction)
{ "type": "progress", "job_id": "job_8f3a", "value": 0.42 }

// completion with result payload
{ "type": "result",   "job_id": "job_8f3a", "status": "DONE", "at": "<iso8601>",
  "result": { "kind": "image", "summary": "12×300s L stacked",
              "thumb_url": "/results/job_8f3a/thumb.jpg", "label": "L · 60 min" } }
```

Result `kind` values seen so far, with their payloads:

- `status` — `{ "summary": string }`. Short textual confirmation (focus locked, on target, sky clear).
- `table` — `{ "summary": string, "columns": [...], "rows": [[...], ...] }`. e.g. the variable-star
  detection table, or an Abell-2151-style galaxy table (NGC/IC id, type, distance, etc.).
- `image` — `{ "summary": string, "thumb_url": string, "full_url"?: string, "label": string }`.
  Stacked frame preview; tapping the thumb could open the full-resolution result.

Keep `kind` open-ended; new job types will add new result kinds.

### 5.4 Prefect mapping

The `srt` pipeline uses Prefect DAG orchestration. Map Prefect flow/task run states onto the UI
state machine at integration time. A likely mapping (verify against actual states):

| Prefect state           | UI state    |
|-------------------------|-------------|
| `Scheduled` / `Pending` | `QUEUED`    |
| `Running`               | `RUNNING`   |
| `Completed`             | `DONE`      |
| `Failed` / `Crashed`    | `ERROR`     |
| `Cancelled`             | `CANCELLED` |

Prefect log records and task state-change events are the natural source for the `log` and `status`
events. If a command maps to a Prefect *flow* composed of several tasks, the per-task transitions
make excellent log lines ("Subframe selection complete", "WBPP stacking started").

### 5.5 Cancellation

```
POST /commands/{job_id}/cancel   → 202
```
Backend cancels the underlying Prefect run and emits a terminal `status: CANCELLED` event. The UI
optimistically disables the cancel button on click but waits for the terminal event to set final
state.

---

## 6. UI Structure

Three regions, top to bottom:

1. **Header** — observatory identity, a live count of active jobs ("2 jobs active" / "Idle"), and a
   clock. Optional: a global condition indicator (sky brightness / safe-to-observe) since that's
   already monitored in `srt`.

2. **Jobs panel** (scrollable, the bulk of the screen) — a vertical stack of **job cards**, newest
   on top. Each card shows:
   - Title ("Imaging NGC 891") and a left edge color-coded by job kind.
   - Status badge (`QUEUED`/`RUNNING`/`DONE`/`ERROR`), pulsing when running.
   - The **most recent log line only** (truncated) — a live "what's happening now" line.
   - A progress bar (or indeterminate shimmer if progress is unknown) and elapsed time.
   - An **unread-log-count badge** when new log lines have arrived since the user last viewed the
     job's detail (see §7 — this is the key mobile affordance).
   - An inline Cancel control while running.

3. **Input bar** (persistent, bottom) — a single text field + send button. Always reachable. New
   commands always originate here. The chat history does *not* live here; it has been replaced by
   the jobs panel.

**Detail sheet** — tapping a card opens a bottom sheet (slides up) showing **only that job's** full
chronological log, timestamped per line, plus the result block when terminal. Cancel (if running)
or Close at the foot. This is where log isolation is enforced: one job's logs, never intermixed.

---

## 7. The Unread Counter (most important mobile affordance)

On a phone, jobs cannot all shout for attention. The design rule:

- Status updates **never steal focus**. No modal, no auto-scroll, no toast that interrupts.
- Instead, each card accumulates an **unread count** of log lines that arrived since the user last
  opened that job's detail sheet. The count renders as a small badge on the card.
- Opening the detail sheet resets that job's unread count to zero.
- A glance at the panel then answers "which job needs my attention?" without any job hijacking the
  screen.

This is what lets multiple 20-minute jobs coexist calmly. Preserve this behavior precisely.

---

## 8. Visual / Interaction Details (from the prototype)

These are deliberate and worth keeping, though the visual theme is open to change:

- **Dark "control room" theme.** The app is used at night during observing sessions; a dark UI
  avoids destroying dark adaptation. If anything, consider an even-redder night-vision mode toggle.
- **Monospace for logs, telemetry, and status badges**; a clean sans for titles. Logs are technical
  and align better monospaced.
- **Color-coded job kinds** on the card's left edge (slew / search / image / focus / quick) for
  fast visual scanning of a mixed panel.
- **Pulsing dot** on a `RUNNING` badge — ambient "alive" signal without motion that distracts.
- **Cards persist after completion**; they fold to a calmer state but don't vanish. The user can
  still open a finished job to retrieve its result.
- **Active-state scale on tap**, bottom-sheet slide-up animation — standard mobile tactility.

---

## 9. Recommended Extensions (not in the prototype)

1. **Persistent job history across sessions.** Currently jobs live only in client memory. A
   completed imaging run should be retrievable next session, not lost to a scroll. Back this with a
   `GET /jobs?since=...` endpoint and/or local persistence, so re-opening the app rehydrates recent
   and in-flight jobs. This matters a lot for 20-minute jobs you walk away from.

2. **Queue / priority view.** Because the mount is exclusive (§4), there can be a meaningful queue of
   `QUEUED` mount jobs. A small queue view (or ordering within the panel) showing what will run next
   and letting the user reorder/cancel pending work would be valuable.

3. **Reconnection handling.** On a phone the websocket *will* drop. On reconnect, re-fetch current
   job state (`GET /jobs`) and reconcile, rather than assuming the in-memory cards are current. Show
   a subtle "reconnecting…" indicator; never silently show stale `RUNNING` cards for jobs that have
   since finished.

4. **Result deep-views.** An `image` result thumbnail should open the full-resolution stacked frame;
   a `table` result (e.g. the Abell 2151 galaxy table) could become sortable/exportable. The detail
   sheet is the natural host.

5. **Night-vision mode.** A red-dominant palette toggle for use at the eyepiece / in the dome.

6. **Notifications.** For 20-minute jobs, an optional push/local notification on terminal state lets
   the user leave the app and come back. Pair with the persistence in (1).

---

## 10. Integration Checklist for Claude Code

When integrating into the real app:

- [ ] **Remove the entire `setTimeout`-based simulation** (`COMMAND_LIBRARY`, `runJob`, fake log
      schedules). It exists only to demo the model.
- [ ] Replace job creation with the real `POST /commands` call; seed a card from the `202` response.
- [ ] Subscribe to the websocket/SSE stream; route incoming events to jobs **by `job_id`** (never by
      array position).
- [ ] Implement the reducer so `status`, `log`, `progress`, and `result` events each patch the right
      job immutably. Log is append-only.
- [ ] Map Prefect states → UI states (§5.4); confirm the real state vocabulary first.
- [ ] Wire cancellation to `POST /commands/{job_id}/cancel`; set terminal state from the event, not
      optimistically.
- [ ] Implement unread-count logic exactly as in §7 (increment on log/result when the job's detail
      sheet is not open; reset on open).
- [ ] Add `GET /jobs` rehydration on app load and on websocket reconnect (§9.1, §9.3).
- [ ] Confirm the `resource_class` field and render "waiting for mount" on queued mount-exclusive
      jobs.
- [ ] Decide result-kind rendering (`status`, `table`, `image`) and how thumbnails resolve against
      the `srt` results directory / URL scheme.
- [ ] Keep the interaction invariants: input bar always reachable; no focus-stealing updates; one
      job's logs never intermixed with another's.

---

## 11. Open Questions to Resolve with the Real Backend

- What is the actual transport in `srt` today (websocket, SSE, polling)? Match it rather than
  imposing a new one if one exists.
- What are the real Prefect flow/task names, so log lines read meaningfully?
- How are command strings parsed into jobs — rule-based, or LLM-mediated? If the latter and it's
  slow, the provisional-title-then-refine flow (§5.2) matters.
- What is the URL/path scheme for result artifacts (stacked FITS/JPEG previews, detection tables) so
  the detail sheet can link to full-resolution views?
- Should multiple clients (phone + desktop) observe the same job stream simultaneously? If so, the
  server is the single source of truth and all clients rehydrate from `GET /jobs`.
