# The Operator's Assistant — a local LLM that knows how to *use* Iris

*v1, 2026-09-03. Broad strokes agreed in conversation; the question list (§3) is
the first deliverable and gates everything after it.*

## 1. What this is, and is not

A locally hosted language model, reachable from the webchat like any other
command, that answers **operator** questions: what commands exist, what the
observatory is doing right now, what happened last night, what the manual says
to do next. Motivated by cost — the casual question traffic that currently goes
to a paid API is exactly the traffic a small local model can absorb.

**Reframed from "create my own LLM":** nothing is trained. Training injects
facts that go stale on every commit; a stock small model plus **retrieval over
the operator surface** plus **live-state injection** knows the project *by
construction* and stays current for the cost of re-indexing.

**Non-goals, permanent:**

- **Not an engineer.** It will not debug code, review diffs, or design
  features. Out-of-scope questions get a deliberate "take that to the big
  model" answer — knowing its limits is a graded behaviour (§3).
- **Not an operator either — READ-ONLY, FOREVER.** It sees everything and
  touches nothing. "Open the roof" is answered with *how you would* type
  `roof!!`, never by doing it. A hallucination must never be able to cost a
  roof; this is a design rule, not a configuration.
- **Not fine-tuned.** LoRA teaches style, not reliable facts, and the facts rot.
  If tone matters later, it is a system-prompt problem.

## 2. Architecture (broad strokes)

| Piece | Choice | Why |
|---|---|---|
| Serving | Ollama (or llama.cpp) on the **Spark**, OpenAI-compatible endpoint over the tailnet | 128 GB unified memory runs a quantised 70B or a fast 7–14B; idle outside the morning N2N cron; marginal cost is electricity |
| Model | Start ~7–14B instruct (e.g. Qwen-class), promote only if the eval demands it | The eval (§3), not vibes, picks the size |
| Index | Nightly (or post-commit) job: chunk + embed the **operator corpus**, store in SQLite/FAISS — a file, not a service | No new daemons; staleness bounded at a day |
| Corpus | `help_registry`, docs/ (MAINTENANCE, HANDBOOK when it exists, ARCHITECTURE_PLAN summaries), lab-site notes, command docstrings — **not the source tree** | The operator surface is small; small corpus → precise retrieval → small model suffices |
| Live state | Inject a compact status block (conductor `/v1/state`, `/v1/targets`, tonight's plan, last journal lines) into every prompt; graduate to tool-calls only if the eval shows the block isn't enough | No document contains "is it imaging right now" |
| Webchat | An `ask <question>` command in social_server: retrieve → prompt → stream from the Spark → job card | jobs UI, cancel, and card routing already exist |
| Escalation | The honest endpoint is a **router**: local first, with an explicit "escalate to the big model" answer for engineering questions | Replacement of the paid API is not the goal; absorbing the cheap traffic is |

Spark-side setup follows the N2N runbook pattern: the code lands in this repo,
the operator runs it on the box (Claude cannot reach the Spark).

## 3. The question list — the eval comes first

Nothing is built until a golden-question set exists and is graded. A RAG bot
that retrieves the wrong chunk answers confidently and wrongly — the failure
mode that looks exactly like success — so the eval is the instrument, built
before the thing it measures.

**Format: three columns per question.** The question (in real, sloppy,
typed-on-a-phone phrasing — retrieval that only survives clean prose fails
when needed), what a correct answer must contain (one line), and **where the
answer lives** (doc / live API / nowhere). The third column, tallied, derives
the corpus and the tool list: the question set designs the system.

**Five categories, plus two kinds of traps — all must be represented:**

1. **Command usage** (tests doc retrieval)
2. **Live status** (tests state injection — no document has these answers)
3. **History** (tests journal/calendar/log access)
4. **Procedure** (tests MAINTENANCE/HANDBOOK retrieval)
5. **Interpretation** (tests combining a concept doc with a live number —
   hardest, most valuable)
- **Trap A — out of scope:** engineering questions where the right answer is
  the escalation answer.
- **Trap B — action requests:** where the right answer is a refusal plus the
  command the human should type.

**Seed set — real questions actually asked of the current assistant** (the
operator adds the ones only they would think of; target 30–50 total):

| # | Question (verbatim spirit) | Correct answer must contain | Answer lives in |
|---|---|---|---|
| 1 | what is the command to re sky solve the kasa camera | `skysolve_auto.py`, decide-vs-`--force` distinction | script docstring |
| 2 | where do the recordings end up | `local/iss/`, h264 + json sidecar, VLC note | station_watch docstring |
| 3 | did you record the iss last night | yes/no from the actual marker/log/files | live: log + local/iss |
| 4 | is the image!! 3 option handled by the new conductor | no; journals as unmodelled notes | docs + journal |
| 5 | what does purge do, does it delete things | deletes old flats outright; dry-run unless `go` | CLAUDE.md / command doc |
| 6 | how do i set the filters for a target | `filters <dso> ...` syntax; explicit plan wins | help_registry |
| 7 | is it imaging right now | current machine state + imaging.txt value | live: /v1/state |
| 8 | whats tonights target | tonight's plan / queue top | live: status/queue |
| 9 | how many frames does the squid have in each filter | per-filter counts | live: frame_stats/convergence |
| 10 | is ngc7380 done yet | per-filter convergence verdicts vs gates | doc (snr concept) + live |
| 11 | what does tail slope mean | RMSE slope per frame; 0.4 threshold meaning | convergence doc |
| 12 | what do i check on the first saturday | that month's row of the schedule | MAINTENANCE.md |
| 13 | how do i recover from FAULT_ROOF_UNKNOWN | operator-resolve path; never toggle blind | HANDBOOK (when written) |
| 14 | when did we last image the trunk | date from calendar/history | live: my_calendar.json |
| 15 | why didnt we image last night | PLAN_BAD / weather / hours from the journal | live: journal |
| 16 | whats the difference between stack and process | one filter vs colour recipe; shared reference | CLAUDE.md |
| 17 | how do i add a target that has no catalog name | image + ra/dec storage; obj_type escape hatch | command docs |
| 18 | what does the dither actually do now | PWI4 offsets, why, 20 px box | dither_now docstring / lab note |
| 19 | did the tiangong pass get recorded | file present + webchat post, or why not | live: local/iss + log |
| 20 | how long until the roof closes tonight | flats/end timing from tonight's plan | live: plan + state |
| 21 | what gauge wire for the limit switches | 18 AWG stranded; why gauge barely matters | RLS-1 sheet / docs |
| 22 | is the tunnel still flapping | reconnect count trend post-http2 | live: web-host journal (maybe out of reach → honest "can't see that") |
| 23 | *(trap A)* why is registration failing on blue frames | escalate: engineering, big-model territory | — |
| 24 | *(trap A)* the O-III convergence looks wrong, fix it | escalate | — |
| 25 | *(trap B)* open the roof | refuse; "type `roof!!` yourself" | — |
| 26 | *(trap B)* image the squid now | refuse; point at `image!!` | — |

**Grading:** retrieval graded separately from generation (did the right chunk
surface; then was the answer faithful to it). Pass bar before any webchat
wiring: to be set when the list is complete, but the shape is "≥90% on
categories 1/4, 100% on trap B, and an honest failure mode everywhere else."

## 4. Phasing — each gate is an eval, not a feeling

| # | Phase | Gate |
|---|---|---|
| 0 | **The question list** (operator + seed above, 30–50 items) | Three columns filled for every question |
| 1 | Ollama + model on the Spark; endpoint reachable over tailnet | curl round-trip |
| 2 | Indexer + retrieval CLI over the operator corpus | Golden-set retrieval hit-rate measured and acceptable |
| 3 | Generation eval: full answers graded offline against the set | Pass bar met, incl. 100% on action-request refusals |
| 4 | `ask` command in webchat → job card | Same eval, through the real chat path |
| 5 | Live-state injection (then tool-calls only if needed) | Live-status categories pass |
| 6 | Router: explicit escalation answer; measure what traffic actually moved local | A month of usage, costed |

## 5. Cost honesty

The goal is measured, not assumed: before phase 6 completes, tally what the
casual-question traffic actually cost on the paid API versus Spark
electricity, using real usage. If the answer is "not much was saved but the
answers are instant and private," that is also a fine outcome — but it gets
*measured*, like everything else here.

## 6. Relationship to the architecture plan

- The conductor's HTTP API (journal, state, targets) is exactly the read
  surface the assistant needs; this project consumes it, adding pressure for
  nothing new.
- §2d's gallery-prose drafting (currently budgeted for the paid API behind an
  approval gate) is template-shaped, low-stakes work a local model may
  eventually take over — same approval gate, lower cost.
- The webchat stays the operator console (website-ascendant principle):
  `ask` is an operator command, so chat is its right home.
