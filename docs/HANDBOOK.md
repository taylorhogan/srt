# HANDBOOK -- running Iris when the owner is not there

Written 2026-10-30 (docs/AUTOMATION_PLAN.md F4, ADR 0011: months unattended,
heir-operable). This is the page to read when the phone rings or the chat
goes quiet. It says what each alert means, what to type, and what the
machine will do next. Facts about money, accounts and hardware live in
SUCCESSION.md; inspections in MAINTENANCE.md; the design in ARCHITECTURE.md.

Everything you type goes in the web chat (http://100.95.7.19:8095 over
Tailscale). `help` lists every command; `help <command>` explains one.

## 1. The normal day, so you know what silence means

| When | Who | What you see |
|---|---|---|
| 06:30 | Spark | morning_trigger waits for the flats, syncs the night's frames, renders |
| 07-09 | Spark | one Pushover picture per rendered target (the learned composite) |
| 09:00 | PC | IrisMorningCheck: a push only if something is wrong (roof, plugs, disk) |
| 09:15 | PC | IrisConvergenceRefresh: recomputes any target whose frames outran its record |
| 10:00 | Spark | morning_watch: a push only if the morning run did not happen or failed |
| 11:40 | Spark | the PC's state is mirrored to the Spark |
| 12:00 | PC | the conductor's noon plan; the chat shows the night's slots, the filter table, the auto-stop list |
| 12:30 | PC | IrisNoonWatch: a push only if the noon plan did not finish or the chat is down |
| Mon 12:30 | Spark | the week's queue suggestions |
| sunset | PC | the sunset check confirms or regenerates; the roof opens; N.I.N.A runs |
| dawn | PC | park, close, flats; end.py's push says the roof is confirmed closed |

**A quiet day is a good day.** The watchdogs push only on failure. If you want
proof of life, type `status`.

## 2. The alerts, and what to do

Pushover priority 2 repeats on the phone until you acknowledge it. Priority 1
is one push. Priority 0 is a notice.

| Push (priority) | Meaning | What to do |
|---|---|---|
| `Iris HOLD: ESTOP after ...` (2) | `stop!` ran, or something fired the emergency stop. The scope is parked, the roof closed. | Look (`status`, the picture in the push). When satisfied, `resolve!`. |
| `Iris HOLD: FAULT_ROOF_UNKNOWN after ...` (2) | The roof's position is not trusted: a stall, a timeout, a restart mid-move, or the two cameras disagree. | Look at the roof, with your eyes or `status` / `roof!!` only if you are sure. If it is open, close it by hand (section 5). Then `resolve!`. |
| `Iris HOLD: SAFE_HOLD after ...` (2) | The observatory was marked UNSAFE. | Fix whatever was unsafe, then `safe!`. |
| `Iris: the pre-flight checks failed; no roof open tonight` (2) | goforimagecheck refused: a plug, the mount, the cameras. | `status`, fix the item named, nothing to type; tomorrow's noon plans again. |
| `Iris: the night plan failed twice` (2) | The planner crashed or hung at noon and at the retry. | Read iris.log around 12:00; `replan` once the cause is fixed. |
| `Iris: N.I.N.A disappeared mid-run` (2) | The capture process died. The night is ending normally. | Nothing tonight. In the morning check N.I.N.A opens on the PC. |
| `No imaging tonight: disk: N GB free` (2) | The image drive is below the refuse floor (25 GB on the PC, 200 GB on the Spark). | Free space: `purge` (flats) and `purge bad go` on the PC; `scripts/spark_prune.py --go` on the Spark. Raw lights are never pruned. |
| `ROOF NOT CONFIRMED CLOSED` (2) | end.py could not confirm the roof shut at dawn. | Look. Close by hand if open (section 5). |
| `Iris is still in <hold> since ...` (1, daily at noon) | The hold is still there; no plan was made. | The hold's row above. |
| `Iris noon watch FAILED` (1) | The noon check never finished, or the chat is not answering. | On the PC, restart start_srt (section 6). |
| `Iris morning watch` (1) | The Spark's morning run is missing, skipped or failed. | On the Spark, `tail /tmp/morning_trigger.log`; run `scripts/morning_trigger.py` by hand. |
| `<child> exited (code); relaunching` (1) | start_srt restarted a dead scheduler or conductor. | Nothing. Twice a day: read iris.log. |
| `... give up` (2) | A process crashed five times in ten minutes. | Restart start_srt (section 6) after reading the log. |
| `FINAL <dso> <recipe> -- QC PASS/FAIL` (0) | A converged target's final render and its verdict. | A card follows in the chat (section 4). |

## 3. The holds, and the commands that release them

The conductor (the process that owns the roof and the mount) has three hold
states. Nothing automatic leaves them; that is the point.

| State | How it got there | Release |
|---|---|---|
| `ESTOP` | `stop!` (yours or a watchdog's) | `resolve!` after looking |
| `FAULT_ROOF_UNKNOWN` | a roof move that stalled, timed out, or was interrupted; the two roof cameras disagree | `resolve!` after you have SEEN the roof where the sensors say it is |
| `SAFE_HOLD` | `unsafe` was declared | `safe!` |

`resolve!` is a statement that you looked. If you are not sure the roof is
where `status` says, do not type it. A night will not run in a hold, and the
reminder repeats daily; that is cheaper than a roof driven into a scope.

**Emergency stop:** `stop!` from anywhere. It kills N.I.N.A, parks the scope and
closes the roof, asking the conductor at each step. It may park a scope whose
roof it cannot see, under the evidence rule in CLAUDE.md.

## 4. The queue and the pictures

| Command | Use |
|---|---|
| `image <dso>` | queue a target (type stored from the catalogue) |
| `db` | the queue; `dbc <id>` marks one completed by hand |
| `tonight` | tonight's grid |
| `filters <dso> Ha=1 O-III=2 for 2` | a filter ratio for N nights; `filters <dso>` shows; `clear` removes |
| `hold <dso>` / `hold <dso> off` | keep the planner off a target |
| `moon <dso> any|avoid|dark` | the moon rule |
| `maxhours <dso> N` | a short block a night (monitors) |
| `type <dso> galaxy|nebula` | the type the filter plan reads |
| `reopen <dso> [for N]` | undo auto-stop: back to waiting, auto-stop held off N nights |
| `suggest [N]` | what is worth adding this month |
| `drafts`, `approve <dso>`, `reject <dso> [why]` | the publish cards (below) |
| `lightcurve <dso>` | a monitored star's nightly points (runs itself after each night) |
| `aavso <dso> [filter]` | the AAVSO WebObs file for a monitor, on the PC under `local/lightcurves/aavso/`; upload it by hand at app.aavso.org/webobs (observer code `aavso.obscode`) |
| `replan` | re-run the noon decision now |
| `update` | pull the release and restart the children (not start_srt itself) |

**Auto-stop** retires a target (`converged`) when both its convergence
curve is flat and the advisor's frame goals are met, or when it has shot
twice its goals' worth of frames (the cap: the reason starts "cap:" and
you get one low-priority push); a light-curve monitor when it can draw
its curve. The noon chat lists them. `reopen` is the undo.

**Publishing.** When a converged target's final render passes its quality
check, a card appears in the chat: the picture, the entry's text, who wrote
it, and the auto-publish date. `approve <dso>` publishes it on the Spark's
next 15-minute poll and announces to subscribers; `reject <dso> why` drops
it. Silence publishes after `publish.auto_approve_days` once
`publish.gate` is `live`. A card marked "facts only" was written without
the language model: approve it or wait for the key (SUCCESSION.md).

## 5. Hands on the hardware

- **Closing the roof by hand:** power the roof motor plug (`status` names it)
  only when the scope is parked; the gate opener has a manual release.
  Never move the roof with the scope unparked, never move the scope with
  the roof closed. The conductor enforces both for software; you are the
  only guard for your own hands.
- **Cold start of the PC:** scripts/start_srt.bat (the logon task) fetches
  the release branch, waits for Pegasus Unity, and starts start_srt, which
  runs the chat, the scheduler and the conductor and relaunches any that die.
- **Rolling back:** `git push origin <good sha>:release` from any clone, then
  `update` in the chat. CI fast-forwards release only when the tests pass,
  so release is always a commit that was green.
- **N.I.N.A sequences** are generated from the template named in
  `cfg["nina"]["sequence_input"]`; edit the template, never the generated
  file. `sequence <dso>` writes one by hand.

## 6. When the chat does not answer

Every report goes out through the chat, so a dead chat is silent. The 12:30
and 10:00 watchdogs push directly. On the PC: Task Manager, end `python`
processes of start_srt, run scripts/start_srt.bat. Then `status` in the chat.

## 7. The flags that change behaviour (configs/config_public.py)

| Key | Values | What it does |
|---|---|---|
| `planner.auto_stop` | `off` / `shadow` / `live` | list finished targets at noon / retire them |
| `planner.auto_stop_cap` | multiple, 2.0 | retire a target that has shot this many times its frame goals; 0 = never |
| `nina.filter_advisor` | `off` / `shadow` / `default` | show the advised filter ratio / use it |
| `publish.final_mode` | `off` / `shadow` | render a converged target's final |
| `publish.gate` | `off` / `shadow` / `live` | cards and decisions / publish on approve or timer |
| `publish.auto_approve_days` | days | silence publishes after this (gate live) |
| `alerts.enabled` | bool | the bell |
| `conductor.drives_checks` | bool | the conductor makes the noon and sunset decisions (on) |

A flag change needs `update` (the children re-read the config at start).

## 8. Where things are

| | PC (Windows, `iriso`) | Spark (Linux, `taylor`) |
|---|---|---|
| repo | C:\Users\iriso\Documents\development\srt | /home/taylor/Documents/srt |
| frames | C:\Users\iriso\Documents\N.I.N.A\Targets | /home/taylor/Desktop/Targets (synced at 06:30) |
| queue | local/my_instructions.json | the 11:40 mirror under ~/Desktop/iris_state/live/local/ |
| journal | local/journal/<date>.jsonl | (mirrored) |
| logs | iris.log | /tmp/morning_trigger.log, local/spark_render/morning.log, /tmp/publish_poll.log |
| site checkout | Documents\development\taylorhogan.github.io | ~/Documents/development/taylorhogan.github.io |
| scheduled jobs | docs/tasks/README.md (schtasks) | docs/tasks/spark_crontab.txt (cron) |
