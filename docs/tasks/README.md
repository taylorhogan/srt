# Scheduled tasks: creating and checking them

The observatory PC runs its periodic checks as Windows Scheduled Tasks and
the Spark runs its as cron entries (ADR 0009: publishers and Spark jobs are
cron jobs, not daemons). This file is the runbook for creating one; the
exported definitions (`schtasks /Query /XML`) are to land beside it
(docs/AUTOMATION_PLAN.md F4).

## IrisConvergenceRefresh (PC, daily 09:15) -- added 2026-10-29

Recomputes the convergence curves of every target imaged last night whose
`local/convergence.json` entry is older than its newest frame, so the noon
plan's auto-stop judges current numbers (`scripts/convergence_refresh.py`,
docs/AUTOMATION_PLAN.md A1).

Open a Command Prompt as the `iriso` user and run:

```
cd C:\Users\iriso\Documents\development\srt
git pull
schtasks /Create /TN IrisConvergenceRefresh /SC DAILY /ST 09:15 ^
  /TR "C:\Users\iriso\Documents\development\srt\scripts\convergence_refresh.cmd" ^
  /RL LIMITED /F
```

Check it, run it once by hand, and read its log:

```
schtasks /Query /TN IrisConvergenceRefresh /V /FO LIST
schtasks /Run /TN IrisConvergenceRefresh
type local\convergence_refresh.log
```

The first run says "nothing stale" when last night's end-of-night SNR loop
did its job, or lists the targets it recomputed (a few minutes per target).

Notes:

* The `git pull` matters: the `.cmd` and the script arrived in release at
  7f157df. The web chat `update` command does the same pull and relaunches
  the three children (so the conductor's alert hook loads), but NOT the
  `start_srt` parent that supervises them: the new supervision loop needs
  one stop and start of `start_srt` by hand, or a reboot (`start_srt.bat`
  fast-forwards to release and launches it fresh).
* To run while nobody is logged in, add `/RU iriso /RP *` to the create
  command (it prompts for the password). Match what
  `schtasks /Query /TN IrisMorningCheck /V /FO LIST` shows under
  "Run As User" and "Logon Mode"; the other Iris tasks were made that way.

## morning_watch (Spark, daily 10:00) -- added 2026-10-29

Pushes when the Spark's morning run did not happen, was skipped for the
disk, or did not render (`scripts/morning_watch.py`, reads
`local/spark_render/last_run.json`; docs/AUTOMATION_PLAN.md B8). On the
Spark, `crontab -e` and add:

```
# 2026-10-29: did the morning run happen and render? Reads local/spark_render/last_run.json
# (written by morning_trigger); pushes when it is missing, skipped or non-zero.
0 10 * * * /home/taylor/Documents/srt/scripts/morning_watch.py >> /tmp/morning_watch.log 2>&1
```

## The tasks that exist (for the export)

PC: IrisMorningCheck 09:00 (`morning_check.cmd`), IrisNoonWatch 12:30
(`noon_watch.cmd`), IrisConvergenceRefresh 09:15, IrisBackupState,
IrisSkyMonitor and IrisAllSkyMonitor (every 5 min), IrisLiveSkymap,
IrisForecastLog (hourly), IrisCIWatch (every 5 min), IrisSkysolveAuto.
Spark cron: `morning_trigger.py` 06:30, `sync_state_to_spark.bsh` 11:40,
`morning_watch.py` 10:00.

## publish_poll (Spark, every 15 minutes 08:00-22:00) -- added 2026-10-30

Acts on `approve` / `reject` (docs/AUTOMATION_PLAN.md D4): reads the PC's
`local/publish_decisions.json` over ssh and, with `publish.gate = live`,
publishes an approved draft to the site, announces through the PC and journals
`PUBLISH_DONE`. In `shadow` (the default) it says in the chat what it would do,
once per draft. Add to the Spark's crontab:

```
*/15 8-22 * * * /home/taylor/Documents/srt/scripts/publish_poll.py >> /tmp/publish_poll.log 2>&1
```

`scripts/publish_poll.py --dry-run` prints what it would do and touches nothing.

## suggest_targets (Spark, Mondays 12:30) -- added 2026-10-30

Posts the week's queue suggestions to the web chat (docs/AUTOMATION_PLAN.md
E2): the top three candidates from configs/candidates.json with reasons and
"`image <name>` queues one". Nothing is queued. Add to the Spark's crontab:

```
30 12 * * 1 /home/taylor/Documents/srt/scripts/suggest_targets.py --post >> /tmp/suggest.log 2>&1
```

The chat `suggest [N]` command answers the same on demand from the PC.

## Exporting the PC's tasks (F4) -- do this after creating or changing one

Each task's definition belongs in this folder so a rebuilt PC can recreate
it. On the PC:

```
for %t in (IrisMorningCheck IrisNoonWatch IrisConvergenceRefresh IrisBackupState IrisSkyMonitor IrisAllSkyMonitor IrisLiveSkymap IrisForecastLog IrisCIWatch IrisSkysolveAuto) do schtasks /Query /TN %t /XML > docs\tasks\%t.xml
```

then commit the XML files. Recreate one with
`schtasks /Create /TN IrisMorningCheck /XML docs\tasks\IrisMorningCheck.xml`.

| Task | Script | When |
|---|---|---|
| IrisMorningCheck | scripts/morning_check.cmd | 09:00 |
| IrisConvergenceRefresh | scripts/convergence_refresh.cmd | 09:15 |
| IrisNoonWatch | scripts/noon_watch.cmd | 12:30 |
| IrisBackupState | scripts/backup_state.cmd | daily |
| IrisSkyMonitor | scripts/sky_monitor.bat | every 5 min |
| IrisAllSkyMonitor | scripts/allsky_monitor.bat | every 5 min |
| IrisLiveSkymap | scripts/live_skymap.bat | every few min |
| IrisForecastLog | scripts/forecast_log.bat | hourly |
| IrisCIWatch | scripts/ci_watch.cmd | every 5 min |
| IrisSkysolveAuto | scripts/skysolve_auto.cmd | nightly |

The Spark's crontab is docs/tasks/spark_crontab.txt (`crontab -l` pasted;
restore with `crontab docs/tasks/spark_crontab.txt`). tests/test_docs_tasks.py
checks that every script named in either exists.
