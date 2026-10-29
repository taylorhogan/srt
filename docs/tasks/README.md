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
  7f157df. The web chat `update` command does the same pull plus a restart
  of `start_srt`, which is also wanted so the conductor's alert hook and the
  new process supervision load.
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
